#![cfg(feature = "http")]

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use annpack::build::{BuildOptions, build_pack};
use annpack::model::AccessClass;
use annpack::oci::{RegistryCredentials, pull_pack, push_pack};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/docs-v1")
}

fn build_fixture(temp: &TempDir) -> PathBuf {
    let output = temp.path().join("registry.annpack");
    build_pack(&BuildOptions {
        input: fixture(),
        output: output.clone(),
        name: "registry-docs".into(),
        version: "1.0.0".into(),
        description: None,
        source_revision: Some("git:registry-test".into()),
        base_url: None,
        created_at: None,
        license: None,
        access: AccessClass::Public,
        redistributable: None,
        policy_expires_at: None,
        policy_url: None,
        policy_override: None,
        vector_input: None,
        expansion_input: None,
        splade_input: None,
        target_chars: 1_200,
        max_chars: 2_400,
        input_format: annpack::ingest::InputFormat::Auto,
    })
    .unwrap();
    output
}

#[derive(Default)]
struct RegistryState {
    blobs: HashMap<String, Vec<u8>>,
    manifests: HashMap<String, Vec<u8>>,
}

struct RegistryServer {
    reference: String,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl RegistryServer {
    fn start(require_auth: bool) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let auth_realm = require_auth.then(|| format!("http://{address}/token"));
        let stop = Arc::new(AtomicBool::new(false));
        let state = Arc::new(Mutex::new(RegistryState::default()));
        let upload_id = Arc::new(AtomicUsize::new(0));
        let thread_stop = stop.clone();
        let thread_state = state.clone();
        let thread_upload_id = upload_id.clone();
        let thread = thread::spawn(move || {
            while !thread_stop.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((stream, _)) => serve_registry_connection(
                        stream,
                        &thread_state,
                        &thread_upload_id,
                        auth_realm.as_deref(),
                    ),
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                    }
                    Err(_) => break,
                }
            }
        });
        Self {
            reference: format!("http://{address}/test/docs:1.0.0"),
            stop,
            thread: Some(thread),
        }
    }
}

impl Drop for RegistryServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let address = self
            .reference
            .trim_start_matches("http://")
            .split('/')
            .next()
            .unwrap();
        let _ = TcpStream::connect(address);
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap();
        }
    }
}

fn serve_registry_connection(
    mut stream: TcpStream,
    state: &Mutex<RegistryState>,
    upload_id: &AtomicUsize,
    auth_realm: Option<&str>,
) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() || request_line.is_empty() {
        return;
    }
    let mut content_length = 0_usize;
    let mut authorization = None;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() || line == "\r\n" || line.is_empty() {
            break;
        }
        if let Some(value) = line
            .to_ascii_lowercase()
            .strip_prefix("content-length:")
            .map(str::trim)
        {
            content_length = value.parse().unwrap();
        }
        if let Some(value) = line
            .strip_prefix("Authorization:")
            .or_else(|| line.strip_prefix("authorization:"))
        {
            authorization = Some(value.trim().to_string());
        }
    }
    let mut body = vec![0_u8; content_length];
    reader.read_exact(&mut body).unwrap();
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap();
    let target = parts.next().unwrap();

    if method == "GET" && target.starts_with("/token") {
        if !authorization
            .as_deref()
            .is_some_and(|value| value.starts_with("Basic "))
        {
            write_response(&mut stream, "403 Forbidden", &[], &[]);
            return;
        }
        write_response(
            &mut stream,
            "200 OK",
            &[("Content-Type", "application/json")],
            br#"{"token":"registry-test-token"}"#,
        );
        return;
    }
    if let Some(realm) = auth_realm
        && authorization.as_deref() != Some("Bearer registry-test-token")
    {
        let challenge = format!(
            "Bearer realm=\"{realm}\",service=\"registry.test\",scope=\"repository:test/docs:pull,push\""
        );
        write_response(
            &mut stream,
            "401 Unauthorized",
            &[("WWW-Authenticate", &challenge)],
            &[],
        );
        return;
    }

    if method == "POST" && target == "/v2/test/docs/blobs/uploads/" {
        let id = upload_id.fetch_add(1, Ordering::SeqCst);
        write_response(
            &mut stream,
            "202 Accepted",
            &[("Location", &format!("/upload/{id}"))],
            &[],
        );
        return;
    }
    if method == "PUT" && target.starts_with("/upload/") {
        let parsed = url::Url::parse(&format!("http://registry.invalid{target}")).unwrap();
        let digest = parsed
            .query_pairs()
            .find_map(|(key, value)| (key == "digest").then(|| value.into_owned()))
            .unwrap();
        assert_eq!(digest, format!("sha256:{:x}", Sha256::digest(&body)));
        state.lock().unwrap().blobs.insert(digest.clone(), body);
        write_response(
            &mut stream,
            "201 Created",
            &[("Docker-Content-Digest", &digest)],
            &[],
        );
        return;
    }
    if method == "PUT" && target == "/v2/test/docs/manifests/1.0.0" {
        let digest = format!("sha256:{:x}", Sha256::digest(&body));
        state.lock().unwrap().manifests.insert("1.0.0".into(), body);
        write_response(
            &mut stream,
            "201 Created",
            &[("Docker-Content-Digest", &digest)],
            &[],
        );
        return;
    }
    // Any manifest reference — tag or digest — is answered with the one stored
    // manifest. A digest-pinned client must therefore check the bytes itself
    // rather than trusting that the registry honoured the pin.
    if method == "GET" && target.starts_with("/v2/test/docs/manifests/") {
        let body = state.lock().unwrap().manifests["1.0.0"].clone();
        write_response(
            &mut stream,
            "200 OK",
            &[("Content-Type", "application/vnd.oci.image.manifest.v1+json")],
            &body,
        );
        return;
    }
    if method == "GET" && target.starts_with("/v2/test/docs/blobs/") {
        let digest = target.trim_start_matches("/v2/test/docs/blobs/");
        let body = state.lock().unwrap().blobs[digest].clone();
        write_response(
            &mut stream,
            "200 OK",
            &[("Content-Type", "application/octet-stream")],
            &body,
        );
        return;
    }
    write_response(&mut stream, "404 Not Found", &[], &[]);
}

fn write_response(stream: &mut TcpStream, status: &str, headers: &[(&str, &str)], body: &[u8]) {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    )
    .unwrap();
    for (name, value) in headers {
        write!(stream, "{name}: {value}\r\n").unwrap();
    }
    write!(stream, "\r\n").unwrap();
    stream.write_all(body).unwrap();
}

#[test]
fn push_and_pull_round_trip_through_distribution_api() {
    let temp = TempDir::new().unwrap();
    let source = build_fixture(&temp);
    let server = RegistryServer::start(false);
    let pushed = push_pack(&source, &server.reference, None).unwrap();
    assert!(pushed.pack_digest.starts_with("sha256:"));

    let output = temp.path().join("pulled.annpack");
    let pulled = pull_pack(&server.reference, &output, None, false).unwrap();
    assert_eq!(pushed.pack_root, pulled.pack_root);
    assert_eq!(
        std::fs::read(source).unwrap(),
        std::fs::read(output).unwrap()
    );
}

#[test]
fn digest_pinned_pull_verifies_the_received_manifest_bytes() {
    let temp = TempDir::new().unwrap();
    let source = build_fixture(&temp);
    let server = RegistryServer::start(false);
    let pushed = push_pack(&source, &server.reference, None).unwrap();
    let registry = server
        .reference
        .trim_start_matches("http://")
        .split('/')
        .next()
        .unwrap()
        .to_string();

    // repository@sha256:<expected-manifest-digest> resolves and installs.
    let pinned = format!("http://{registry}/test/docs@{}", pushed.manifest_digest);
    let output = temp.path().join("pinned.annpack");
    let pulled = pull_pack(&pinned, &output, None, false).unwrap();
    assert_eq!(pulled.manifest_digest, pushed.manifest_digest);
    assert_eq!(pulled.pack_root, pushed.pack_root);

    // The registry serves the same manifest for every reference. Pinning a
    // different digest must fail on the hash of the received bytes.
    let wrong_digest = format!("sha256:{:x}", Sha256::digest(b"not-the-manifest"));
    let mismatched = format!("http://{registry}/test/docs@{wrong_digest}");
    let error = pull_pack(
        &mismatched,
        &temp.path().join("mismatch.annpack"),
        None,
        false,
    )
    .expect_err("a substituted manifest must be rejected");
    assert!(
        error.to_string().contains(&wrong_digest),
        "unexpected error: {error}"
    );
    assert!(!temp.path().join("mismatch.annpack").exists());
}

#[test]
fn bearer_challenge_round_trip_uses_external_credentials() {
    let temp = TempDir::new().unwrap();
    let source = build_fixture(&temp);
    let server = RegistryServer::start(true);
    let credentials = RegistryCredentials {
        username: "publisher".into(),
        password: "secret".into(),
    };
    push_pack(&source, &server.reference, Some(credentials.clone())).unwrap();
    let output = temp.path().join("authenticated-pull.annpack");
    pull_pack(&server.reference, &output, Some(credentials), false).unwrap();
    assert_eq!(
        std::fs::read(source).unwrap(),
        std::fs::read(output).unwrap()
    );
}
