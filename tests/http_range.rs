#![cfg(feature = "http")]

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use adyar::build::{BuildOptions, build_pack};
use adyar::model::AccessClass;
use adyar::reader::{HttpRangeReader, ReadAt};
use adyar::search::{SearchEngine, SearchMode, SearchOptions};
use tempfile::TempDir;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/docs-v1")
}

fn build_fixture(temp: &TempDir) -> Vec<u8> {
    let output = temp.path().join("remote.annpack");
    build_pack(&BuildOptions {
        input: fixture(),
        output: output.clone(),
        name: "vendor-docs".into(),
        version: "1.0.0".into(),
        description: None,
        source_revision: Some("git:v1".into()),
        base_url: Some("https://vendor.example/docs/v1".into()),
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
        input_format: adyar::ingest::InputFormat::Auto,
    })
    .unwrap();
    std::fs::read(output).unwrap()
}

struct TestServer {
    url: String,
    stop: Arc<AtomicBool>,
    requests: Arc<AtomicUsize>,
    range_requests: Arc<AtomicUsize>,
    body_bytes: Arc<AtomicUsize>,
    thread: Option<thread::JoinHandle<()>>,
}

impl TestServer {
    fn start(body: Vec<u8>, ignore_ranges: bool, wrong_content_range: bool) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let requests = Arc::new(AtomicUsize::new(0));
        let range_requests = Arc::new(AtomicUsize::new(0));
        let body_bytes = Arc::new(AtomicUsize::new(0));
        let thread_stop = stop.clone();
        let thread_requests = requests.clone();
        let thread_ranges = range_requests.clone();
        let thread_bytes = body_bytes.clone();
        let thread = thread::spawn(move || {
            while !thread_stop.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        prepare_accepted(&stream);
                        serve_connection(
                            stream,
                            &body,
                            ignore_ranges,
                            wrong_content_range,
                            &thread_requests,
                            &thread_ranges,
                            &thread_bytes,
                        )
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                    }
                    Err(_) => break,
                }
            }
        });
        Self {
            url: format!("http://{address}/knowledge.annpack"),
            stop,
            requests,
            range_requests,
            body_bytes,
            thread: Some(thread),
        }
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(
            self.url
                .trim_start_matches("http://")
                .split('/')
                .next()
                .unwrap(),
        );
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap();
        }
    }
}

/// Put an accepted socket into blocking mode with finite timeouts.
///
/// The listener is non-blocking so the accept loop can poll for shutdown. A
/// socket accepted from a non-blocking listener is not guaranteed to be
/// blocking itself, and when it is not, the first read returns `WouldBlock`
/// before the request has arrived. The handlers below treat a read error as
/// "client hung up" and close without responding, so the client sees an
/// unexpected EOF instead of its response -- intermittently, depending purely
/// on whether the bytes had landed yet. Timeouts keep a wedged socket from
/// hanging the single-threaded accept loop forever.
fn prepare_accepted(stream: &TcpStream) {
    stream.set_nonblocking(false).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    stream
        .set_write_timeout(Some(Duration::from_secs(10)))
        .unwrap();
}

fn serve_connection(
    mut stream: TcpStream,
    body: &[u8],
    ignore_ranges: bool,
    wrong_content_range: bool,
    requests: &AtomicUsize,
    range_requests: &AtomicUsize,
    body_bytes: &AtomicUsize,
) {
    let clone = stream.try_clone().unwrap();
    let mut reader = BufReader::new(clone);
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() || request_line.is_empty() {
        return;
    }
    let mut range = None;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() || line == "\r\n" || line.is_empty() {
            break;
        }
        if let Some(value) = line
            .strip_prefix("Range: bytes=")
            .or_else(|| line.strip_prefix("range: bytes="))
        {
            let value = value.trim();
            if let Some((start, end)) = value.split_once('-') {
                range = Some((
                    start.parse::<usize>().unwrap(),
                    end.parse::<usize>().unwrap(),
                ));
            }
        }
    }
    requests.fetch_add(1, Ordering::SeqCst);
    if request_line.starts_with("HEAD ") {
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\nETag: \"fixture-v1\"\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .unwrap();
        return;
    }
    if ignore_ranges {
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nETag: \"fixture-v1\"\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .unwrap();
        stream.write_all(body).unwrap();
        body_bytes.fetch_add(body.len(), Ordering::SeqCst);
        return;
    }
    let (start, end) = range.unwrap();
    range_requests.fetch_add(1, Ordering::SeqCst);
    let response = &body[start..=end];
    let content_start = if wrong_content_range {
        start + 1
    } else {
        start
    };
    write!(
        stream,
        "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {}-{}/{}\r\nAccept-Ranges: bytes\r\nETag: \"fixture-v1\"\r\nConnection: close\r\n\r\n",
        response.len(),
        content_start,
        end,
        body.len()
    )
    .unwrap();
    stream.write_all(response).unwrap();
    body_bytes.fetch_add(response.len(), Ordering::SeqCst);
}

#[test]
fn remote_search_reads_a_bounded_number_of_narrow_ranges() {
    let temp = TempDir::new().unwrap();
    let pack = build_fixture(&temp);
    let server = TestServer::start(pack.clone(), false, false);
    let source = Arc::new(HttpRangeReader::open(server.url.clone()).unwrap());
    let engine = SearchEngine::open_source(source).unwrap();
    let response = engine
        .search(
            "AP-104",
            &SearchOptions {
                mode: SearchMode::Lexical,
                limit: 1,
                ..SearchOptions::default()
            },
        )
        .unwrap();
    assert!(response.results[0].text.contains("API key has expired"));

    // Request count is a round-trip budget, not an efficiency claim: resolving
    // a term through the block-addressable index costs one more request than
    // downloading the whole index did, and far fewer bytes. Bytes are what the
    // range design is actually for, so the byte bound is the strict one and
    // the request bound is a loose ceiling that catches a runaway fetch loop.
    let requests = server.range_requests.load(Ordering::SeqCst);
    assert!(
        (1..=12).contains(&requests),
        "expected a small bounded number of range reads, got {requests}"
    );
    assert_eq!(server.requests.load(Ordering::SeqCst), requests + 1); // one HEAD
    let bytes = server.body_bytes.load(Ordering::SeqCst);
    assert!(
        bytes < pack.len(),
        "a ranged search must transfer less than the artifact"
    );
    assert!(bytes < 300 * 1024);
}

#[test]
fn range_ignoring_server_is_rejected() {
    let temp = TempDir::new().unwrap();
    let pack = build_fixture(&temp);
    let server = TestServer::start(pack, true, false);
    let reader = HttpRangeReader::open(server.url.clone()).unwrap();
    let mut header = [0_u8; 128];
    assert!(reader.read_exact_at(0, &mut header).is_err());
}

#[test]
fn incorrect_content_range_is_rejected() {
    let temp = TempDir::new().unwrap();
    let pack = build_fixture(&temp);
    let server = TestServer::start(pack, false, true);
    let reader = HttpRangeReader::open(server.url.clone()).unwrap();
    let mut header = [0_u8; 128];
    assert!(reader.read_exact_at(0, &mut header).is_err());
}
