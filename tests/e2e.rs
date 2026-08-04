use std::fs;
use std::io::{BufReader, Cursor};
use std::path::{Path, PathBuf};

use annpack::build::{BuildOptions, VectorInput, build_pack, build_pack_bytes};
use annpack::delta::{apply_delta, create_delta};
use annpack::format::{PackReader, SectionType};
use annpack::ingest::{IngestOptions, ingest_directory};
use annpack::mcp::McpServer;
use annpack::model::{AccessClass, EmbeddingProfile};
use annpack::search::{SearchEngine, SearchMode, SearchOptions};
use annpack::signing::{generate_keypair, sign_pack, verify_signatures};
use serde_json::{Value, json};
use tempfile::TempDir;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name)
}

fn options(input: PathBuf, output: PathBuf, version: &str) -> BuildOptions {
    BuildOptions {
        input,
        output,
        name: "vendor-docs".into(),
        version: version.into(),
        description: Some("Version-exact integration fixture".into()),
        source_revision: Some(format!("git:v{version}")),
        base_url: Some(format!("https://vendor.example/docs/v{version}")),
        created_at: None,
        license: Some("Apache-2.0".into()),
        access: AccessClass::Public,
        redistributable: Some(true),
        policy_expires_at: None,
        policy_url: None,
        policy_override: None,
        vector_input: None,
        expansion_input: None,
        splade_input: None,
        target_chars: 1_200,
        max_chars: 2_400,
        input_format: annpack::ingest::InputFormat::Auto,
    }
}

#[test]
fn deterministic_build_and_version_exact_retrieval() {
    let temp = TempDir::new().unwrap();
    let v1_a = options(
        fixture("docs-v1"),
        temp.path().join("v1-a.annpack"),
        "1.0.0",
    );
    let v1_b = options(
        fixture("docs-v1"),
        temp.path().join("v1-b.annpack"),
        "1.0.0",
    );
    build_pack(&v1_a).unwrap();
    build_pack(&v1_b).unwrap();
    assert_eq!(
        fs::read(&v1_a.output).unwrap(),
        fs::read(&v1_b.output).unwrap()
    );

    let v2 = options(fixture("docs-v2"), temp.path().join("v2.annpack"), "2.0.0");
    build_pack(&v2).unwrap();
    let search_options = SearchOptions {
        mode: SearchMode::Lexical,
        ..SearchOptions::default()
    };
    let result_v1 = SearchEngine::open_path(&v1_a.output)
        .unwrap()
        .search("What does AP-104 mean?", &search_options)
        .unwrap();
    let result_v2 = SearchEngine::open_path(&v2.output)
        .unwrap()
        .search("What does AP-104 mean?", &search_options)
        .unwrap();
    assert!(result_v1.results[0].text.contains("API key has expired"));
    assert!(result_v2.results[0].text.contains("unsupported algorithm"));
    assert_eq!(result_v1.pack.version, "1.0.0");
    assert_eq!(result_v2.pack.version, "2.0.0");
    assert_ne!(result_v1.pack.root_hash, result_v2.pack.root_hash);
    assert!(result_v1.pack.conformance.core_conformant);
    assert_eq!(
        result_v1.pack.conformance.core_profile,
        "annpack-core-v1.0-draft"
    );
    assert!(result_v1.pack.conformance.extensions.is_empty());
    assert_eq!(result_v1.results[0].evidence.schema, "annpack-evidence-v1");
    assert_eq!(
        result_v1.results[0].evidence.pack_root,
        result_v1.pack.root_hash
    );
    assert_eq!(
        result_v1.results[0].evidence.passage_id,
        result_v1.results[0].passage_id
    );
    assert_ne!(
        result_v1.results[0].evidence.passage_hash,
        result_v1.results[0].passage_id
    );
    assert_eq!(result_v1.results[0].evidence.passage_hash.len(), 64);
    assert_eq!(result_v1.pack.publisher.status, "unsigned");
}

#[test]
fn corruption_fails_integrity_without_panicking() {
    let temp = TempDir::new().unwrap();
    let build = options(
        fixture("docs-v1"),
        temp.path().join("valid.annpack"),
        "1.0.0",
    );
    build_pack(&build).unwrap();
    let reader = PackReader::open_path(&build.output).unwrap();
    let passage = reader
        .entries
        .iter()
        .find(|entry| entry.section_type.name() == "passage_data")
        .unwrap();
    let mut bytes = fs::read(&build.output).unwrap();
    bytes[passage.offset as usize] ^= 0x01;
    let corrupt = temp.path().join("corrupt.annpack");
    fs::write(&corrupt, bytes).unwrap();
    let corrupt_reader = PackReader::open_path(corrupt).unwrap();
    assert!(corrupt_reader.verify_all().is_err());
}

#[test]
fn a_build_never_emits_a_pack_its_own_reader_would_reject() {
    // Passage identifiers derive from document, heading path, and normalized
    // text. A document repeating the same body under the same heading twice
    // produces two passages with the same identifier, and `SearchEngine::open`
    // refuses a pack with duplicate passage IDs. The build must fail first,
    // naming the source, rather than writing an unreadable artifact.
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("duplicate-source");
    fs::create_dir_all(&input).unwrap();

    // Long enough that the two paragraphs cannot be merged into one chunk, so
    // they survive chunking as two passages with identical content.
    let body = "Retry the request after an exponential backoff interval. ".repeat(16);
    fs::write(
        input.join("errors.md"),
        format!("# Errors\n\n## Retries\n\n{body}\n\n## Retries\n\n{body}\n"),
    )
    .unwrap();

    let build = options(input, temp.path().join("duplicate.annpack"), "1.0.0");
    let error = build_pack(&build).expect_err("a colliding passage ID must fail the build");
    let message = error.to_string();
    assert!(message.contains("errors.md"), "{message}");
    assert!(message.contains("passage identifier"), "{message}");
    assert!(
        !build.output.exists(),
        "no artifact may be written for a rejected build"
    );
}

#[test]
fn discovery_refuses_a_pack_with_corrupted_non_manifest_content() {
    // A discovery document publishes a root and invites clients to fetch the
    // artifact behind it. Opening the container and reading the manifest is not
    // enough: the manifest section can be intact while passage data is not.
    let temp = TempDir::new().unwrap();
    let build = options(
        fixture("docs-v1"),
        temp.path().join("valid.annpack"),
        "1.0.0",
    );
    build_pack(&build).unwrap();
    assert!(annpack::discovery::create_discovery(&[&build.output], None, None).is_ok());

    let reader = PackReader::open_path(&build.output).unwrap();
    let passage = reader
        .entries
        .iter()
        .find(|entry| entry.section_type == SectionType::PassageData)
        .unwrap();
    let mut bytes = fs::read(&build.output).unwrap();
    bytes[passage.offset as usize] ^= 0x01;
    let corrupt = temp.path().join("corrupt.annpack");
    fs::write(&corrupt, bytes).unwrap();

    // The manifest is untouched, so this pack still opens and inspects cleanly.
    let corrupt_reader = PackReader::open_path(&corrupt).unwrap();
    corrupt_reader.manifest().unwrap();

    let error = annpack::discovery::create_discovery(&[&corrupt], None, None)
        .expect_err("discovery must not publish a pack whose sections do not verify");
    assert!(
        matches!(error, annpack::error::AnnpackError::Integrity(_)),
        "{error:?}"
    );
}

#[test]
fn sign_verify_and_reject_wrong_trust_key() {
    let temp = TempDir::new().unwrap();
    let build = options(
        fixture("docs-v1"),
        temp.path().join("unsigned.annpack"),
        "1.0.0",
    );
    build_pack(&build).unwrap();
    let secret = temp.path().join("publisher.key");
    let public = temp.path().join("publisher.pub");
    generate_keypair(&secret, Some(&public)).unwrap();
    let signed = temp.path().join("signed.annpack");
    sign_pack(
        &build.output,
        &signed,
        &secret,
        Some("vendor.example".into()),
        None,
    )
    .unwrap();
    let reader = PackReader::open_path(&signed).unwrap();
    let reports = verify_signatures(&reader, Some(&public)).unwrap();
    assert_eq!(reports.len(), 1);
    assert!(reports[0].cryptographically_valid);
    assert!(reports[0].identity_trusted);
    let signed_search = SearchEngine::open_path(&signed)
        .unwrap()
        .search(
            "AP-104",
            &SearchOptions {
                mode: SearchMode::Lexical,
                ..SearchOptions::default()
            },
        )
        .unwrap();
    assert_eq!(
        signed_search.pack.publisher.status,
        "cryptographically_verified"
    );
    assert_eq!(
        signed_search.pack.publisher.asserted_identities,
        vec!["vendor.example"]
    );
    assert!(!signed_search.pack.publisher.identity_trusted);
    let trusted_search = SearchEngine::open_path_with_trusted_key(&signed, Some(&public))
        .unwrap()
        .search(
            "AP-104",
            &SearchOptions {
                mode: SearchMode::Lexical,
                ..SearchOptions::default()
            },
        )
        .unwrap();
    assert!(trusted_search.pack.publisher.identity_trusted);
    assert!(
        trusted_search.results[0]
            .evidence
            .publisher
            .identity_trusted
    );

    let wrong_secret = temp.path().join("wrong.key");
    let wrong_public = temp.path().join("wrong.pub");
    generate_keypair(&wrong_secret, Some(&wrong_public)).unwrap();
    assert!(verify_signatures(&reader, Some(&wrong_public)).is_err());

    let twice_signed = temp.path().join("twice-signed.annpack");
    sign_pack(
        &signed,
        &twice_signed,
        &wrong_secret,
        Some("rotated.vendor.example".into()),
        None,
    )
    .unwrap();
    let twice_reader = PackReader::open_path(&twice_signed).unwrap();
    let reports = verify_signatures(&twice_reader, None).unwrap();
    assert_eq!(reports.len(), 2);
    assert_eq!(twice_reader.root_hex(), reader.root_hex());
    let reports = verify_signatures(&twice_reader, Some(&public)).unwrap();
    assert_eq!(
        reports
            .iter()
            .filter(|report| report.identity_trusted)
            .count(),
        1
    );
    let reports = verify_signatures(&twice_reader, Some(&wrong_public)).unwrap();
    assert_eq!(
        reports
            .iter()
            .filter(|report| report.identity_trusted)
            .count(),
        1
    );
}

#[cfg(unix)]
#[test]
fn generated_secret_key_is_owner_only_and_refuses_overwrite() {
    use std::os::unix::fs::PermissionsExt;

    let temp = TempDir::new().unwrap();
    let secret = temp.path().join("publisher.key");
    generate_keypair(&secret, None).unwrap();
    assert_eq!(
        fs::metadata(&secret).unwrap().permissions().mode() & 0o777,
        0o600
    );
    assert!(generate_keypair(&secret, None).is_err());
}

#[test]
fn a_failed_public_key_write_leaves_no_orphaned_secret_key() {
    // The secret key is created first. If the public key cannot be written, the
    // caller gets an error and must not be left holding a private key with no
    // matching public key on disk.
    let temp = TempDir::new().unwrap();
    let secret = temp.path().join("publisher.key");
    // A directory that does not exist makes `create_new` fail on the public key
    // alone, after the secret key has already been created.
    let public = temp.path().join("missing-directory/publisher.pub");

    assert!(generate_keypair(&secret, Some(&public)).is_err());
    assert!(
        !secret.exists(),
        "the orphaned secret key must be removed when key generation fails"
    );
    assert!(!public.exists());

    // The cleanup must not have consumed the ability to generate a key here.
    generate_keypair(&secret, None).unwrap();
    assert!(secret.exists());
}

#[test]
fn key_generation_never_removes_a_pre_existing_file() {
    let temp = TempDir::new().unwrap();
    let secret = temp.path().join("publisher.key");
    fs::write(&secret, "pre-existing operator key\n").unwrap();
    let public = temp.path().join("missing-directory/publisher.pub");

    assert!(generate_keypair(&secret, Some(&public)).is_err());
    assert_eq!(
        fs::read_to_string(&secret).unwrap(),
        "pre-existing operator key\n",
        "a pre-existing key file must survive a failed generation"
    );
}

#[test]
fn delta_requires_exact_base_and_reproduces_target() {
    let temp = TempDir::new().unwrap();
    let v1 = options(fixture("docs-v1"), temp.path().join("v1.annpack"), "1.0.0");
    let v2 = options(fixture("docs-v2"), temp.path().join("v2.annpack"), "2.0.0");
    build_pack(&v1).unwrap();
    build_pack(&v2).unwrap();
    let delta = temp.path().join("v1-v2.anndelta");
    create_delta(&v1.output, &v2.output, &delta).unwrap();
    let applied = temp.path().join("applied.annpack");
    apply_delta(&v1.output, &delta, &applied).unwrap();
    assert_eq!(fs::read(applied).unwrap(), fs::read(&v2.output).unwrap());
    assert!(apply_delta(&v2.output, &delta, &temp.path().join("wrong.annpack")).is_err());
}

#[test]
fn copy_add_delta_is_materially_smaller_for_a_localized_docs_change() {
    let temp = TempDir::new().unwrap();
    let docs = temp.path().join("large-docs");
    fs::create_dir(&docs).unwrap();
    for index in 0..240 {
        fs::write(
            docs.join(format!("component-{index:03}.md")),
            format!(
                "# Component {index}\n\n## API\n\nComponent {index} returns code AP-{index:03}. {}\n",
                "Stable reference material for agents. ".repeat(40)
            ),
        )
        .unwrap();
    }
    let base = options(docs.clone(), temp.path().join("large-v1.annpack"), "1.0.0");
    build_pack(&base).unwrap();
    fs::write(
        docs.join("component-120.md"),
        "# Component 120\n\n## API\n\nComponent 120 now returns AP-NEW after a localized update.\n",
    )
    .unwrap();
    let target = options(docs, temp.path().join("large-v2.annpack"), "1.0.0");
    build_pack(&target).unwrap();
    let delta = temp.path().join("localized.anndelta");
    let report = create_delta(&base.output, &target.output, &delta).unwrap();
    assert_eq!(report.kind, "copy_add_v1");
    assert!(report.copied_bytes > report.target_bytes / 3);
    assert!(
        report.delta_bytes < report.target_bytes * 3 / 4,
        "delta report: {report:?}"
    );
    let reconstructed = temp.path().join("localized-reconstructed.annpack");
    apply_delta(&base.output, &delta, &reconstructed).unwrap();
    assert_eq!(
        fs::read(reconstructed).unwrap(),
        fs::read(target.output).unwrap()
    );
}

#[test]
fn vector_and_lexical_candidates_fuse() {
    let temp = TempDir::new().unwrap();
    let corpus = ingest_directory(fixture("docs-v1"), &IngestOptions::default()).unwrap();
    let vectors: Vec<Vec<f32>> = corpus
        .passages
        .iter()
        .map(|passage| {
            if passage.text.contains("API key has expired") {
                vec![1.0, 0.0, 0.0]
            } else if passage.text.contains("rotateKey") {
                vec![0.0, 1.0, 0.0]
            } else {
                vec![0.0, 0.0, 1.0]
            }
        })
        .collect();
    let vector_input = VectorInput {
        profile: EmbeddingProfile {
            id: "fixture-v1".into(),
            model: "deterministic-test-model".into(),
            revision: "sha256:test".into(),
            dimensions: 3,
            dtype: "float32".into(),
            pooling: "fixture".into(),
            normalized: true,
            query_prefix: None,
            document_prefix: None,
            runtime: None,
        },
        vectors,
        passage_ids: corpus
            .passages
            .iter()
            .map(|passage| passage.id.clone())
            .collect(),
    };
    let vector_path = temp.path().join("vectors.json");
    fs::write(&vector_path, serde_json::to_vec(&vector_input).unwrap()).unwrap();
    let mut build = options(
        fixture("docs-v1"),
        temp.path().join("hybrid.annpack"),
        "1.0.0",
    );
    build.vector_input = Some(vector_path);
    build_pack(&build).unwrap();
    let reader = PackReader::open_path(&build.output).unwrap();
    assert!(reader.first_entry(SectionType::VectorIndex).is_some());
    assert!(
        reader
            .manifest()
            .unwrap()
            .capabilities
            .contains(&"vector-ivf-flat-dot".to_string())
    );
    let response = SearchEngine::open_path(&build.output)
        .unwrap()
        .search(
            "expired credential",
            &SearchOptions {
                mode: SearchMode::Hybrid,
                query_vector: Some(vec![1.0, 0.0, 0.0]),
                vector_profile: Some("fixture-v1".into()),
                debug: true,
                ..SearchOptions::default()
            },
        )
        .unwrap();
    assert_eq!(response.effective_mode, SearchMode::Hybrid);
    assert_eq!(response.pack.conformance.extensions, vec!["ANN-1"]);
    assert!(response.results[0].text.contains("API key has expired"));
    assert!(response.results[0].lexical_rank.is_some());
    assert!(response.results[0].vector_rank.is_some());

    let response = SearchEngine::open_path(&build.output)
        .unwrap()
        .search(
            "semantic-only",
            &SearchOptions {
                mode: SearchMode::Vector,
                query_vector: Some(vec![1.0, 0.0, 0.0]),
                vector_profile: Some("fixture-v1".into()),
                vector_probes: 1,
                ..SearchOptions::default()
            },
        )
        .unwrap();
    assert!(response.results[0].text.contains("API key has expired"));
}

#[test]
fn mcp_processes_multiple_requests_and_survives_bad_tool_call() {
    let temp = TempDir::new().unwrap();
    let build = options(fixture("docs-v1"), temp.path().join("mcp.annpack"), "1.0.0");
    build_pack(&build).unwrap();
    let engine = SearchEngine::open_path(&build.output).unwrap();
    let input = [
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}),
        json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"unknown","arguments":{}}}),
        json!({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"knowledge_search","arguments":{"query":"AP-104","limit":2,"mode":"lexical"}}}),
        json!({"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"knowledge_pack_info","arguments":{}}}),
    ]
    .iter()
    .map(|value| value.to_string())
    .collect::<Vec<_>>()
    .join("\n")
        + "\n";
    let mut output = Vec::new();
    McpServer::new(engine)
        .run(BufReader::new(Cursor::new(input)), &mut output)
        .unwrap();
    let responses: Vec<Value> = String::from_utf8(output)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(responses.len(), 5);
    assert_eq!(responses[0]["result"]["serverInfo"]["name"], "annpack");
    assert_eq!(responses[1]["result"]["tools"].as_array().unwrap().len(), 4);
    assert!(responses[2].get("error").is_some());
    assert_eq!(
        responses[3]["result"]["structuredContent"]["pack"]["version"],
        "1.0.0"
    );
    assert_eq!(
        responses[4]["result"]["structuredContent"]["name"],
        "vendor-docs"
    );
}

#[test]
fn build_bytes_is_deterministic_without_touching_output() {
    let temp = TempDir::new().unwrap();
    let build = options(
        fixture("docs-v1"),
        temp.path().join("unused.annpack"),
        "1.0.0",
    );
    assert_eq!(
        build_pack_bytes(&build).unwrap(),
        build_pack_bytes(&build).unwrap()
    );
}
