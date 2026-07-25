use std::path::Path;

use annpack::build::{BuildOptions, build_pack_bytes};
use annpack::format::PackReader;
use annpack::model::AccessClass;
use annpack::reader::MemoryReader;
use std::sync::Arc;

const GOLDEN: &[u8] = include_bytes!("../spec/test-vectors/minimal-v3.annpack");

#[test]
fn golden_pack_is_byte_identical_and_searchable() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let generated = build_pack_bytes(&BuildOptions {
        input: root.join("spec/test-vectors/source"),
        output: root.join("target/unused-golden.annpack"),
        name: "minimal-conformance".into(),
        version: "3.0.0".into(),
        description: Some("Canonical ANNPack v3 golden artifact".into()),
        source_revision: Some("spec:FORMAT-v3".into()),
        base_url: Some("https://example.test".into()),
        created_at: None,
        license: Some("CC0-1.0".into()),
        access: AccessClass::Public,
        redistributable: None,
        policy_expires_at: None,
        policy_url: None,
        dependencies: Vec::new(),
        policy_override: None,
        vector_input: None,
        expansion_input: None,
        splade_input: None,
        anchors_input: None,
        target_chars: 1_200,
        max_chars: 2_400,
        input_format: annpack::ingest::InputFormat::Auto,
    })
    .unwrap();
    assert_eq!(generated, GOLDEN);
    let reader = PackReader::open(Arc::new(MemoryReader::new(GOLDEN.to_vec()))).unwrap();
    assert_eq!(
        reader.root_hex(),
        "88073e8443ab7638b21991a2584507abf8e4e96d0e89d1cfbe30db2805e286b8"
    );
    reader.verify_all().unwrap();
}
