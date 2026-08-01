use std::fs;
use std::io::{BufReader, stdin, stdout};
use std::path::{Path, PathBuf};
#[cfg(feature = "http")]
use std::sync::Arc;

use annpack::build::{BuildOptions, build_pack};
use annpack::conformance::{inspect_conformance, inspect_conformance_with_manifest};
use annpack::delta::{apply_delta, create_delta, inspect_delta};
use annpack::discovery::create_discovery;
use annpack::error::{AnnpackError, Result};
use annpack::format::PackReader;
use annpack::ingest::InputFormat;
use annpack::mcp::McpServer;
use annpack::model::{AccessClass, PackDependency, PackPolicy};
use annpack::oci::{
    RegistryCredentials, create_oci_manifest, pull_pack as oci_pull_pack,
    push_pack as oci_push_pack,
};
use annpack::search::{ProfileRequest, SearchEngine, SearchMode, SearchOptions};
use annpack::signing::{generate_keypair, sign_pack, verify_signatures};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde_json::json;

#[derive(Debug, Parser)]
#[command(
    name = "annpack",
    version,
    about = "Build, verify, distribute, and search authoritative knowledge packs"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
// This value is constructed once during CLI parsing; boxing the build arguments
// would add indirection without reducing any persistent runtime allocation.
#[allow(clippy::large_enum_variant)]
enum Command {
    /// Build a deterministic v3 knowledge pack from Markdown or MDX.
    Build(BuildCommand),
    /// Deterministically canonicalize an offline model's raw output into a
    /// pinned, hashed retrieval sidecar (ANN-7/ANN-8/ANN-9). No model runs here.
    Generate {
        #[command(subcommand)]
        command: GenerateCommand,
    },
    /// Inspect a pack without searching it. Output is JSON unless `--human`.
    Inspect {
        input: PathBuf,
        /// Emit the full report as JSON. This is the default; the flag is
        /// accepted so a caller can state the format explicitly.
        #[arg(long, conflicts_with = "human")]
        json: bool,
        /// Print a short readable summary instead of the JSON report.
        #[arg(long)]
        human: bool,
    },
    /// Verify container bounds, hashes, and any embedded signatures.
    Verify {
        input: PathBuf,
        #[arg(long)]
        public_key: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Search a local pack or an HTTP(S) URL.
    Search {
        input: String,
        query: String,
        #[arg(short, long, default_value_t = 10)]
        limit: usize,
        #[arg(long, value_enum, default_value_t = CliSearchMode::Lexical)]
        mode: CliSearchMode,
        #[arg(long)]
        query_vector: Option<PathBuf>,
        #[arg(long)]
        vector_profile: Option<String>,
        #[arg(long, default_value_t = 4)]
        vector_probes: usize,
        /// ANN-7 expansion overlay weight (0.0 = no effect, reproduces Core).
        #[arg(long, default_value_t = 0.0)]
        expansion_weight: f64,
        /// ANN-8 vocabulary overlay weight (0.0 = no effect, reproduces Core).
        #[arg(long, default_value_t = 0.0)]
        splade_weight: f64,
        /// ANN-10 profile: a profile id, "auto" (first supported), or "lexical"
        /// (default; Core lexical, never activates a derived profile).
        #[arg(long)]
        profile: Option<String>,
        #[arg(long)]
        debug: bool,
        #[arg(long)]
        public_key: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Tokenize text with the normative Core tokenizer (FORMAT-v3 §6.1).
    ///
    /// Exists so an independent implementation can be compared against the
    /// reference token-for-token by the conformance runner. Tokenization is
    /// normative and was the single largest source of reader divergence, so it
    /// needs to be directly observable rather than only inferable from rankings.
    Tokenize { text: String },
    /// Issue a standalone evidence receipt for one passage.
    ///
    /// The receipt carries the passage record, its inclusion proof, the manifest
    /// and the section directory, so a third party can verify the citation with
    /// `verify-evidence` — offline, without the pack, and without trusting this
    /// tool or any hosted service.
    Receipt {
        input: String,
        passage_id: String,
        #[arg(short, long)]
        output: Option<PathBuf>,
        #[arg(long)]
        public_key: Option<PathBuf>,
    },
    /// Verify a standalone evidence receipt. Needs no pack and no network.
    ///
    /// Exits non-zero if any integrity claim fails. Without a trusted key the
    /// report still states signature and identity status separately, and a
    /// valid signature is never reported as identity-trusted.
    VerifyEvidence {
        receipt: PathBuf,
        /// Assert that this exact Ed25519 public key (hex) signed the receipt.
        /// The command exits non-zero unless a valid signature from that key is
        /// present, even when the integrity chain itself verifies.
        #[arg(long)]
        trusted_public_key: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Export verified passages for embedding or relevance labeling.
    ExportPassages {
        input: String,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Serve a pack as an MCP server over standard input/output.
    Mcp {
        input: String,
        #[arg(long)]
        public_key: Option<PathBuf>,
    },
    /// Create an Ed25519 signing keypair.
    Keygen {
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long)]
        public_output: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Add an Ed25519 signature section without changing the content root.
    Sign {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long)]
        key: PathBuf,
        #[arg(long)]
        identity: Option<String>,
        #[arg(long)]
        expires_at: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Generate a candidate /.well-known/annpack.json discovery document.
    Discovery {
        #[arg(required = true)]
        packs: Vec<PathBuf>,
        #[arg(short, long)]
        output: Option<PathBuf>,
        #[arg(long)]
        public_base_url: Option<String>,
        #[arg(long)]
        publisher: Option<String>,
    },
    /// Emit an OCI artifact manifest for a pack blob.
    OciManifest {
        input: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Push a verified pack to any OCI Distribution registry.
    Push {
        input: PathBuf,
        reference: String,
        #[arg(long)]
        username: Option<String>,
        #[arg(long, default_value = "ANNPACK_REGISTRY_PASSWORD")]
        password_env: String,
        #[arg(long)]
        json: bool,
    },
    /// Pull and verify a pack from any OCI Distribution registry.
    Pull {
        reference: String,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long)]
        username: Option<String>,
        #[arg(long, default_value = "ANNPACK_REGISTRY_PASSWORD")]
        password_env: String,
        #[arg(long)]
        force: bool,
        #[arg(long)]
        json: bool,
    },
    /// Create, inspect, or apply a verified update layer.
    Delta {
        #[command(subcommand)]
        command: DeltaCommand,
    },
    /// Configure agent runtimes to consume an ANNPack through MCP.
    Integrate {
        #[command(subcommand)]
        command: IntegrationCommand,
    },
}

#[derive(Debug, Args)]
struct BuildCommand {
    input: PathBuf,
    #[arg(short, long)]
    output: PathBuf,
    #[arg(long)]
    name: String,
    #[arg(long)]
    version: String,
    #[arg(long)]
    description: Option<String>,
    #[arg(long)]
    source_revision: Option<String>,
    #[arg(long)]
    base_url: Option<String>,
    #[arg(long)]
    created_at: Option<String>,
    #[arg(long)]
    license: Option<String>,
    #[arg(long, value_enum, default_value_t = CliAccessClass::Public)]
    access: CliAccessClass,
    #[arg(long)]
    redistributable: Option<bool>,
    #[arg(long)]
    policy_expires_at: Option<String>,
    #[arg(long)]
    policy_url: Option<String>,
    #[arg(long)]
    dependencies: Option<PathBuf>,
    #[arg(long)]
    policy_file: Option<PathBuf>,
    #[arg(long)]
    vectors: Option<PathBuf>,
    /// ANN-7 pinned expansion sidecar (see `annpack generate expansion`).
    #[arg(long)]
    expansion: Option<PathBuf>,
    /// ANN-8 pinned splade sidecar (see `annpack generate splade`).
    #[arg(long)]
    splade: Option<PathBuf>,
    /// ANN-9 pinned anchor sidecar (see `annpack generate anchors`).
    #[arg(long)]
    anchors: Option<PathBuf>,
    #[arg(long, default_value_t = 1_200)]
    target_chars: usize,
    #[arg(long, default_value_t = 2_400)]
    max_chars: usize,
    #[arg(long, value_enum, default_value_t = CliInputFormat::Auto)]
    source_format: CliInputFormat,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Subcommand)]
enum GenerateCommand {
    /// ANN-7: filter and canonicalize raw doc2query candidates into a pinned
    /// expansion sidecar. `--threshold` drops low-relevance generated queries.
    Expansion {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value_t = 0.5)]
        threshold: f64,
    },
    /// ANN-8: quantize and canonicalize raw SPLADE term weights into a pinned
    /// vocabulary-overlay sidecar.
    Splade {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// ANN-9: quantize and canonicalize raw anchor similarities into a pinned
    /// anchor sidecar. Research-grade and unvalidated.
    Anchors {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
}

#[derive(Debug, Subcommand)]
enum DeltaCommand {
    Create {
        base: PathBuf,
        target: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    Inspect {
        input: PathBuf,
    },
    Apply {
        base: PathBuf,
        delta: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
}

#[derive(Debug, Subcommand)]
enum IntegrationCommand {
    /// Add a verified pack-backed MCP server to Gemini CLI project settings.
    Gemini {
        input: String,
        #[arg(short, long, default_value = ".gemini/settings.json")]
        output: PathBuf,
        #[arg(long, default_value = "annpack")]
        server_name: String,
        #[arg(long)]
        annpack_command: Option<PathBuf>,
        #[arg(long)]
        force: bool,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliSearchMode {
    Lexical,
    Vector,
    Hybrid,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliAccessClass {
    Public,
    Authenticated,
    Licensed,
    OrganizationRestricted,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliInputFormat {
    Auto,
    Markdown,
    Okf,
}

impl From<CliInputFormat> for InputFormat {
    fn from(value: CliInputFormat) -> Self {
        match value {
            CliInputFormat::Auto => InputFormat::Auto,
            CliInputFormat::Markdown => InputFormat::Markdown,
            CliInputFormat::Okf => InputFormat::Okf,
        }
    }
}

impl From<CliAccessClass> for AccessClass {
    fn from(value: CliAccessClass) -> Self {
        match value {
            CliAccessClass::Public => AccessClass::Public,
            CliAccessClass::Authenticated => AccessClass::Authenticated,
            CliAccessClass::Licensed => AccessClass::Licensed,
            CliAccessClass::OrganizationRestricted => AccessClass::OrganizationRestricted,
        }
    }
}

impl From<CliSearchMode> for SearchMode {
    fn from(value: CliSearchMode) -> Self {
        match value {
            CliSearchMode::Lexical => SearchMode::Lexical,
            CliSearchMode::Vector => SearchMode::Vector,
            CliSearchMode::Hybrid => SearchMode::Hybrid,
        }
    }
}

fn main() {
    if let Err(error) = run(Cli::parse()) {
        eprintln!("annpack: {error}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Build(BuildCommand {
            input,
            output,
            name,
            version,
            description,
            source_revision,
            base_url,
            created_at,
            license,
            access,
            redistributable,
            policy_expires_at,
            policy_url,
            dependencies,
            policy_file,
            vectors,
            expansion,
            splade,
            anchors,
            target_chars,
            max_chars,
            source_format,
            json,
        }) => {
            let report = build_pack(&BuildOptions {
                input,
                output,
                name,
                version,
                description,
                source_revision,
                base_url,
                created_at,
                license,
                access: access.into(),
                redistributable,
                policy_expires_at,
                policy_url,
                dependencies: dependencies
                    .map(read_dependencies)
                    .transpose()?
                    .unwrap_or_default(),
                policy_override: policy_file.map(read_policy).transpose()?,
                vector_input: vectors,
                expansion_input: expansion,
                splade_input: splade,
                anchors_input: anchors,
                target_chars,
                max_chars,
                input_format: source_format.into(),
            })?;
            if json {
                print_json(&report)?;
            } else {
                println!(
                    "built {} ({} documents, {} passages, {} bytes)\nroot {}",
                    report.output,
                    report.documents,
                    report.passages,
                    report.bytes,
                    report.root_hash
                );
            }
        }
        Command::Generate { command } => run_generate(command)?,
        Command::Inspect {
            input,
            json: _,
            human,
        } => {
            let reader = PackReader::open_path(&input)?;
            let manifest = reader.manifest()?;
            let conformance = inspect_conformance_with_manifest(&reader, &manifest);
            let signatures = verify_signatures(&reader, None)?;
            let value = json!({
                "path": input,
                "root_hash": reader.root_hex(),
                "flags": reader.header.flags,
                "manifest": manifest,
                "conformance": conformance,
                "sections": reader.entries.iter().map(|entry| json!({
                    "id": entry.section_id,
                    "type": entry.section_type.name(),
                    "type_id": entry.section_type.as_u16(),
                    "required": entry.required(),
                    "derived": entry.derived(),
                    "format_version": entry.format_version,
                    "offset": entry.offset,
                    "stored_length": entry.stored_length,
                    "logical_length": entry.logical_length,
                    "item_count": entry.item_count,
                    "hash": hex::encode(entry.hash),
                })).collect::<Vec<_>>(),
                "retrieval_profiles": manifest.retrieval_profiles.iter().map(|profile| json!({
                    "id": profile.id,
                    "kind": profile.kind,
                    "section_ids": profile.section_ids,
                    "requires": profile.requires,
                    "supported_by_reference_runtime":
                        profile.requires.iter().all(|capability| REFERENCE_CAPABILITIES.contains(&capability.as_str())),
                })).collect::<Vec<_>>(),
                "signatures": signatures,
            });
            // JSON stays the default so existing callers are unaffected;
            // `--human` is the opt-in summary.
            if human {
                println!("{}@{}", manifest.name, manifest.version);
                println!("root {}", reader.root_hex());
                println!(
                    "{} documents, {} passages, {} sections",
                    manifest.document_count,
                    manifest.passage_count,
                    reader.entries.len()
                );
                println!(
                    "core conformant: {}; extensions conformant: {}",
                    conformance.core_conformant, conformance.extensions_conformant
                );
                if !conformance.extensions.is_empty() {
                    println!("extensions: {}", conformance.extensions.join(", "));
                }
                for issue in &conformance.issues {
                    println!("issue: {issue}");
                }
                println!("valid signatures: {}", signatures.len());
            } else {
                print_json(&value)?;
            }
        }
        Command::Verify {
            input,
            public_key,
            json,
        } => {
            let reader = PackReader::open_path(&input)?;
            let report = reader.verify_all()?;
            let conformance = inspect_conformance(&reader)?;
            let signatures = verify_signatures(&reader, public_key.as_deref())?;
            let publisher_identity_trusted = public_key.is_some()
                && signatures
                    .iter()
                    .any(|signature| signature.identity_trusted);
            let value = json!({
                "integrity_verified": true,
                "root_hash": report.root_hash,
                "bytes": report.bytes,
                "verified_section_ids": report.section_ids,
                "signatures": signatures,
                "publisher_identity_trusted": publisher_identity_trusted,
                "conformance": conformance,
            });
            if json {
                print_json(&value)?;
            } else {
                println!(
                    "verified {} bytes, {} sections, root {}",
                    report.bytes,
                    report.section_ids.len(),
                    report.root_hash
                );
                println!("valid signatures: {}", signatures.len());
                if public_key.is_none() && !signatures.is_empty() {
                    println!("identity trust: not asserted (no trusted public key supplied)");
                }
            }
        }
        Command::Search {
            input,
            query,
            limit,
            mode,
            query_vector,
            vector_profile,
            vector_probes,
            expansion_weight,
            splade_weight,
            profile,
            debug,
            public_key,
            json,
        } => {
            let engine = open_engine(&input, public_key.as_deref())?;
            let query_vector = query_vector.map(read_query_vector).transpose()?;
            let profile = match profile.as_deref() {
                None | Some("lexical") => ProfileRequest::Lexical,
                Some("auto") => ProfileRequest::Auto,
                Some(id) => ProfileRequest::Named(id.to_string()),
            };
            let response = engine.search(
                &query,
                &SearchOptions {
                    limit,
                    mode: mode.into(),
                    query_vector,
                    vector_profile,
                    vector_probes,
                    expansion_weight,
                    splade_weight,
                    profile,
                    debug,
                    ..SearchOptions::default()
                },
            )?;
            if json {
                print_json(&response)?;
            } else {
                println!(
                    "{}@{} ({})",
                    response.pack.name, response.pack.version, response.pack.root_hash
                );
                let sel = &response.profile_selection;
                if let Some(id) = &sel.selected {
                    println!("profile: {id} — {}", sel.reason);
                }
                for hit in response.results {
                    println!("\n{}. [{:.6}] {}", hit.rank, hit.score, hit.title);
                    if let Some(url) = hit.url {
                        println!("   {url}");
                    }
                    println!("   {}", hit.text.replace('\n', " "));
                }
            }
        }
        Command::Tokenize { text } => {
            print_json(&annpack::search::tokenize(&text))?;
        }
        Command::Receipt {
            input,
            passage_id,
            output,
            public_key,
        } => {
            let engine = open_engine(&input, public_key.as_deref())?;
            let receipt = engine.receipt_for_passage(&passage_id)?;
            write_or_print(output.as_deref(), &receipt)?;
        }
        Command::VerifyEvidence {
            receipt,
            trusted_public_key,
            json,
        } => {
            // Bound the file before reading it, not after.
            let bytes = fs::metadata(&receipt)?.len();
            if bytes > annpack::evidence::MAX_RECEIPT_FILE_BYTES {
                return Err(AnnpackError::InvalidInput(format!(
                    "receipt is {bytes} bytes, above the {} byte limit",
                    annpack::evidence::MAX_RECEIPT_FILE_BYTES
                )));
            }
            let parsed: annpack::evidence::EvidenceReceipt =
                serde_json::from_slice(&fs::read(&receipt)?)?;
            let report = annpack::evidence::verify_receipt(&parsed, trusted_public_key.as_deref())?;
            if json {
                print_json(&report)?;
            } else {
                println!("receipt {} passage {}", parsed.pack, parsed.passage_id);
                println!(
                    "  passage hash matches:        {}",
                    report.passage_hash_matches
                );
                println!(
                    "  inclusion proof valid:       {}",
                    report.inclusion_proof_valid
                );
                println!(
                    "  manifest commits merkle root:{}",
                    report.manifest_commits_merkle_root
                );
                println!(
                    "  manifest matches directory:  {}",
                    report.manifest_matches_directory
                );
                println!(
                    "  directory matches pack root: {}",
                    report.directory_matches_pack_root
                );
                println!(
                    "  passage metadata matches:    {}",
                    report.passage_metadata_matches
                );
                println!(
                    "  source revision matches:     {}",
                    report.source_revision_matches
                );
                println!("  pack matches:                {}", report.pack_matches);
                println!(
                    "  canonical url matches:       {}",
                    report.canonical_url_matches
                );
                println!("  signature valid:             {}", report.signature_valid);
                println!("  identity trusted:            {}", report.identity_trusted);
                for issue in &report.issues {
                    println!("  issue: {issue}");
                }
                println!(
                    "{}",
                    if report.verified {
                        "VERIFIED: this passage was in the named artifact, unmodified."
                    } else {
                        "NOT VERIFIED"
                    }
                );
            }
            if !report.verified {
                return Err(AnnpackError::Integrity(
                    "evidence receipt failed verification".into(),
                ));
            }
            // `verified` is an integrity verdict and deliberately stays separate
            // from authenticity and identity: the structured report keeps all
            // three. Supplying `--trusted-public-key` is an explicit assertion
            // that this publisher signed the receipt, so the command must fail
            // when that assertion does not hold, even though the chain itself
            // verified.
            if trusted_public_key.is_some() && !(report.signature_valid && report.identity_trusted)
            {
                return Err(AnnpackError::Signature(
                    "receipt integrity verified, but no valid signature from the supplied \
                     trusted public key is present"
                        .into(),
                ));
            }
        }
        Command::ExportPassages { input, output } => {
            let engine = open_engine(&input, None)?;
            let passages = engine.passages()?;
            write_or_print(output.as_deref(), &passages)?;
        }
        Command::Mcp { input, public_key } => {
            let engine = open_engine(&input, public_key.as_deref())?;
            eprintln!(
                "annpack MCP serving {}@{} root {}",
                engine.manifest().name,
                engine.manifest().version,
                engine.reader().root_hex()
            );
            McpServer::new(engine).run(BufReader::new(stdin().lock()), stdout().lock())?;
        }
        Command::Keygen {
            output,
            public_output,
            json,
        } => {
            let (_, public_key) = generate_keypair(&output, public_output.as_deref())?;
            let public_path = public_output.unwrap_or_else(|| output.with_extension("pub"));
            let value = json!({
                "secret_key_path": output,
                "public_key_path": public_path,
                "public_key": public_key,
                "warning": "The public key is not a trusted publisher identity until bound by external policy."
            });
            if json {
                print_json(&value)?;
            } else {
                println!("created {} and {}", output.display(), public_path.display());
            }
        }
        Command::Sign {
            input,
            output,
            key,
            identity,
            expires_at,
            json,
        } => {
            let report = sign_pack(&input, &output, &key, identity, expires_at)?;
            if json {
                print_json(&report)?;
            } else {
                println!("signed {} with key {}", output.display(), report.key_id);
                println!("cryptographic validity: yes; publisher identity trust: not asserted");
            }
        }
        Command::Discovery {
            packs,
            output,
            public_base_url,
            publisher,
        } => {
            let discovery = create_discovery(&packs, public_base_url.as_deref(), publisher)?;
            write_or_print(output.as_deref(), &discovery)?;
        }
        Command::OciManifest { input, output } => {
            let manifest = create_oci_manifest(&input)?;
            write_or_print(output.as_deref(), &manifest)?;
        }
        Command::Push {
            input,
            reference,
            username,
            password_env,
            json,
        } => {
            let report = oci_push_pack(
                &input,
                &reference,
                registry_credentials(username, &password_env)?,
            )?;
            if json {
                print_json(&report)?;
            } else {
                println!(
                    "pushed {} bytes to {}\npack root {}\nmanifest {}",
                    report.bytes, report.reference, report.pack_root, report.manifest_digest
                );
            }
        }
        Command::Pull {
            reference,
            output,
            username,
            password_env,
            force,
            json,
        } => {
            let report = oci_pull_pack(
                &reference,
                &output,
                registry_credentials(username, &password_env)?,
                force,
            )?;
            if json {
                print_json(&report)?;
            } else {
                println!(
                    "pulled {} bytes to {}\npack root {}\nmanifest {}",
                    report.bytes, report.output, report.pack_root, report.manifest_digest
                );
            }
        }
        Command::Delta { command } => {
            let report = match command {
                DeltaCommand::Create {
                    base,
                    target,
                    output,
                } => create_delta(&base, &target, &output)?,
                DeltaCommand::Inspect { input } => inspect_delta(&input)?,
                DeltaCommand::Apply {
                    base,
                    delta,
                    output,
                } => apply_delta(&base, &delta, &output)?,
            };
            print_json(&report)?;
        }
        Command::Integrate { command } => match command {
            IntegrationCommand::Gemini {
                input,
                output,
                server_name,
                annpack_command,
                force,
                json,
            } => {
                let report = write_gemini_integration(
                    &input,
                    &output,
                    &server_name,
                    annpack_command.as_deref(),
                    force,
                )?;
                if json {
                    print_json(&report)?;
                } else {
                    println!(
                        "configured Gemini CLI MCP server {} in {}\npack {}\nroot {}",
                        report["server_name"].as_str().unwrap_or("annpack"),
                        output.display(),
                        report["pack"].as_str().unwrap_or(&input),
                        report["root_hash"].as_str().unwrap_or("remote")
                    );
                }
            }
        },
    }
    Ok(())
}

fn write_gemini_integration(
    input: &str,
    output: &Path,
    server_name: &str,
    annpack_command: Option<&Path>,
    force: bool,
) -> Result<serde_json::Value> {
    if server_name.is_empty()
        || !server_name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
    {
        return Err(AnnpackError::InvalidInput(
            "Gemini MCP server name must contain only letters, numbers, - or _".into(),
        ));
    }
    let (pack, root_hash) = if input.starts_with("http://") || input.starts_with("https://") {
        (input.to_string(), None)
    } else {
        let path = fs::canonicalize(input).map_err(|error| {
            AnnpackError::InvalidInput(format!("cannot resolve pack {input}: {error}"))
        })?;
        let reader = PackReader::open_path(&path)?;
        reader.verify_all()?;
        (path.display().to_string(), Some(reader.root_hex()))
    };
    let executable = annpack_command
        .map(Path::to_path_buf)
        .unwrap_or(std::env::current_exe()?);
    let mut settings = if output.exists() {
        serde_json::from_slice::<serde_json::Value>(&fs::read(output)?)?
    } else {
        json!({})
    };
    let object = settings.as_object_mut().ok_or_else(|| {
        AnnpackError::InvalidInput("Gemini settings root must be a JSON object".into())
    })?;
    let servers = object
        .entry("mcpServers")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| {
            AnnpackError::InvalidInput("Gemini mcpServers setting must be a JSON object".into())
        })?;
    if servers.contains_key(server_name) && !force {
        return Err(AnnpackError::InvalidInput(format!(
            "Gemini MCP server {server_name:?} already exists; pass --force to replace it"
        )));
    }
    servers.insert(
        server_name.into(),
        json!({
            "command": executable,
            "args": ["mcp", pack],
            "timeout": 30_000,
            "trust": true,
            "includeTools": [
                "knowledge_pack_info",
                "knowledge_search",
                "knowledge_evidence_receipt",
                "knowledge_get_passage"
            ]
        }),
    );
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = output.with_extension(format!("tmp-{}", std::process::id()));
    fs::write(&temporary, serde_json::to_vec_pretty(&settings)?)?;
    fs::rename(&temporary, output)?;
    let verified_before_configuration = root_hash.is_some();
    Ok(json!({
        "integration": "gemini-cli-mcp",
        "server_name": server_name,
        "settings": output,
        "annpack_command": executable,
        "pack": pack,
        "root_hash": root_hash,
        "verified_before_configuration": verified_before_configuration,
        "read_only_tools": true
    }))
}

/// Capabilities the reference runtime implements, for ANN-10 profile selection.
/// `anchor-relative` is intentionally absent: ANN-9 relative-coordinate retrieval
/// was withdrawn, so anchor profiles are never advertised or selected.
const REFERENCE_CAPABILITIES: [&str; 4] = [
    "lexical-bm25",
    "vector-ivf-flat-dot",
    "term-overlay-expansion",
    "term-overlay-splade",
];

fn run_generate(command: GenerateCommand) -> Result<()> {
    use annpack::derive::{generate_anchors, generate_expansion, generate_splade};
    match command {
        GenerateCommand::Expansion {
            input,
            output,
            threshold,
        } => {
            let raw = serde_json::from_slice(&fs::read(input)?)?;
            let sidecar = generate_expansion(&raw, threshold)?;
            let bytes = serde_json::to_vec_pretty(&sidecar)?;
            write_sidecar(&output, &bytes)?;
        }
        GenerateCommand::Splade { input, output } => {
            let raw = serde_json::from_slice(&fs::read(input)?)?;
            let sidecar = generate_splade(&raw)?;
            let bytes = serde_json::to_vec_pretty(&sidecar)?;
            write_sidecar(&output, &bytes)?;
        }
        GenerateCommand::Anchors { input, output } => {
            let raw = serde_json::from_slice(&fs::read(input)?)?;
            let sidecar = generate_anchors(&raw)?;
            let bytes = serde_json::to_vec_pretty(&sidecar)?;
            write_sidecar(&output, &bytes)?;
        }
    }
    Ok(())
}

fn write_sidecar(output: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = output.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, bytes)?;
    let digest = annpack::derive::sidecar_digest(bytes);
    print_json(&json!({
        "sidecar": output,
        "bytes": bytes.len(),
        "sidecar_digest": digest,
        "note": "commit this sidecar; `annpack build` records this digest in manifest.derived_inputs",
    }))
}

fn open_engine(input: &str, trusted_public_key: Option<&Path>) -> Result<SearchEngine> {
    if input.starts_with("http://") || input.starts_with("https://") {
        #[cfg(feature = "http")]
        {
            return SearchEngine::open_source_with_trusted_key(
                Arc::new(annpack::reader::HttpRangeReader::open(input.to_string())?),
                trusted_public_key,
            );
        }
        #[cfg(not(feature = "http"))]
        {
            return Err(AnnpackError::Unsupported(
                "binary was built without HTTP support".into(),
            ));
        }
    }
    SearchEngine::open_path_with_trusted_key(input, trusted_public_key)
}

fn read_query_vector(path: PathBuf) -> Result<Vec<f32>> {
    let bytes = fs::read(path)?;
    let vector: Vec<f32> = serde_json::from_slice(&bytes)?;
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(AnnpackError::InvalidInput(
            "query vector contains a non-finite value".into(),
        ));
    }
    Ok(vector)
}

fn read_dependencies(path: PathBuf) -> Result<Vec<PackDependency>> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn read_policy(path: PathBuf) -> Result<PackPolicy> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn registry_credentials(
    username: Option<String>,
    password_environment_variable: &str,
) -> Result<Option<RegistryCredentials>> {
    let username = username.or_else(|| std::env::var("ANNPACK_REGISTRY_USERNAME").ok());
    let password = std::env::var(password_environment_variable).ok();
    match (username, password) {
        (None, None) => Ok(None),
        (Some(username), Some(password)) if !username.is_empty() && !password.is_empty() => {
            Ok(Some(RegistryCredentials { username, password }))
        }
        _ => Err(AnnpackError::InvalidInput(format!(
            "registry authentication requires a username and nonempty {password_environment_variable}"
        ))),
    }
}

fn print_json(value: &impl serde::Serialize) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

fn write_or_print(path: Option<&Path>, value: &impl serde::Serialize) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    if let Some(path) = path {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, &bytes)?;
    } else {
        println!("{}", String::from_utf8_lossy(&bytes));
    }
    Ok(())
}
