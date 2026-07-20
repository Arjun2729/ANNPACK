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
use annpack::search::{SearchEngine, SearchMode, SearchOptions};
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
    /// Inspect a pack without searching it.
    Inspect {
        input: PathBuf,
        #[arg(long)]
        json: bool,
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
        #[arg(long, value_enum, default_value_t = CliSearchMode::Hybrid)]
        mode: CliSearchMode,
        #[arg(long)]
        query_vector: Option<PathBuf>,
        #[arg(long)]
        vector_profile: Option<String>,
        #[arg(long, default_value_t = 4)]
        vector_probes: usize,
        #[arg(long)]
        debug: bool,
        #[arg(long)]
        public_key: Option<PathBuf>,
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
        Command::Inspect { input, json: _ } => {
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
                    "format_version": entry.format_version,
                    "offset": entry.offset,
                    "stored_length": entry.stored_length,
                    "logical_length": entry.logical_length,
                    "item_count": entry.item_count,
                    "hash": hex::encode(entry.hash),
                })).collect::<Vec<_>>(),
                "signatures": signatures,
            });
            print_json(&value)?;
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
            debug,
            public_key,
            json,
        } => {
            let engine = open_engine(&input, public_key.as_deref())?;
            let query_vector = query_vector.map(read_query_vector).transpose()?;
            let response = engine.search(
                &query,
                &SearchOptions {
                    limit,
                    mode: mode.into(),
                    query_vector,
                    vector_profile,
                    vector_probes,
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
                for hit in response.results {
                    println!("\n{}. [{:.6}] {}", hit.rank, hit.score, hit.title);
                    if let Some(url) = hit.url {
                        println!("   {url}");
                    }
                    println!("   {}", hit.text.replace('\n', " "));
                }
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
