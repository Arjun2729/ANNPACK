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
use annpack::model::{AccessClass, PackPolicy};
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
    ///
    /// With `--policy` the five claims are evaluated separately and reported
    /// separately. A stronger policy that cannot be satisfied fails; it never
    /// behaves like a weaker one.
    Verify {
        input: PathBuf,
        #[arg(long)]
        public_key: Option<PathBuf>,
        /// Defaults to `integrity-only`, which is what this command has always
        /// checked. Stronger policies require the inputs below.
        #[arg(long, value_enum, default_value_t = CliPolicy::IntegrityOnly)]
        policy: CliPolicy,
        #[arg(long)]
        trust_root: Option<PathBuf>,
        #[arg(long, requires = "channel_state")]
        expect_publisher: Option<String>,
        #[arg(long, requires = "channel_state")]
        expect_corpus: Option<String>,
        #[arg(long, requires = "channel_state")]
        expect_channel: Option<String>,
        #[arg(long)]
        channel_state: Option<PathBuf>,
        #[arg(long)]
        retained_state: Option<PathBuf>,
        #[command(flatten)]
        clock: ClockArgs,
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
        /// Emit OpenTelemetry attributes for this retrieval instead of the
        /// search response, so a tracer can record which immutable artifact the
        /// returned text came from. Always JSON.
        #[arg(long, conflicts_with = "json")]
        otel: bool,
        /// Where this deployment serves receipts, as a template containing
        /// `{passage_id}` and optionally `{root}`. ANNPack does not define that
        /// location, so `annpack.receipt_uri` is emitted only when it is given.
        #[arg(long, requires = "otel")]
        otel_receipt_uri: Option<String>,
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
    /// Assemble one agent run's retrieval evidence into a portable bundle.
    ///
    /// Runs the query, then issues a standalone receipt for every passage the
    /// run retrieved and wraps them with the metadata needed to find the run in
    /// an application's own logs. The bundle adds no cryptography: verifying it
    /// is `verify-evidence` applied to each receipt in turn.
    ///
    /// The query, application, model and answer are carried, not attested. Only
    /// the receipts prove anything.
    Bundle {
        input: String,
        query: String,
        #[arg(short, long)]
        output: Option<PathBuf>,
        #[arg(short, long, default_value_t = 5)]
        limit: usize,
        #[arg(long, value_enum, default_value_t = CliSearchMode::Lexical)]
        mode: CliSearchMode,
        /// Identifier for this run. Defaults to a digest of the query and the
        /// passages it retrieved, which makes the bundle reproducible but does
        /// not identify one occurrence; pass the application's own run ID to
        /// correlate with its logs.
        #[arg(long)]
        run_id: Option<String>,
        #[arg(long)]
        application: Option<String>,
        #[arg(long)]
        model: Option<String>,
        /// File holding the model's answer, carried in the bundle and digested.
        /// The digest establishes only that the bundle was not corrupted in
        /// transit: anyone who can edit the answer can edit the digest.
        #[arg(long)]
        answer: Option<PathBuf>,
        /// Timestamp to record. Omitted by default so that two bundles built
        /// from the same query and artifact are byte-identical.
        #[arg(long)]
        created_at: Option<String>,
        #[arg(long)]
        public_key: Option<PathBuf>,
    },
    /// Verify every receipt in a run bundle. Needs no pack and no network.
    ///
    /// Exits non-zero unless every receipt verifies. Reports what the receipts
    /// attest separately from what the bundle merely carries.
    VerifyRun {
        bundle: PathBuf,
        /// Assert that this exact Ed25519 public key (hex) signed every receipt.
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
    /// Manage publisher trust roots: which keys may act in which role.
    Trust {
        #[command(subcommand)]
        command: TrustCommand,
    },
    /// Manage channel state: which artifact a channel currently stands behind.
    Release {
        #[command(subcommand)]
        command: ReleaseCommand,
    },
    /// Manage build provenance: which source, builder and execution produced
    /// an artifact. Separate from `trust`/`release`: provenance proves how an
    /// artifact was built, not who authorises publishing or using it.
    Provenance {
        #[command(subcommand)]
        command: ProvenanceCommand,
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
    policy_file: Option<PathBuf>,
    #[arg(long)]
    vectors: Option<PathBuf>,
    /// ANN-7 pinned expansion sidecar (see `annpack generate expansion`).
    #[arg(long)]
    expansion: Option<PathBuf>,
    /// ANN-8 pinned splade sidecar (see `annpack generate splade`).
    #[arg(long)]
    splade: Option<PathBuf>,
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
}

/// The scope a consumer is asking about, established outside the statement.
///
/// The publisher is taken from the trusted root unless overridden, and the
/// corpus and channel are mandatory. Nothing here may be defaulted from the
/// document under verification: a statement that supplies its own expectations
/// is only ever compared against itself, which is how the first version of this
/// CLI shipped a scope check that could not fail.
#[derive(Debug, Args, Clone)]
struct ScopeArgs {
    /// Defaults to the publisher named by the trusted root.
    #[arg(long)]
    expect_publisher: Option<String>,
    #[arg(long)]
    expect_corpus: String,
    #[arg(long)]
    expect_channel: String,
}

/// A caller must state where its time comes from.
///
/// There is deliberately no default. Reading the local clock silently would make
/// every expiry check depend on something the operator never vouched for, and an
/// attacker who can move a clock could then extend any statement indefinitely.
/// Supplying neither flag reports validity as unknown, which does not verify.
#[derive(Debug, Args, Clone)]
struct ClockArgs {
    /// Trusted time as `YYYY-MM-DDTHH:MM:SSZ`.
    #[arg(long)]
    now: Option<String>,
    /// Use this machine's clock, asserting that it is trustworthy.
    #[arg(long, conflicts_with = "now")]
    system_clock: bool,
}

#[derive(Debug, Subcommand)]
enum TrustCommand {
    /// Create an unsigned trust root from public keys.
    Init {
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long)]
        publisher: String,
        #[arg(long, default_value_t = 1)]
        version: u64,
        #[arg(long)]
        issued_at: Option<String>,
        #[arg(long)]
        valid_until: String,
        /// Public key file authorised to sign trust roots. Repeatable.
        #[arg(long = "root-key", required = true)]
        root_keys: Vec<PathBuf>,
        #[arg(long = "artifact-key", required = true)]
        artifact_keys: Vec<PathBuf>,
        #[arg(long = "release-key", required = true)]
        release_keys: Vec<PathBuf>,
        #[arg(long = "revocation-key", required = true)]
        revocation_keys: Vec<PathBuf>,
        #[arg(long, default_value_t = 1)]
        root_threshold: u32,
        #[arg(long, default_value_t = 1)]
        artifact_threshold: u32,
        #[arg(long, default_value_t = 1)]
        release_threshold: u32,
        #[arg(long, default_value_t = 1)]
        revocation_threshold: u32,
    },
    /// Add a signature from one key. Signing again with the same key replaces it.
    Sign {
        input: PathBuf,
        #[arg(long)]
        key: PathBuf,
        /// Defaults to rewriting the input in place.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Verify a trust root, optionally as a rotation from a prior trusted one.
    Verify {
        input: PathBuf,
        /// The currently trusted root. Without it this is a first-contact
        /// acceptance and no rotation rule is evaluated.
        #[arg(long)]
        prior: Option<PathBuf>,
        #[command(flatten)]
        clock: ClockArgs,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Subcommand)]
enum ReleaseCommand {
    /// Create an unsigned channel-state statement.
    Statement {
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long)]
        publisher: String,
        #[arg(long)]
        corpus: String,
        #[arg(long, default_value = "production")]
        channel: String,
        #[arg(long)]
        sequence: u64,
        #[arg(long)]
        current_root: String,
        #[arg(long)]
        current_version: String,
        #[arg(long)]
        issued_at: Option<String>,
        #[arg(long)]
        valid_until: String,
        /// Artifact root superseded by the current release. Repeatable.
        #[arg(long = "supersede")]
        superseded: Vec<String>,
        /// Artifact root withdrawn as a security event. Repeatable.
        #[arg(long = "revoke")]
        revoked: Vec<String>,
        #[arg(long, default_value = "withdrawn-by-publisher")]
        revoke_reason: String,
    },
    /// Add a signature from one key.
    Sign {
        input: PathBuf,
        #[arg(long)]
        key: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Verify a statement against a trust root and retained client state.
    Verify {
        input: PathBuf,
        #[arg(long)]
        trust_root: PathBuf,
        #[command(flatten)]
        scope: ScopeArgs,
        /// Retained monotonic state for this channel. Absent means first
        /// contact, which has no rollback resistance.
        #[arg(long)]
        retained_state: Option<PathBuf>,
        #[command(flatten)]
        clock: ClockArgs,
        /// Persist retained state when the statement verifies and advances.
        #[arg(long, requires = "retained_state")]
        accept: bool,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Subcommand)]
// See Command's own allow: the Create variant legitimately carries more
// fields than Sign/Verify, and boxing it would only add indirection.
#[allow(clippy::large_enum_variant)]
enum ProvenanceCommand {
    /// Create an unsigned build-provenance statement for a completed artifact.
    ///
    /// Every binding fact -- the distributed file's digest, the artifact root,
    /// the logical content root, and (for a format-4 artifact) the source
    /// digest -- is derived from the artifact and, when given, the builder
    /// executable. None of them may be supplied as a bare string: there is no
    /// flag for "source digest" here, because the only digest this command can
    /// record is the one it reads out of the artifact.
    Create {
        artifact: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long)]
        repository: String,
        #[arg(long)]
        revision: String,
        #[arg(long)]
        builder_id: String,
        /// Path to the exact executable that performed the build. Hashed by
        /// this command; omit only when builder-binary binding is not needed.
        #[arg(long)]
        builder_binary: Option<PathBuf>,
        #[arg(long)]
        invocation_id: Option<String>,
        #[arg(long)]
        started_at: Option<String>,
        #[arg(long)]
        finished_at: Option<String>,
        /// Use this machine's clock for both timestamps. There is no default
        /// clock, matching `release`/`trust`: a caller must state where time
        /// comes from.
        #[arg(long)]
        system_clock: bool,
        /// `key=value`, repeatable. Nothing is captured unless named here.
        #[arg(long = "param")]
        parameters: Vec<String>,
        #[arg(long = "env")]
        environment: Vec<String>,
        #[arg(long)]
        platform: Option<String>,
        #[arg(long)]
        locked: Option<bool>,
        /// Create provenance for an artifact whose manifest predates format 4.
        /// Requires an explicit source digest, recorded honestly as a builder
        /// claim the artifact cannot corroborate -- never silently accepted as
        /// though it were authenticated.
        #[arg(long, requires = "legacy_source_digest")]
        legacy: bool,
        #[arg(long)]
        legacy_source_digest: Option<String>,
    },
    /// Sign a provenance statement, producing a DSSE envelope.
    Sign {
        input: PathBuf,
        #[arg(long)]
        key: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Verify a provenance envelope against a distributed artifact.
    Verify {
        artifact: PathBuf,
        provenance: PathBuf,
        /// Hex Ed25519 public keys trusted as builders. Repeatable. Unrelated
        /// to any `trust` role -- an artifact-signing key is not thereby a
        /// trusted builder unless listed here explicitly.
        #[arg(long = "trusted-builder-key")]
        trusted_builder_keys: Vec<String>,
        /// Path to the exact builder executable, to independently check the
        /// builder-version and builder-binary claims rather than merely
        /// carrying them.
        #[arg(long)]
        builder_binary: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum CliPolicy {
    IntegrityOnly,
    AuthorizedPublisher,
    AuthorizedCurrent,
    AuthorizedCurrentWitnessed,
}

impl From<CliPolicy> for annpack::policy::TrustPolicy {
    fn from(value: CliPolicy) -> Self {
        match value {
            CliPolicy::IntegrityOnly => Self::IntegrityOnly,
            CliPolicy::AuthorizedPublisher => Self::AuthorizedPublisher,
            CliPolicy::AuthorizedCurrent => Self::AuthorizedCurrent,
            CliPolicy::AuthorizedCurrentWitnessed => Self::AuthorizedCurrentWitnessed,
        }
    }
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

/// Broad, stable exit classes.
///
/// Deliberately coarse. A code per failure would make the numeric table a
/// brittle public API and still carry no context; the precise reason travels in
/// `error.kind` instead. What the class must support is a caller deciding
/// *what to do* — retry, alert, re-fetch, page a human — without parsing prose.
mod exit {
    pub const USAGE: i32 = 2;
    pub const INPUT: i32 = 3;
    pub const OPERATIONAL: i32 = 4;
    /// Cryptographic, authority, or scope verification failure.
    pub const VERIFICATION: i32 = 5;
    /// Temporal or monotonic-state safety failure.
    pub const SAFETY: i32 = 6;
    /// The artifact or statement is authentic and its status denies use.
    pub const DENIED: i32 = 7;
}

/// A failure a machine caller can act on.
struct CliFailure {
    class: i32,
    /// Stable identifier. Callers match on this, never on `message`.
    kind: &'static str,
    /// Which verification stage produced it.
    stage: &'static str,
    message: String,
    /// The full structured report, when one was produced before the failure.
    /// Carried inside the envelope so that JSON mode emits exactly one object:
    /// printing the report and then the envelope produced two, which is not
    /// parseable as a stream of one.
    details: Option<serde_json::Value>,
}

impl CliFailure {
    fn new(
        class: i32,
        kind: &'static str,
        stage: &'static str,
        message: impl Into<String>,
    ) -> Self {
        Self {
            class,
            kind,
            stage,
            message: message.into(),
            details: None,
        }
    }

    fn with_details(mut self, details: serde_json::Value) -> Self {
        self.details = Some(details);
        self
    }
}

/// Default mapping for errors raised before a command classified them itself.
///
/// Commands in the release layer construct precise failures; everything else
/// lands here. Even this is more informative than the single exit code every
/// failure previously shared.
impl From<AnnpackError> for CliFailure {
    fn from(error: AnnpackError) -> Self {
        let (class, kind, stage) = match &error {
            AnnpackError::Io(io) if io.kind() == std::io::ErrorKind::NotFound => {
                (exit::INPUT, "input_unavailable", "input")
            }
            AnnpackError::Io(_) => (exit::OPERATIONAL, "io_failure", "input"),
            AnnpackError::Json(_) => (exit::INPUT, "malformed_input", "parse"),
            AnnpackError::InvalidFormat(_) => (exit::INPUT, "malformed_input", "parse"),
            AnnpackError::InvalidInput(_) => (exit::USAGE, "invalid_usage", "usage"),
            AnnpackError::Unsupported(_) => (exit::INPUT, "unsupported", "parse"),
            AnnpackError::Integrity(_) => (exit::VERIFICATION, "integrity_failed", "artifact"),
            AnnpackError::Signature(_) => (exit::VERIFICATION, "invalid_signature", "signature"),
            AnnpackError::Search(_) => (exit::USAGE, "invalid_usage", "search"),
            AnnpackError::Protocol(_) => (exit::OPERATIONAL, "protocol_failure", "transport"),
            #[cfg(feature = "http")]
            AnnpackError::Http(_) => (exit::OPERATIONAL, "transport_failure", "transport"),
        };
        Self::new(class, kind, stage, error.to_string())
    }
}

/// Classify a policy denial by the first requirement that was not met.
///
/// Ordered by severity, not by evaluation order: a revoked artifact that is also
/// superseded is reported as revoked, because a caller reacting to one of those
/// should react to the more serious one.
fn policy_failure(decision: &annpack::policy::PolicyDecision) -> CliFailure {
    use annpack::policy::{ArtifactIntegrity, PublisherAuthority};
    use annpack::release::Currency;

    let reasons = decision.unmet_requirements.join("; ");
    let (class, kind, stage) = if decision.artifact_integrity != ArtifactIntegrity::Valid {
        (exit::VERIFICATION, "integrity_failed", "artifact")
    } else if decision.currency == Currency::Revoked {
        (exit::DENIED, "revoked", "currency")
    } else if decision.publisher_authority == PublisherAuthority::Unauthorized {
        (exit::VERIFICATION, "unauthorized_role", "authority")
    } else if decision.publisher_authority == PublisherAuthority::Unknown {
        (exit::VERIFICATION, "trust_root_unavailable", "authority")
    } else if decision.currency == Currency::Superseded {
        (exit::DENIED, "superseded", "currency")
    } else if decision.currency == Currency::Unknown {
        (exit::DENIED, "currency_unknown", "currency")
    } else {
        (exit::DENIED, "unmet_policy_requirement", "policy")
    };
    CliFailure::new(
        class,
        kind,
        stage,
        format!("denied under policy {}: {reasons}", decision.policy),
    )
}

/// Classify a channel-state verification failure by its first unmet property.
fn channel_state_failure(report: &annpack::release::ChannelStateVerification) -> CliFailure {
    use annpack::release::{SequenceVerdict, SigningAuthority};

    let detail = report.issues.join("; ");
    let (class, kind, stage) = if !report.schema_supported {
        (exit::INPUT, "unsupported_schema", "schema")
    } else if !report.structurally_valid {
        (exit::INPUT, "malformed_input", "parse")
    } else if !report.trust_root_verified {
        (exit::VERIFICATION, "trust_root_unavailable", "trust-root")
    } else if !report.scope_matches {
        // Ranked above signature checks: a statement for another channel is not
        // this consumer's business regardless of how well it is signed.
        (exit::VERIFICATION, "scope_mismatch", "scope")
    } else if report.authority == SigningAuthority::None {
        (exit::VERIFICATION, "unauthorized_role", "signature")
    } else if report.within_validity == Some(false) {
        (exit::SAFETY, "expired", "time")
    } else if report.within_validity.is_none() {
        (exit::SAFETY, "no_trusted_clock", "time")
    } else {
        match report.sequence_verdict {
            SequenceVerdict::Rollback => (exit::SAFETY, "rollback", "sequence"),
            SequenceVerdict::Equivocation => (exit::SAFETY, "equivocation", "sequence"),
            SequenceVerdict::NotEvaluated => (exit::VERIFICATION, "scope_mismatch", "scope"),
            _ => (exit::VERIFICATION, "verification_failed", "statement"),
        }
    };
    CliFailure::new(class, kind, stage, format!("statement rejected: {detail}"))
}

/// Classify a trust-root verification failure.
fn trust_root_failure(report: &annpack::trust::TrustRootVerification) -> CliFailure {
    let detail = report.issues.join("; ");
    let (class, kind, stage) = if !report.schema_supported {
        (exit::INPUT, "unsupported_schema", "schema")
    } else if !report.structurally_valid || !report.key_ids_match_keys {
        (exit::INPUT, "malformed_input", "parse")
    } else if !report.self_signed || report.signed_by_prior_root == Some(false) {
        (exit::VERIFICATION, "unauthorized_role", "signature")
    } else if report.version_advanced == Some(false) {
        (exit::SAFETY, "rollback", "version")
    } else if report.within_validity == Some(false) {
        (exit::SAFETY, "expired", "time")
    } else if report.within_validity.is_none() {
        (exit::SAFETY, "no_trusted_clock", "time")
    } else {
        (exit::VERIFICATION, "verification_failed", "trust-root")
    };
    CliFailure::new(class, kind, stage, format!("trust root rejected: {detail}"))
}

impl From<std::io::Error> for CliFailure {
    fn from(error: std::io::Error) -> Self {
        AnnpackError::from(error).into()
    }
}

impl From<serde_json::Error> for CliFailure {
    fn from(error: serde_json::Error) -> Self {
        AnnpackError::from(error).into()
    }
}

/// Whether the running command was asked for JSON.
///
/// Set as soon as a command is matched, so that a failure occurring *inside* the
/// handler — a malformed file, an unreadable path — still produces one
/// structured object rather than leaving a JSON caller to scrape stderr.
static JSON_OUTPUT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn set_json_output(enabled: bool) {
    JSON_OUTPUT.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

fn main() {
    if let Err(failure) = run(Cli::parse()) {
        if JSON_OUTPUT.load(std::sync::atomic::Ordering::Relaxed) {
            // Exactly one structured object on stdout, on every failing path.
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "ok": false,
                    "permitted": false,
                    "stage": failure.stage,
                    "error": {
                        "kind": failure.kind,
                        "message": failure.message,
                    },
                    "details": failure.details,
                }))
                .expect("failure envelope is serializable")
            );
        }
        eprintln!("annpack: {}", failure.message);
        std::process::exit(failure.class);
    }
}

fn run(cli: Cli) -> std::result::Result<(), CliFailure> {
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
            policy_file,
            vectors,
            expansion,
            splade,
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
                policy_override: policy_file.map(read_policy).transpose()?,
                vector_input: vectors,
                expansion_input: expansion,
                splade_input: splade,
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
            policy,
            trust_root,
            expect_publisher,
            expect_corpus,
            expect_channel,
            channel_state,
            retained_state,
            clock,
            json,
        } => {
            set_json_output(json);
            let reader = PackReader::open_path(&input)?;
            let report = reader.verify_all()?;
            let conformance = inspect_conformance(&reader)?;
            let signatures = verify_signatures(&reader, public_key.as_deref())?;
            let publisher_identity_trusted = public_key.is_some()
                && signatures
                    .iter()
                    .any(|signature| signature.identity_trusted);

            let decision = evaluate_artifact_policy(
                &report.root_hash,
                &signatures,
                policy.into(),
                trust_root.as_deref(),
                channel_state.as_deref(),
                retained_state.as_deref(),
                (
                    expect_publisher.as_deref(),
                    expect_corpus.as_deref(),
                    expect_channel.as_deref(),
                ),
                &clock,
            )?;

            let value = json!({
                "integrity_verified": true,
                "root_hash": report.root_hash,
                "bytes": report.bytes,
                "verified_section_ids": report.section_ids,
                "signatures": signatures,
                "publisher_identity_trusted": publisher_identity_trusted,
                "conformance": conformance,
                "policy": decision,
            });
            if !decision.permitted {
                return Err(policy_failure(&decision).with_details(value));
            }
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
                println!("policy {}:", decision.policy);
                println!("  artifact integrity:  {:?}", decision.artifact_integrity);
                println!("  publisher authority: {:?}", decision.publisher_authority);
                println!("  release currency:    {:?}", decision.currency);
                println!("  transparency:        {:?}", decision.transparency);
                for note in &decision.assumptions {
                    println!("  assumption: {note}");
                }
                for reason in &decision.unmet_requirements {
                    println!("  unmet: {reason}");
                }
                println!(
                    "{}",
                    if decision.permitted {
                        "PERMITTED under the requested policy."
                    } else {
                        "DENIED under the requested policy."
                    }
                );
            }
            // A denied policy must be an exit code, not a line of output a
            // script can miss. Integrity alone already exited non-zero on
            // failure; this extends that to whatever the caller actually asked.
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
            otel,
            otel_receipt_uri,
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
            if otel {
                print_json(&annpack::telemetry::retrieval_telemetry(
                    &response,
                    otel_receipt_uri.as_deref(),
                )?)?;
            } else if json {
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
                ))
                .into());
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
                return Err(
                    AnnpackError::Integrity("evidence receipt failed verification".into()).into(),
                );
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
                )
                .into());
            }
        }
        Command::Bundle {
            input,
            query,
            output,
            limit,
            mode,
            run_id,
            application,
            model,
            answer,
            created_at,
            public_key,
        } => {
            let engine = open_engine(&input, public_key.as_deref())?;
            let response = engine.search(
                &query,
                &SearchOptions {
                    limit,
                    mode: mode.into(),
                    ..SearchOptions::default()
                },
            )?;
            let mut receipts = Vec::with_capacity(response.results.len());
            for hit in &response.results {
                receipts.push(engine.receipt_for_passage(&hit.passage_id)?);
            }
            let answer = answer.map(read_answer).transpose()?;
            let bundle = annpack::bundle::RunBundle {
                schema: annpack::bundle::RUN_BUNDLE_SCHEMA_V1.to_string(),
                run_id: run_id.unwrap_or_else(|| annpack::bundle::derive_run_id(&query, &receipts)),
                created_at,
                application,
                model,
                answer_hash: answer.as_deref().map(annpack::bundle::answer_hash),
                answer,
                query,
                receipts,
            };
            write_or_print(output.as_deref(), &bundle)?;
        }
        Command::VerifyRun {
            bundle,
            trusted_public_key,
            json,
        } => {
            // Bound the file before reading it, not after.
            let bytes = fs::metadata(&bundle)?.len();
            if bytes > annpack::bundle::MAX_BUNDLE_FILE_BYTES {
                return Err(AnnpackError::InvalidInput(format!(
                    "run bundle is {bytes} bytes, above the {} byte limit",
                    annpack::bundle::MAX_BUNDLE_FILE_BYTES
                ))
                .into());
            }
            let parsed: annpack::bundle::RunBundle = serde_json::from_slice(&fs::read(&bundle)?)?;
            let report =
                annpack::bundle::verify_run_bundle(&parsed, trusted_public_key.as_deref())?;
            if json {
                print_json(&report)?;
            } else {
                println!("run {}", report.run_id);
                println!("  query:            {:?}", report.query);
                println!(
                    "  receipts:         {} of {} verified",
                    report.receipts_verified, report.receipts_total
                );
                for root in &report.pack_roots {
                    println!("  artifact:         {root}");
                }
                for revision in &report.source_revisions {
                    println!("  source revision:  {revision}");
                }
                println!("  all signed:       {}", report.all_receipts_signed);
                println!("  all signers trusted: {}", report.all_signers_trusted);
                println!(
                    "  answer digest:    {}",
                    match report.answer_hash_consistent {
                        Some(true) => "consistent",
                        Some(false) => "MISMATCH",
                        None => "not carried",
                    }
                );
                for issue in &report.issues {
                    println!("  issue: {issue}");
                }
                println!(
                    "{}",
                    if report.attested {
                        "ATTESTED: every cited passage was in the named artifact, unmodified."
                    } else {
                        "NOT ATTESTED"
                    }
                );
                // Stated every time, including on success. The whole failure
                // mode this command guards against is a reader treating the
                // carried fields as though the receipts covered them.
                println!(
                    "Carried but not attested: query, application, model, answer. \
                     Only the receipts prove anything."
                );
            }
            if !report.attested {
                return Err(
                    AnnpackError::Integrity("run bundle failed verification".into()).into(),
                );
            }
            if trusted_public_key.is_some() && !report.all_signers_trusted {
                return Err(AnnpackError::Signature(
                    "run bundle receipts verified, but not every receipt carries a valid \
                     signature from the supplied trusted public key"
                        .into(),
                )
                .into());
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
        Command::Trust { command } => run_trust(command)?,
        Command::Release { command } => run_release(command)?,
        Command::Provenance { command } => run_provenance(command)?,
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
    use annpack::derive::{generate_expansion, generate_splade};
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

fn run_trust(command: TrustCommand) -> std::result::Result<(), CliFailure> {
    use annpack::trust::{
        MAX_TRUST_ROOT_FILE_BYTES, ROLE_ARTIFACT, ROLE_EMERGENCY_REVOCATION, ROLE_RELEASE_STATE,
        ROLE_ROOT, TRUST_ROOT_SCHEMA_V1, TrustRoot, sign_trust_root, verify_trust_root,
    };

    match command {
        TrustCommand::Init {
            output,
            publisher,
            version,
            issued_at,
            valid_until,
            root_keys,
            artifact_keys,
            release_keys,
            revocation_keys,
            root_threshold,
            artifact_threshold,
            release_threshold,
            revocation_threshold,
        } => {
            annpack::trust::parse_utc_timestamp(&valid_until)?;
            let issued_at = match issued_at {
                Some(value) => {
                    annpack::trust::parse_utc_timestamp(&value)?;
                    value
                }
                None => annpack::trust::format_utc_timestamp(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map_err(|_| {
                            AnnpackError::InvalidInput("system clock is before 1970".into())
                        })?
                        .as_secs() as i64,
                ),
            };

            let mut keys = std::collections::BTreeMap::new();
            let mut roles = std::collections::BTreeMap::new();
            for (role, paths, threshold) in [
                (ROLE_ROOT, &root_keys, root_threshold),
                (ROLE_ARTIFACT, &artifact_keys, artifact_threshold),
                (ROLE_RELEASE_STATE, &release_keys, release_threshold),
                (
                    ROLE_EMERGENCY_REVOCATION,
                    &revocation_keys,
                    revocation_threshold,
                ),
            ] {
                roles.insert(
                    role.to_string(),
                    role_from_key_files(paths, threshold, &mut keys)?,
                );
            }

            let root = TrustRoot {
                schema: TRUST_ROOT_SCHEMA_V1.into(),
                publisher,
                version,
                issued_at,
                valid_until,
                roles,
                keys,
                signatures: Vec::new(),
            };
            write_or_print(Some(&output), &root)?;
            eprintln!(
                "wrote unsigned trust root to {}; sign it with `annpack trust sign`",
                output.display()
            );
        }
        TrustCommand::Sign { input, key, output } => {
            let mut root: TrustRoot = read_json(&input, MAX_TRUST_ROOT_FILE_BYTES, "trust root")?;
            let key_id = sign_trust_root(&mut root, &read_secret_key(&key)?)?;
            write_or_print(Some(output.as_deref().unwrap_or(&input)), &root)?;
            eprintln!("signed by {key_id}");
        }
        TrustCommand::Verify {
            input,
            prior,
            clock,
            json,
        } => {
            set_json_output(json);
            let root: TrustRoot = read_json(&input, MAX_TRUST_ROOT_FILE_BYTES, "trust root")?;
            let prior_root = prior
                .map(|path| read_json::<TrustRoot>(&path, MAX_TRUST_ROOT_FILE_BYTES, "prior root"))
                .transpose()?;
            let now = resolve_clock(&clock)?;
            let report = verify_trust_root(&root, prior_root.as_ref(), now.as_deref())?;
            if !report.verified {
                return Err(trust_root_failure(&report)
                    .with_details(serde_json::to_value(&report).unwrap_or_default()));
            }
            if json {
                print_json(&report)?;
            } else {
                println!("trust root {} version {}", report.publisher, report.version);
                println!("  self-signed:        {}", report.self_signed);
                println!("  signed by prior:    {:?}", report.signed_by_prior_root);
                println!("  version advanced:   {:?}", report.version_advanced);
                println!("  within validity:    {:?}", report.within_validity);
                println!("  first contact:      {}", report.first_contact);
                for note in &report.assumptions {
                    println!("  assumption: {note}");
                }
                for issue in &report.issues {
                    println!("  issue: {issue}");
                }
                println!(
                    "{}",
                    if report.verified {
                        "VERIFIED"
                    } else {
                        "NOT VERIFIED"
                    }
                );
            }
        }
    }
    Ok(())
}

fn run_release(command: ReleaseCommand) -> std::result::Result<(), CliFailure> {
    use annpack::release::{
        CHANNEL_STATE_SCHEMA_V1, ChannelState, CurrentRelease, MAX_CHANNEL_STATE_FILE_BYTES,
        Revocation, Supersession, load_retained_state, persist_retained_state, sign_channel_state,
        state_to_retain, verify_channel_state,
    };
    use annpack::trust::{MAX_TRUST_ROOT_FILE_BYTES, TrustRoot, verify_trust_root};

    match command {
        ReleaseCommand::Statement {
            output,
            publisher,
            corpus,
            channel,
            sequence,
            current_root,
            current_version,
            issued_at,
            valid_until,
            superseded,
            revoked,
            revoke_reason,
        } => {
            annpack::trust::parse_utc_timestamp(&valid_until)?;
            let issued_at = match issued_at {
                Some(value) => {
                    annpack::trust::parse_utc_timestamp(&value)?;
                    value
                }
                None => annpack::trust::format_utc_timestamp(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map_err(|_| {
                            AnnpackError::InvalidInput("system clock is before 1970".into())
                        })?
                        .as_secs() as i64,
                ),
            };
            let statement = ChannelState {
                schema: CHANNEL_STATE_SCHEMA_V1.into(),
                publisher,
                corpus,
                channel,
                sequence,
                issued_at: issued_at.clone(),
                valid_until,
                current: CurrentRelease {
                    version: current_version,
                    artifact_root: current_root.to_lowercase(),
                },
                superseded: superseded
                    .into_iter()
                    .map(|root| Supersession {
                        artifact_root: root.to_lowercase(),
                        by: current_root.to_lowercase(),
                        at: issued_at.clone(),
                    })
                    .collect(),
                revoked: revoked
                    .into_iter()
                    .map(|root| Revocation {
                        artifact_root: root.to_lowercase(),
                        at: issued_at.clone(),
                        reason: revoke_reason.clone(),
                    })
                    .collect(),
                signatures: Vec::new(),
            };
            write_or_print(Some(&output), &statement)?;
            eprintln!(
                "wrote unsigned statement to {}; sign it with `annpack release sign`",
                output.display()
            );
        }
        ReleaseCommand::Sign { input, key, output } => {
            let mut statement: ChannelState =
                read_json(&input, MAX_CHANNEL_STATE_FILE_BYTES, "channel state")?;
            let key_id = sign_channel_state(&mut statement, &read_secret_key(&key)?)?;
            write_or_print(Some(output.as_deref().unwrap_or(&input)), &statement)?;
            eprintln!("signed by {key_id}");
        }
        ReleaseCommand::Verify {
            input,
            trust_root,
            scope,
            retained_state,
            clock,
            accept,
            json,
        } => {
            set_json_output(json);
            let now = resolve_clock(&clock)?;
            // Checked before any work: `--accept` writes an acceptance time into
            // durable state that later decisions read, and taking that from a
            // clock nobody vouched for would persist an unverified value. This
            // has to be its own check rather than a branch inside the success
            // path, because without a clock nothing verifies and such a branch
            // would be unreachable -- which is what the first version was.
            if accept && now.is_none() {
                return Err(CliFailure::new(
                    exit::USAGE,
                    "invalid_usage",
                    "usage",
                    "--accept requires a stated clock (--now or --system-clock)",
                ));
            }
            let statement: ChannelState =
                read_json(&input, MAX_CHANNEL_STATE_FILE_BYTES, "channel state")?;
            let root: TrustRoot = read_json(&trust_root, MAX_TRUST_ROOT_FILE_BYTES, "trust root")?;
            let trust = verify_trust_root(&root, None, now.as_deref())?;
            let retained = retained_state
                .as_deref()
                .map(load_retained_state)
                .transpose()?
                .flatten();

            // Publisher from the trusted root, corpus and channel from the
            // caller. None of the three is read from the statement.
            let expect_publisher = scope
                .expect_publisher
                .clone()
                .unwrap_or_else(|| root.publisher.clone());
            let report = verify_channel_state(
                &statement,
                &root,
                &trust,
                retained.as_ref(),
                now.as_deref(),
                (
                    &expect_publisher,
                    &scope.expect_corpus,
                    &scope.expect_channel,
                ),
            )?;

            if !report.verified {
                return Err(channel_state_failure(&report)
                    .with_details(serde_json::to_value(&report).unwrap_or_default()));
            }
            if json {
                print_json(&report)?;
            } else {
                println!(
                    "{}/{}/{} sequence {}",
                    report.publisher, report.corpus, report.channel, report.sequence
                );
                println!("  authority:        {:?}", report.authority);
                println!("  sequence verdict: {:?}", report.sequence_verdict);
                println!("  within validity:  {:?}", report.within_validity);
                for note in &report.assumptions {
                    println!("  assumption: {note}");
                }
                for issue in &report.issues {
                    println!("  issue: {issue}");
                }
                println!(
                    "{}",
                    if report.verified {
                        "VERIFIED"
                    } else {
                        "NOT VERIFIED"
                    }
                );
            }

            if report.verified
                && accept
                && let (Some(path), Some(now)) = (retained_state.as_deref(), now.as_deref())
            {
                match state_to_retain(
                    &statement,
                    &report,
                    (
                        &expect_publisher,
                        &scope.expect_corpus,
                        &scope.expect_channel,
                    ),
                    now,
                ) {
                    Some(state) => {
                        persist_retained_state(path, &state)?;
                        eprintln!(
                            "retained sequence {} for this channel",
                            state.highest_sequence
                        );
                    }
                    None => eprintln!("nothing to retain: the sequence did not advance"),
                }
            }
        }
    }
    Ok(())
}

/// Classify a build-provenance verification failure by its first unmet
/// property, most severe first: a broken envelope or untrusted signer outranks
/// a binding mismatch, since the bindings are meaningless without them.
fn provenance_failure(report: &annpack::provenance::BuildProvenanceVerification) -> CliFailure {
    use annpack::provenance::{
        BindingStatus, BuilderIdentity, EnvelopeSignature, SourceDigestBinding,
    };

    let detail = report.issues.join("; ");
    let (class, kind, stage) = if !report.predicate_type_supported {
        (exit::INPUT, "unsupported_predicate", "schema")
    } else if !report.subject_valid {
        (exit::INPUT, "malformed_input", "subject")
    } else if matches!(
        report.envelope_signature,
        EnvelopeSignature::Unsigned | EnvelopeSignature::Invalid
    ) {
        (exit::VERIFICATION, "invalid_signature", "envelope")
    } else if report.builder_identity != BuilderIdentity::Trusted {
        (exit::VERIFICATION, "untrusted_builder", "builder")
    } else if report.artifact_integrity != BindingStatus::Verified {
        (exit::VERIFICATION, "integrity_failed", "artifact")
    } else if report.distributed_file_digest != BindingStatus::Verified {
        (exit::VERIFICATION, "file_digest_mismatch", "subject")
    } else if report.artifact_root_binding != BindingStatus::Verified {
        (exit::VERIFICATION, "artifact_root_mismatch", "artifact")
    } else if report.logical_root_binding == BindingStatus::Mismatched {
        (exit::VERIFICATION, "logical_root_mismatch", "artifact")
    } else if report.builder_binary_binding == BindingStatus::Mismatched {
        (exit::VERIFICATION, "builder_binary_mismatch", "builder")
    } else if report.builder_version_binding == BindingStatus::Mismatched {
        (exit::VERIFICATION, "builder_version_mismatch", "builder")
    } else if matches!(
        report.source_digest_binding,
        SourceDigestBinding::Mismatched | SourceDigestBinding::Missing
    ) {
        (exit::VERIFICATION, "source_digest_mismatch", "source")
    } else {
        (exit::VERIFICATION, "verification_failed", "provenance")
    };
    CliFailure::new(class, kind, stage, format!("provenance rejected: {detail}"))
}

fn run_provenance(command: ProvenanceCommand) -> std::result::Result<(), CliFailure> {
    use annpack::provenance::{
        BuildProvenanceInput, Envelope, Statement, create_build_provenance,
        create_legacy_build_provenance, sign_provenance, verify_build_provenance,
    };

    match command {
        ProvenanceCommand::Create {
            artifact,
            output,
            repository,
            revision,
            builder_id,
            builder_binary,
            invocation_id,
            started_at,
            finished_at,
            system_clock,
            parameters,
            environment,
            platform,
            locked,
            legacy,
            legacy_source_digest,
        } => {
            let now = if system_clock {
                Some(annpack::trust::format_utc_timestamp(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map_err(|_| {
                            CliFailure::new(
                                exit::OPERATIONAL,
                                "io_failure",
                                "clock",
                                "system clock is before 1970",
                            )
                        })?
                        .as_secs() as i64,
                ))
            } else {
                None
            };
            let started_at = started_at.or_else(|| now.clone()).ok_or_else(|| {
                CliFailure::new(
                    exit::USAGE,
                    "invalid_usage",
                    "usage",
                    "--started-at is required unless --system-clock is given",
                )
            })?;
            let finished_at = finished_at.or_else(|| now.clone()).ok_or_else(|| {
                CliFailure::new(
                    exit::USAGE,
                    "invalid_usage",
                    "usage",
                    "--finished-at is required unless --system-clock is given",
                )
            })?;

            let parse_pairs = |pairs: Vec<String>, label: &str| {
                pairs
                    .into_iter()
                    .map(|pair| {
                        pair.split_once('=')
                            .map(|(k, v)| (k.to_string(), v.to_string()))
                            .ok_or_else(|| {
                                CliFailure::new(
                                    exit::USAGE,
                                    "invalid_usage",
                                    "usage",
                                    format!("--{label} entries must be key=value, got {pair:?}"),
                                )
                            })
                    })
                    .collect::<std::result::Result<std::collections::BTreeMap<_, _>, _>>()
            };
            let parameters = parse_pairs(parameters, "param")?;
            let environment = parse_pairs(environment, "env")?;

            let invocation_id = invocation_id.unwrap_or_else(|| {
                #[cfg(feature = "signing")]
                {
                    use rand::RngCore;
                    let mut bytes = [0_u8; 16];
                    rand::rngs::OsRng.fill_bytes(&mut bytes);
                    hex::encode(bytes)
                }
                #[cfg(not(feature = "signing"))]
                {
                    format!("{:x}", std::process::id())
                }
            });

            let input = BuildProvenanceInput {
                artifact_path: &artifact,
                repository,
                revision,
                builder_id,
                builder_binary_path: builder_binary.as_deref(),
                invocation_id,
                started_at,
                finished_at,
                parameters,
                environment,
                platform,
                locked,
            };
            let statement: Statement = if legacy {
                let digest = legacy_source_digest.ok_or_else(|| {
                    CliFailure::new(
                        exit::USAGE,
                        "invalid_usage",
                        "usage",
                        "--legacy requires --legacy-source-digest",
                    )
                })?;
                create_legacy_build_provenance(input, digest)?
            } else {
                create_build_provenance(input)?
            };
            write_or_print(Some(&output), &statement)?;
            eprintln!(
                "wrote unsigned provenance to {}; sign it with `annpack provenance sign`",
                output.display()
            );
        }
        ProvenanceCommand::Sign { input, key, output } => {
            let statement: Statement = read_json(&input, 4 * 1024 * 1024, "provenance statement")?;
            let envelope = sign_provenance(&statement, &read_secret_key(&key)?)?;
            write_or_print(Some(output.as_deref().unwrap_or(&input)), &envelope)?;
            eprintln!(
                "signed by {}",
                envelope
                    .signatures
                    .first()
                    .map(|s| s.keyid.as_str())
                    .unwrap_or("?")
            );
        }
        ProvenanceCommand::Verify {
            artifact,
            provenance,
            trusted_builder_keys,
            builder_binary,
            json,
        } => {
            set_json_output(json);
            let envelope: Envelope =
                read_json(&provenance, 4 * 1024 * 1024, "provenance envelope")?;
            let report = verify_build_provenance(
                &envelope,
                &artifact,
                &trusted_builder_keys,
                builder_binary.as_deref(),
            )?;

            if !report.verified {
                return Err(provenance_failure(&report)
                    .with_details(serde_json::to_value(&report).unwrap_or_default()));
            }
            if json {
                print_json(&report)?;
            } else {
                println!("provenance for {}", artifact.display());
                println!("  envelope signature:      {:?}", report.envelope_signature);
                println!("  builder identity:        {:?}", report.builder_identity);
                println!("  artifact integrity:      {:?}", report.artifact_integrity);
                println!(
                    "  distributed file digest: {:?}",
                    report.distributed_file_digest
                );
                println!(
                    "  artifact root binding:   {:?}",
                    report.artifact_root_binding
                );
                println!(
                    "  logical root binding:    {:?}",
                    report.logical_root_binding
                );
                println!(
                    "  source digest binding:   {:?}",
                    report.source_digest_binding
                );
                println!(
                    "  builder binary binding:  {:?}",
                    report.builder_binary_binding
                );
                println!(
                    "  builder version binding: {:?}",
                    report.builder_version_binding
                );
                println!(
                    "  repository claim:        {:?} (carried, not proven)",
                    report.repository_claim
                );
                println!(
                    "  revision claim:          {:?} (carried, not proven)",
                    report.revision_claim
                );
                for note in &report.assumptions {
                    println!("  assumption: {note}");
                }
                println!("completeness: {:?}", report.completeness);
                println!("VERIFIED");
            }
        }
    }
    Ok(())
}

/// Gather stages A–C and evaluate the requested policy.
///
/// Every input is optional because a weaker policy legitimately needs none of
/// them. What must not happen is a stronger policy quietly succeeding when its
/// inputs are absent, so the missing pieces are passed through as `None` and the
/// policy engine names them as unmet rather than this function substituting
/// defaults.
#[allow(clippy::too_many_arguments)]
fn evaluate_artifact_policy(
    artifact_root: &str,
    signatures: &[annpack::signing::SignatureReport],
    policy: annpack::policy::TrustPolicy,
    trust_root: Option<&Path>,
    channel_state: Option<&Path>,
    retained_state: Option<&Path>,
    expect: (Option<&str>, Option<&str>, Option<&str>),
    clock: &ClockArgs,
) -> Result<annpack::policy::PolicyDecision> {
    use annpack::policy::{ArtifactIntegrity, PolicyInputs, TransparencyEvidence, evaluate_policy};
    use annpack::release::{Currency, currency_for_root};
    use annpack::trust::MAX_TRUST_ROOT_FILE_BYTES;

    let now = resolve_clock(clock)?;
    let signers: Vec<String> = signatures
        .iter()
        .filter(|signature| signature.cryptographically_valid)
        .map(|signature| signature.key_id.clone())
        .collect();

    let trust_document = trust_root
        .map(|path| {
            read_json::<annpack::trust::TrustRoot>(path, MAX_TRUST_ROOT_FILE_BYTES, "trust root")
        })
        .transpose()?;
    let trust_verification = trust_document
        .as_ref()
        .map(|root| annpack::trust::verify_trust_root(root, None, now.as_deref()))
        .transpose()?;

    let statement = channel_state
        .map(|path| {
            read_json::<annpack::release::ChannelState>(
                path,
                annpack::release::MAX_CHANNEL_STATE_FILE_BYTES,
                "channel state",
            )
        })
        .transpose()?;
    let retained = retained_state
        .map(annpack::release::load_retained_state)
        .transpose()?
        .flatten();

    let state_verification = match (&statement, &trust_document, &trust_verification) {
        (Some(statement), Some(root), Some(trust)) => {
            // Corpus and channel must be stated by the caller; the publisher
            // comes from the trusted root. Reading any of them from the
            // statement would compare it only against itself.
            let (publisher, corpus, channel) = expect;
            let (Some(corpus), Some(channel)) = (corpus, channel) else {
                return Err(AnnpackError::InvalidInput(
                    "--channel-state requires --expect-corpus and --expect-channel".into(),
                ));
            };
            let publisher = publisher.unwrap_or(root.publisher.as_str());
            Some(annpack::release::verify_channel_state(
                statement,
                root,
                trust,
                retained.as_ref(),
                now.as_deref(),
                (publisher, corpus, channel),
            )?)
        }
        _ => None,
    };

    let currency = match (&statement, &state_verification) {
        (Some(statement), Some(verification)) => {
            currency_for_root(statement, verification, artifact_root)
        }
        _ => Currency::Unknown,
    };

    Ok(evaluate_policy(
        &PolicyInputs {
            artifact_root,
            artifact_integrity: ArtifactIntegrity::Valid,
            artifact_signers: &signers,
            trust: trust_verification.as_ref(),
            channel_state: state_verification.as_ref(),
            currency,
            // Stage F is not implemented. Reporting it as unavailable is what
            // makes the witnessed policy deny instead of silently degrading.
            transparency: TransparencyEvidence::Unavailable,
        },
        policy,
    ))
}

/// Resolve the caller's stated clock, or `None` when they stated none.
fn resolve_clock(clock: &ClockArgs) -> Result<Option<String>> {
    if let Some(now) = &clock.now {
        // Parse eagerly so a malformed value fails here rather than being
        // reported as an expiry problem later.
        annpack::trust::parse_utc_timestamp(now)?;
        return Ok(Some(now.clone()));
    }
    if clock.system_clock {
        let seconds = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|_| AnnpackError::InvalidInput("system clock is before 1970".into()))?
            .as_secs() as i64;
        return Ok(Some(annpack::trust::format_utc_timestamp(seconds)));
    }
    Ok(None)
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path, limit: u64, label: &str) -> Result<T> {
    let bytes = fs::metadata(path)?.len();
    if bytes > limit {
        return Err(AnnpackError::InvalidInput(format!(
            "{label} is {bytes} bytes, above the {limit} byte limit"
        )));
    }
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

/// Read a 32-byte Ed25519 public key from a `.pub` file.
fn read_public_key(path: &Path) -> Result<String> {
    let text = fs::read_to_string(path)?;
    let hex_value = text.trim();
    if hex_value.len() != 64 || !hex_value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(AnnpackError::InvalidInput(format!(
            "{} does not contain a 64-character hex Ed25519 public key",
            path.display()
        )));
    }
    Ok(hex_value.to_lowercase())
}

fn read_secret_key(path: &Path) -> Result<[u8; 32]> {
    let text = fs::read_to_string(path)?;
    let bytes = hex::decode(text.trim())
        .map_err(|_| AnnpackError::InvalidInput("secret key is not valid hex".into()))?;
    bytes
        .try_into()
        .map_err(|_| AnnpackError::InvalidInput("secret key is not 32 bytes".into()))
}

fn role_from_key_files(
    paths: &[PathBuf],
    threshold: u32,
    keys: &mut std::collections::BTreeMap<String, annpack::trust::KeyDescriptor>,
) -> Result<annpack::trust::RoleDescriptor> {
    let mut ids = Vec::new();
    for path in paths {
        let public_key = read_public_key(path)?;
        let decoded = hex::decode(&public_key).expect("validated hex");
        let key_id = blake3::hash(&decoded).to_hex().to_string();
        keys.insert(
            key_id.clone(),
            annpack::trust::KeyDescriptor {
                algorithm: "Ed25519".into(),
                public_key,
            },
        );
        if !ids.contains(&key_id) {
            ids.push(key_id);
        }
    }
    Ok(annpack::trust::RoleDescriptor {
        threshold,
        keys: ids,
    })
}

fn read_answer(path: PathBuf) -> Result<String> {
    // Bound the file before reading it: the answer is carried verbatim into the
    // bundle, so an unbounded read here becomes an unbounded bundle.
    let bytes = fs::metadata(&path)?.len();
    if bytes > annpack::bundle::MAX_BUNDLE_ANSWER_BYTES {
        return Err(AnnpackError::InvalidInput(format!(
            "answer is {bytes} bytes, above the {} byte limit",
            annpack::bundle::MAX_BUNDLE_ANSWER_BYTES
        )));
    }
    Ok(fs::read_to_string(path)?)
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
