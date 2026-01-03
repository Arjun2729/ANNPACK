# Security Policy

## Threat model (summary)
- **Untrusted inputs:** pack files, manifests, and deltas obtained from external sources.
- **Trusted components:** local builder/runtime binaries, your file system, and registry secrets.
- **Primary risks:** malformed packs causing crashes/CPU spikes, registry misuse via weak auth, and tampering without signature verification.

## Supported versions
- Latest release on `main` only.

## Reporting a vulnerability
Email security@annpack.dev with details. If you do not receive a response within 72 hours, open a GitHub issue with minimal details and request contact.

## Disclosure process
- Acknowledge within 72 hours.
- Provide an ETA for a fix.
- Coordinate a release and CVE if needed.

## Security guarantees
- Registry JWT auth is required when `REGISTRY_DEV_MODE` is disabled.
- Pack file paths are validated to prevent traversal.
- Upload sizes and request rates are capped by default.

## Non-guarantees
- No confidentiality guarantees for local disk storage without encryption.
- No protection against compromised client environments or poisoned models.
- No tamper-proofing beyond manifest validation and signature checks.

## Secure defaults checklist
- Enable signature verification (`annpack verify`) for packs from untrusted sources.
- Run the registry with `REGISTRY_DEV_MODE=0` and a strong `REGISTRY_JWT_SECRET`.
- Set explicit `REGISTRY_JWT_AUD` and `REGISTRY_JWT_ISS`.
- Enforce request size limits and rate limiting in the registry.
