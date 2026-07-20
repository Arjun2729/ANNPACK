# Independent security review brief

This is a review assignment, not a claim that an independent review occurred. The reviewer must not use the implementation's tests as the source of expected behavior; they should derive invariants from the specifications and adversarially test the binaries.

## Binary parser

- Checked arithmetic for every header, directory, section, compressed-block, posting, vector, and delta range.
- Allocation bounds applied before allocation, including counts that are plausible relative to remaining input bytes.
- Duplicate, overlapping, out-of-order, truncated, noncanonical, unknown-required, and decompression-bomb cases.
- Content-root behavior around signatures, unknown optional sections, offsets, codecs, flags, and reserved bytes.
- Deterministic JSON assumptions and passage evidence hashing across Rust and JavaScript.
- Exact parity between native, WASM, and browser rejection behavior.
- Fuzz harness reachability: confirm targets reach logical parsers rather than rejecting almost all input at the magic bytes.

## OCI authentication and transport

- Quoted Bearer challenges containing commas, escapes, duplicate keys, missing realm, hostile scopes, and oversized token bodies.
- No credential transmission to non-HTTPS non-loopback endpoints or insecure authentication realms.
- Redirect and upload `Location` handling, including scheme downgrade, userinfo, foreign origins, and authorization stripping.
- Registry reference parsing, repository traversal, tag/digest confusion, and Unicode/percent-encoding edge cases.
- Response-size limits, digest/length verification, media-type validation, temporary-file cleanup, overwrite behavior, and atomic installation.
- Proxy, DNS rebinding, loopback classification, IPv6, and redirect-policy assumptions in the HTTP library.

## Deliverable

The reviewer records the exact commit, toolchain, corpus, commands, findings, and minimized regression artifacts. A second agent session still counts as agent-assisted review, not independent implementation. Public release requires a reviewer who did not author the parser or its tests.
