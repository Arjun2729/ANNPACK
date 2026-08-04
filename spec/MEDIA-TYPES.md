# Candidate ANNPack Media Types and OCI Mapping

The following identifiers are unregistered candidate media types:

```text
application/vnd.annpack.v3
application/vnd.annpack.manifest+json
application/vnd.annpack.discovery+json
application/vnd.annpack.delta.v1
application/vnd.annpack.config.v1+json
```

HTTP publishers SHOULD serve `.annpack` using `application/vnd.annpack.v3` without transfer compression so stored byte offsets remain stable.

## OCI

ANNPack can be represented as an OCI artifact:

- OCI manifest media type: `application/vnd.oci.image.manifest.v1+json`
- `artifactType`: `application/vnd.annpack.v3`
- Empty JSON config: `application/vnd.annpack.config.v1+json`
- One pack layer: `application/vnd.annpack.v3`

Recommended annotations:

```text
org.opencontainers.image.title
org.opencontainers.image.version
org.opencontainers.image.revision
dev.annpack.root
```

OCI requires SHA-256 descriptor digests. ANNPack retains its BLAKE3 content root as the protocol identity; the two hashes serve different container layers and both are emitted by the reference CLI.

The reference client implements the OCI Distribution `POST`/`PUT` blob-upload sequence, manifest push, and manifest/blob pull. It accepts `REGISTRY/REPOSITORY:TAG`, `oci://...`, and exact `@sha256:...` references; insecure HTTP is selected only by an explicit `http://` reference or a loopback registry.
