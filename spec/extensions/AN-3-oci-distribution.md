# AN-3: OCI Distribution

Status: implemented draft. Requires Adyar Core v1.0-draft.

AN-3 maps one pack to one OCI artifact layer as defined by [the media-type registry](../MEDIA-TYPES.md). Clients verify OCI SHA-256 descriptors and then independently verify the Adyar BLAKE3 root and sections.

The reference transport supports anonymous access, Basic credentials supplied outside command arguments, and standard Bearer challenges. It refuses credential disclosure over insecure non-loopback transport, rejects insecure authentication realms, does not forward authorization to foreign upload origins, bounds token responses, and verifies downloads before atomic installation.

AN-3 is transport only. A Core implementation does not need an OCI client.
