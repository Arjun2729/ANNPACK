# AN-2: Verified deltas

Status: implemented draft. Requires Adyar Core v1.0-draft.

AN-2 update artifacts use media type `application/vnd.annpack.delta.v1`. Every delta binds an exact base BLAKE3 root and target root. Codec zero carries a target snapshot. Codec one carries bounded copy/add operations over the exact verified base.

Readers bound target size and operation count before allocation, reject zero-length or out-of-range operations, reconstruct into a temporary sibling, parse and fully verify the resulting Core pack, compare the target root, and only then install it atomically. AN-2 does not define mutable patch chains, conflict resolution, or federation graphs.
