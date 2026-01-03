# API Usage

## Build a pack

```python
from annpack.api import build_pack

build_pack(
    input_csv="tiny_docs.csv",
    output_dir="./out/pack",
    text_col="text",
    id_col="id",
    lists=4,
    seed=0,
    offline=True,
)
```

## Search a pack

```python
from annpack.api import open_pack

pack = open_pack("./out/pack")
try:
    results = pack.search("hello", top_k=5)
    print(results)
finally:
    pack.close()
```

For very large metadata files, you can skip eager metadata loading:

```python
pack = open_pack("./out/pack", load_meta=False)
```

Metadata is size-capped by default (1GB). Set `ANNPACK_MAX_META_BYTES` or
`ANNPACK_ALLOW_LARGE_META=1` to override in trusted environments.

## PackSets (base + deltas)

```python
from annpack.packset import build_packset_base, build_delta, update_packset_manifest
from annpack.api import open_pack

build_packset_base("tiny_docs.csv", "./packset", text_col="text", id_col="id", lists=4, seed=123, offline=True)
build_delta("./packset/base", add_csv="delta_add.csv", delete_ids=[1], out_delta_dir="./packset/deltas/0001.delta", lists=4, seed=123, offline=True)
update_packset_manifest("./packset", "./packset/deltas/0001.delta", seq=1)

pack = open_pack("./packset")
print(pack.search("delta add", top_k=3))
pack.close()
```

You can also call `open_packset()` directly:

```python
from annpack.packset import open_packset

packset = open_packset("./packset")
print(packset.search("delta add", top_k=3))
packset.close()
```

## Inspect and verify packs

```python
from annpack.verify import inspect_pack, verify_pack

print(inspect_pack("./out/pack"))
print(verify_pack("./out/pack", deep=True))
```
