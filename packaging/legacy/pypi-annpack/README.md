# annpack

**ANNPack is now [Adyar](https://github.com/Arjun2729/ANNPACK).**

```bash
pip install adyar
```

This package is the final release under the old name. It forwards every call to
`adyar` and warns on import.

The wire format did not change with the rename. Artifacts and receipts produced
by ANNPack remain valid and verify unchanged; the magic bytes, schema strings,
media types and predicate types all name a format version rather than a project.
