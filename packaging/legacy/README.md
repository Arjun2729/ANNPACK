# Terminal releases for the pre-rename packages

`annpack` on PyPI and `@annpack/node` on npm are published and will keep serving
whatever was last uploaded to them, forever, to anyone who runs the install
command in an old tutorial. Renaming the packages in this repository does not
change that. Each needs one final release whose only job is to say where the
project went.

Both shims re-export the new package rather than breaking on import. A terminal
release that raises on import punishes the people who were slowest to hear about
the rename, which is exactly the wrong audience to punish. They warn loudly, they
depend on the real package, and they are marked inactive so the registries stop
presenting them as current.

## Publishing

Neither is published by CI. Both are one-time, irreversible, outward-facing
actions against registries this repository does not own credentials for, so they
are run deliberately by a maintainer.

Publish the real `adyar` packages first. A shim that depends on a package the
registry does not yet have is broken on arrival.

```bash
# 1. PyPI
cd packaging/legacy/pypi-annpack
python3 -m build
python3 -m twine upload dist/*

# 2. npm
cd packaging/legacy/npm-annpack-node
npm publish --access public

# 3. Mark both deprecated registry-side. This is what actually greys the
#    package out in the UI and prints a warning on every install; the metadata
#    in the manifests alone does not.
npm deprecate @annpack/node "renamed to @adyar/node"
```

PyPI has no `npm deprecate` equivalent. The `Development Status :: 7 - Inactive`
classifier and the description are the available signals; do not yank the older
releases, which would break existing pinned installs without helping anyone.

## Version

Both shims are versioned one patch above the last real release under the old
name, so they sort above it and become what a fresh `pip install annpack` or
`npm install @annpack/node` resolves to.
