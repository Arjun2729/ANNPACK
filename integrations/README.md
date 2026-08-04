# Static documentation integrations

These adapters make an ANNPack artifact a build output instead of a manually maintained asset. All adapters call the same Rust builder; none reimplement parsing or the binary format.

- `docusaurus/`: plugin with a `postBuild` hook
- `vitepress/`: Vite plugin with a `closeBundle` hook
- `astro/`: Astro integration using `astro:build:done`
- `mintlify/`: environment-configured post-build command

Every adapter accepts or forwards `binary`, `source`, `output`, `name`, `version`, `baseUrl`, `sourceRevision`, `license`, and extra CLI arguments. Projects should publish the output and a generated discovery document under `/.well-known/`.

The adapters intentionally index source Markdown rather than rendered HTML. Publisher build systems remain responsible for selecting the authoritative source tree and version.

