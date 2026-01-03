# Verify + Sign

ANNPack includes CLI checks for integrity and optional signing.

## Verify a pack
```bash
annpack verify ./out/tiny
```

## Inspect a pack
```bash
annpack inspect ./out/tiny/pack.annpack
```

## Sign a pack
```bash
annpack sign ./out/tiny --key ./keys/ed25519.key --out ./out/tiny/signatures.json
```

## Notes
- Verification checks manifest structure, file presence, and bounds.
- Signatures cover manifest contents; keep keys out of git.
