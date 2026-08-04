---
title: Key rotation API
url: https://vendor.example/docs/v2/key-rotation
---
# Key rotation

Version 2 replaces the boolean rotation flag with explicit options.

## rotateKey

`rotateKey()` accepts a `RotationOptions` object. Set `invalidatePrevious` to `true` to force immediate invalidation.

```typescript
const replacement = await client.rotateKey({
  invalidatePrevious: true,
  reason: "scheduled rotation"
});
```

Passing a boolean is invalid in version 2.

