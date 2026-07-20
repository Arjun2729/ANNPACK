---
title: Key rotation API
url: https://vendor.example/docs/v1/key-rotation
---
# Key rotation

Version 1 exposes a simple synchronous rotation API.

## rotateKey

`rotateKey()` accepts a boolean `force` parameter. Pass `true` to invalidate the previous key immediately.

```typescript
const replacement = client.rotateKey(true);
```

When `force` is false, the previous credential remains valid during the configured grace period.

