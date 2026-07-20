---
title: Authentication errors
url: https://vendor.example/docs/v2/errors
product: Example SDK
---
# Authentication errors

Version 2 validates signed requests and reports stable error codes for unsupported cryptographic inputs.

## AP-104

`AP-104` means that the request signature uses an unsupported algorithm. Configure the client to sign with Ed25519 or another algorithm allowed by the account policy.

Rotating an API key does not correct this error because the rejected value is the signature algorithm.

## AP-207

`AP-207` means that the signed timestamp is outside the permitted clock-skew window.

