---
title: Authentication errors
url: https://vendor.example/docs/v1/errors
product: Example SDK
---
# Authentication errors

The authentication service reports stable error codes so applications can react without parsing prose.

## AP-104

`AP-104` means that the API key has expired. Generate a replacement key, deploy it to the client, and revoke the expired credential.

Do not retry an expired key. Requests using it will continue to fail until the credential is replaced.

## AP-207

`AP-207` means the account has exceeded its authentication-rate limit. Retry with exponential backoff.

