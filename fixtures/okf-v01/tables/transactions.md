---
type: BigQuery Table
title: Ledger Transactions
description: One row per immutable ledger transaction.
resource: https://example.test/bigquery/ledger/transactions
tags: [accounting, transactions, ap-104]
timestamp: '2026-07-19T00:00:00Z'
---

# Schema

`transaction_id` is the stable identifier. `posted_at` records the commit time.

# Operational knowledge

Error `AP-104` means the transaction signature was produced for an obsolete ledger revision.

# Citations

[1] [Ledger API](https://example.test/ledger-api)
