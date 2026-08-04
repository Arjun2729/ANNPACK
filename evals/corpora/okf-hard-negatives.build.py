#!/usr/bin/env python3
"""Build the OKF hard-negative evaluation query set.

Two strata, so the comparison measures something a single stratum cannot:

  technical-token — the query uses the passage's own distinctive identifiers.
                    Lexical retrieval should be strong here, and the question is
                    whether adding vectors *costs* anything.

  hard-negative   — the query is a paraphrase that shares **no discriminative
                    token** with the target passage under the normative
                    tokenizer (FORMAT-v3 §6.1). Lexical retrieval has nothing to
                    rank the target on, so the question is whether vectors
                    recover it.

"Discriminative" is defined against the corpus, not by taste: a token appearing
in at least 20% of passages is corpus-common, and BM25 gives it an idf near
zero, so sharing "the" or "table" cannot rank anything. Sharing any rarer token
can. The check below computes document frequencies from the corpus itself and
rejects any hard-negative candidate sharing a token below that threshold, so the
candidate must be rewritten. That makes "hard" an objective property of the data
even though the queries themselves are machine-authored.

WHAT THIS QUERY SET IS NOT
--------------------------
The queries and the relevance labels are machine-authored. `evals/README.md`
requires human-written queries and independent human judgments before any
retrieval-quality claim, and this does not meet that bar. It exists to answer a
narrower question that does not need human labels: on a corpus where lexical
provably has no lexical signal, does the vector path recover the target?
"""

import json
import pathlib
import subprocess
import sys
import unicodedata

ROOT = pathlib.Path(__file__).resolve().parents[2]
TECHNICAL_PUNCTUATION = frozenset("_-.:/@#")


def tokenize(text: str) -> set[str]:
    """FORMAT-v3 §6.1, mirrored so the check needs no built binary."""
    out = set()
    for raw in unicodedata.normalize("NFKC", text).lower().split():
        start, end = 0, len(raw)
        keep = lambda ch: unicodedata.category(ch)[0] in ("L", "N") or ch in TECHNICAL_PUNCTUATION
        while start < end and not keep(raw[start]):
            start += 1
        while end > start and not keep(raw[end - 1]):
            end -= 1
        if raw[start:end]:
            out.add(raw[start:end])
    return out


# (target passage index, category, query)
#
# Targets are indices into the exported passage list, which is in deterministic
# corpus order. Categories are the two strata above.
QUERIES = [
    # ── technical-token: the query names identifiers the passage contains ──
    (1, "technical-token", "blocks table hash STRING REQUIRED field schema"),
    (12, "technical-token", "dup_transaction_count standardSQL duplicate transaction query"),
    (15, "technical-token", "inputs table transaction_hash schema field"),
    (17, "technical-token", "Taproot script types adoption inputs over time"),
    (22, "technical-token", "one row per transaction output grain interpretation"),
    (25, "technical-token", "largest outputs in block number 301641 receiving addresses"),
    (26, "technical-token", "pubkeyhash scripthash locking script types popularity"),
    (34, "technical-token", "witness_v0_keyhash scripthash address type distribution"),
    (42, "technical-token", "event_date STRING NULLABLE events table field"),
    (43, "technical-token", "user_ltv.revenue FLOAT lifetime value"),
    (46, "technical-token", "unnest event_params to extract page_location"),
    (47, "technical-token", "unnest items repeated record purchase top selling products"),
    (55, "technical-token", "wildcard suffix pattern query sharded tables"),
    (73, "technical-token", "in_app_purchase purchasers audience query"),
    (75, "technical-token", "SAFE_DIVIDE AcceptedAnswerId COUNT Id formula"),
    (77, "technical-token", "COUNTIF VoteTypeId IN (4, 12) flag ratio formula"),
    (79, "technical-token", "badges table id INTEGER name STRING schema"),
    (84, "technical-token", "ON comments.post_id join relationship comments posts"),
    (87, "technical-token", "ContentLicense value date start CC BY-SA license table"),
    (98, "technical-token", "PostTypeId 1 Question 2 lookup catalog"),
    (100, "technical-token", "ON votes.post_id = posts.id join"),
    (105, "technical-token", "posts_answers parent_id posts_questions join"),
    (117, "technical-token", "posts_questions id INTEGER title STRING body schema"),
    (138, "technical-token", "stackoverflow_posts deprecated do not use for new queries"),
    (140, "technical-token", "tags table tag_name python INTEGER id"),
    (143, "technical-token", "users table display_name reputation schema"),
    (146, "technical-token", "VoteTypeId -1 InformModerator lookup"),
    (150, "technical-token", "count upvotes for a specific post"),

    # ── hard-negative: paraphrase only, verified zero token overlap ──
    (0, "hard-negative", "details kept about newly mined units of the coin ledger"),
    (2, "hard-negative", "count of units produced every 24 hours plus mean payment load"),
    (7, "hard-negative", "which freely available warehouse holds the whole coin ledger"),
    (11, "hard-negative", "why did identical entries once show up twice on the chain"),
    (14, "hard-negative", "listing holding coin references already used up"),
    (18, "hard-negative", "follow money backwards to discover its earlier origin"),
    (21, "hard-negative", "catalogue of newly minted spendable coin slots"),
    (29, "hard-negative", "every money movement logged from the earliest mined unit onward"),
    (37, "hard-negative", "gather visitors brought in by one paid advertising push"),
    (40, "hard-negative", "anonymised behavioural stream exported from the web measurement product"),
    (50, "hard-negative", "segment visitors triggering activity many separate occasions"),
    (53, "hard-negative", "the anonymised retail sample collection shipped for practice"),
    (57, "hard-negative", "shoppers who arrived seven days ago via the search giant's ads"),
    (60, "hard-negative", "people whose engaged duration exceeds a chosen threshold"),
    (64, "hard-negative", "visitors appearing sometime inside a rolling stretch of dates"),
    (67, "hard-negative", "visitors gone quiet lately though present beforehand"),
    (71, "hard-negative", "the cohort covering everyone that ever bought anything"),
    (74, "hard-negative", "what fraction of enquiries reach a settled outcome"),
    (76, "hard-negative", "how frequently do submissions get reported for poor standard"),
    (78, "hard-negative", "where are community achievement awards to members recorded"),
    (81, "hard-negative", "short remarks people leave underneath a submission"),
    (86, "hard-negative", "how reuse permissions shifted across different periods"),
    (89, "hard-negative", "a running trail of every change applied to a submission"),
    (92, "hard-negative", "rows connecting a submission with its counterpart elsewhere"),
    (107, "hard-negative", "submissions putting someone forward for a governance role"),
    (110, "hard-negative", "descriptive write-ups for labels nobody applies any more"),
    (113, "hard-negative", "explanations of what earned rights let members do"),
    (116, "hard-negative", "enquiries raised by members plus assorted particulars"),
    (124, "hard-negative", "condensed blurbs displayed beside label write-ups"),
    (127, "hard-negative", "a catch-all store for assorted encyclopedia-style entries"),
    (131, "hard-negative", "the entire open repository for a coding help forum"),
    (139, "hard-negative", "keyword markers pinned onto submissions plus usage counts"),
    (142, "hard-negative", "member profiles carrying standing plus signup particulars"),
    (145, "hard-negative", "what each ballot category signifies"),
    (148, "hard-negative", "separate ballots registered against a submission"),
]


def main() -> int:
    passages = json.loads((pathlib.Path(sys.argv[1])).read_text())
    if isinstance(passages, dict):
        passages = passages.get("passages", [])

    # Corpus document frequencies, for the discriminative-token rule above.
    frequencies: dict[str, int] = {}
    for passage in passages:
        for token in tokenize(" ".join(passage.get("heading_path", [])) + " " + passage["text"]):
            frequencies[token] = frequencies.get(token, 0) + 1
    common_threshold = 0.20 * len(passages)
    common = {t for t, c in frequencies.items() if c >= common_threshold}

    records = []
    violations = []
    for index, category, query in QUERIES:
        passage = passages[index]
        # Tokenize the passage exactly as the builder does: heading path plus text.
        passage_tokens = tokenize(" ".join(passage.get("heading_path", [])) + " " + passage["text"])
        query_tokens = tokenize(query)
        overlap = query_tokens & passage_tokens
        discriminative = overlap - common
        if category == "hard-negative" and discriminative:
            violations.append((index, query, sorted(discriminative)))
            continue
        records.append({
            "id": f"{category}-{index}",
            "category": category,
            "query": query,
            "relevant_passage_ids": [passage["id"]],
            "shared_tokens": sorted(overlap),
            "shared_discriminative_tokens": sorted(overlap - common),
        })

    if violations:
        print(f"{len(violations)} hard-negative queries share discriminative tokens with their target:", file=sys.stderr)
        for index, query, tokens in violations:
            print(f"  [{index}] {query!r} overlaps: {tokens}", file=sys.stderr)
        return 1

    output = pathlib.Path(sys.argv[2])
    output.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    hard = sum(1 for r in records if r["category"] == "hard-negative")
    print(f"wrote {len(records)} queries to {output}")
    print(f"  technical-token: {len(records) - hard}")
    print(f"  hard-negative:   {hard} (verified: no shared token with df < {common_threshold:.0f}/{len(passages)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
