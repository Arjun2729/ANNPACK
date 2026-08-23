# Adjudication review of `okf-hard-negatives`

Every query/passage pair was read and compared against its target. This is a
machine-assisted review, **not** the human adjudication the corpus still needs —
`README.md` is right that generated queries without human judgment are not an
honest evaluation, and a machine reviewing machine-authored pairs does not
change that. What it does is make the human pass cheap: the labels are sound, so
the remaining work is judgment about corpus *design*, not error-hunting.

## Labels: 63/63 sound, no mislabels found

Every technical-token query names identifiers that appear in its target. Every
hard-negative query is a fair paraphrase of its target's subject.

One pair had been suspected of being mislabelled: `technical-token-143`
("users table display_name reputation schema") was missed by lexical, vector and
hybrid simultaneously, which usually indicates a bad label. It is correct — the
target contains both `reputation` and `display_name`. The suspicion came from
reasoning about the failure rather than reading the passage.

**So label quality is not the reason this corpus cannot resolve small effects.**
Size and stratum design are.

## The hard-negative stratum tests one narrow skill

21 of 35 hard-negative targets are table or dataset *overview* passages, so the
query reduces to "which table is this, described in other words":

```
hn-14  listing holding coin references already used up   -> inputs table overview
hn-21  catalogue of newly minted spendable coin slots    -> outputs table overview
hn-116 enquiries raised by members plus assorted details -> posts_questions overview
```

The remaining 14 target audience definitions, metric definitions and query
patterns, which are more varied but still descriptive.

This is a legitimate test of synonym matching over descriptions. It is not a
test of retrieval generally, and it explains the observed mode behaviour: the
stratum was built by rejecting any candidate sharing a discriminative token, so
lexical scores 1/35 by construction while vector scores 24/35 by doing the one
thing the stratum asks for.

## The corpus cannot support the axes it is missing

Adding negation, temporal, quantity and exception strata was the obvious next
step. The source material does not allow it. Passages containing each pattern:

| axis | passages | usable |
|---|---|---|
| negation | 27/168 | mostly incidental (`does not have a...`), not semantic negation |
| quantity / threshold | 11/168 | a handful of audience definitions |
| temporal | 6/168 | too few to form a stratum |
| exception | 2/168 | not viable |

These are BigQuery schema reference documents. They describe what tables contain.
They almost never say what is *excluded*, what happens *after 30 days*, or what
applies *unless* some condition holds — because that is not what reference
documentation does.

**Writing such queries against this corpus would mean inventing content the
passages do not contain, and labelling the nearest passage as correct.** That
produces questions with no right answer, which is worse than having no stratum.

## What this means

This corpus is close to fully exploited. It can distinguish lexical from
semantic retrieval on descriptive paraphrase, and it has done so. It cannot
answer whether one encoder, extension, or fusion policy is better than another,
because those differences are 1-2 queries wide and 63 queries with one stratum
axis cannot resolve them.

The next corpus needs **different source material**, not more queries over this
one. Documents that state conditions, exclusions, limits and time windows —
policy text, terms of service, changelogs, support documentation — support the
axes reference documentation cannot.

Until then, treat published numbers here as characterising this corpus, and do
not read a two-query difference as a quality difference.
