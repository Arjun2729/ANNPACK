# ADR-0010: External source adapters produce deterministic canonical input

Status: accepted, 2026-08-10. Locks decisions and boundaries only; the
canonical schema and locator grammar are later, normative work — see
"What this does not do."

## Context

The compiler accepts Markdown/MDX and OKF today. Every additional source
system — Confluence, SharePoint, HTML, OpenAPI, GitBook, PDF — raises the
same question: what does the compiler actually consume from it.

The wrong answer is one the codebase already knows how to avoid: teach the
compiler each source system directly. That is the same shape of mistake
ADR-0009 rejected for fleet policy (folding a second party's concern into
an existing authority) and ADR-0006 rejected for provenance (letting a
narrower mechanism absorb a broader one it doesn't fit) — a component
accreting responsibilities that belong to something else, because nothing
drew the boundary first.

Two properties of the existing pipeline are easy to lose crossing this
boundary if it is not stated explicitly:

**Source binding is authenticated, not asserted.** ADR-0005 made every
artifact commit to a digest of the exact bytes it was built from, precisely
so a source claim cannot diverge from what was actually consumed. That
digest is presently computed over a git-checked-out tree. Confluence,
SharePoint, and PDF sources are not git trees, and a locator borrowed from
one of them — a page ID, a mutable URL — is not by itself a digest of
anything. Whatever crosses this boundary has to preserve "authenticated,
not asserted," not merely "labeled with where it came from."

**The build is deterministic by construction.** Same inputs, same bytes,
always — enforced today because ingestion reads a fixed tree. An adapter
that calls a live API or parses a PDF is not automatically deterministic:
API response ordering, fetch timestamps, and locale-dependent formatting
are all real ways for "the same" source to produce different bytes on two
runs. If adapter output can drift, the compiler's determinism guarantee is
only as strong as the least careful adapter feeding it, regardless of how
carefully the compiler itself is written.

## Decision

### Adapters convert source systems into one canonical input representation, outside the core compiler

Confluence, HTML, OpenAPI, GitBook, PDF, SharePoint, and any future source
each get their own adapter. The compiler consumes exactly one input shape,
regardless of how many adapters exist. Adapters are not part of the core
compiler crate: each brings its own dependencies (an HTTP client, a PDF
parser) that the compiler has no reason to carry, and a bug in one adapter
cannot be a compiler bug.

### The source locator is scheme-based and opaque to the compiler

Every canonical document and addressable block may carry a source locator.
Its syntax is `scheme:...`; the compiler preserves and authenticates it as
an opaque string and never branches on the scheme. Illustrative, not
normative — the grammar is later work:

```text
git:<repository>@<revision>#<path>
confluence:<site>/<page-id>@<version>#<block>
sharepoint:<site>/<document-id>@<revision>#<fragment>
pdf:<document-id>@<digest>#page=7&bbox=...
openapi:<document-id>@<digest>#operationId=getUser
```

A compiler with `if source == Confluence { … }` anywhere in it has already
violated this decision, no matter how the branch got there.

### Source identity is separate from source location

A locator says where a block claims to come from. It does not by itself
prove anything — a mutable URL or a page ID can be repointed after the
fact. Source identity is the authenticated digest over the exact canonical
input bytes, in the same sense ADR-0005 already established for the
ingested tree. The locator travels alongside the digest; it never
substitutes for it.

### Adapter determinism is a byte-identical requirement, not an aspiration

Given the same immutable source revision and the same adapter
version/configuration, an adapter MUST emit byte-identical canonical
input. This is the same standard the ingestion pipeline already meets for
git trees, restated for adapters that do not have a git tree to rely on.

Adapters normalize or exclude unstable data: fetch timestamps, API
response ordering, incidental whitespace variation, temporary IDs,
pagination order, locale-dependent formatting, and any other metadata that
varies between two fetches of what is meant to be the same content.

If the source system cannot provide an immutable revision or snapshot, the
adapter has exactly two honest options: capture enough of the source
material itself to compute an immutable snapshot digest, or report that
repeatability from the upstream source cannot be guaranteed. Silently
proceeding as though the source were stable is not a third option.

### Adapter provenance binds four things

An adapter run's provenance names, at minimum:

```text
adapter identity/version
adapter configuration digest
source-system locator/revision
canonical output digest
```

This is what makes the chain checkable end to end:

```text
external source revision
  → adapter
  → canonical input bytes
  → authenticated canonical digest
  → Adyar compiler
  → artifact
```

Each arrow is a place something could have altered the content; each
binding is what makes that alteration detectable rather than assumed
absent.

### The canonical input model is a compiler IR, not a universal document ontology

It captures what the compiler needs to compile faithfully — document
identity, title, hierarchy, ordered blocks, block type, text/code/table
content, source locator, stable source metadata, optional relationships —
and stops there. No arbitrary graph semantics, no source-system ACL model,
no workflow state, no attempt to preserve every concept Confluence,
SharePoint, or OpenAPI can express. A source system that can represent
something the canonical model cannot is not a bug in the canonical model.

## What this does not do

Define the canonical schema or the locator grammar as normative. The field
list and the example schemes above are illustrative. The actual spec is
later work, written once at least two genuinely different adapters exist
to write it against — one adapter is a shape nobody has generalized from
yet.

Specify how an adapter obtains credentials for its source system, how
adapter configuration is distributed or versioned, or how canonical input
gets from an adapter to the compiler (a file on disk, a directory, a
stream — unspecified here).

Change anything about the existing Markdown/MDX/OKF ingestion path. That
path already satisfies this ADR's requirements for the sources it handles
today; it is not required to be reimplemented as an adapter to remain
conforming.

## Alternatives rejected

**Compiler-native support per source system.** Rejected: unbounded growth
in compiler surface and dependencies, one bug class per source system
instead of one, and the exact "component absorbs a concern that belongs to
something else" mistake ADR-0006 and ADR-0009 already rejected in their
own contexts.

**A universal document model covering source-system semantics generally
(ACLs, workflow state, arbitrary relationship graphs).** Rejected: that is
an ontology-design problem, not a compiler-input-format problem, and it
has no natural stopping point — every additional source system would
motivate extending it further. The canonical model's job is to compile,
not to represent.

**Treating a source locator as sufficient provenance on its own.**
Rejected: a locator is a claim about origin, not evidence of it. Without a
digest binding, a mutable URL or reassignable page ID can be repointed
after the fact with nothing to detect it — exactly the gap ADR-0005 closed
for git-sourced content.

**Leaving adapter determinism as an informal expectation.** Rejected:
"should be deterministic" is unfalsifiable and has no compliance
criterion. Stating the byte-identical requirement, and the concrete list
of what an adapter must normalize or exclude, gives adapter authors and
reviewers something to actually check an adapter against.
