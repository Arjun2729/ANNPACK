#!/usr/bin/env python3
"""
ANNPack Core v1.0-draft — a second reader, in Python.

Written against the specification text only:
  spec/CORE-v1.0-draft.md, spec/FORMAT-v3.md, spec/SECURITY.md,
  spec/PROTOCOL-v1.md, spec/EVIDENCE-v1.md

READ THIS BEFORE TREATING IT AS INDEPENDENT VALIDATION
------------------------------------------------------
This reader was written in the same session, by the same author, as the changes
to the Rust reference implementation it is checked against. It is therefore NOT
the independent implementation that Core v1.0-draft requires before the -draft
marker comes off. Shared authorship means shared blind spots: an assumption the
reference makes silently is one this reader is likely to make too, and neither
would notice.

What it does establish, which is worth something:

  * The specification is sufficient to build a working reader from — every
    constant, layout, and algorithm needed is written down.
  * The reference implementation is not relying on undocumented behaviour: a
    reader built only from the prose agrees with it on the conformance suite.
  * Every ambiguity found while writing it is recorded in AMBIGUITIES below,
    which is the actual deliverable for whoever writes the real second reader.

What it does not establish: interoperability. That still requires someone with
no access to this repository's implementation.

AMBIGUITIES AND UNDERSPECIFIED POINTS FOUND WHILE WRITING THIS
--------------------------------------------------------------
1. FORMAT-v3 §6.1 step 2 says "lowercase using Unicode simple lowercase
   mapping". Python's str.lower() implements *full* lowercase mapping, which
   differs for a handful of characters (most visibly 'İ' U+0130, which full
   mapping expands to two code points). The spec should say which it means; this
   reader uses str.lower() and notes the divergence risk.

2. FORMAT-v3 §6.1 step 4 says to trim characters that are "neither Unicode
   alphanumeric (\\p{L} or \\p{N}) nor a member of the technical punctuation
   set". Python's str.isalnum() is broader than L|N — it also accepts Nl, No and
   some Mn marks. This reader tests category membership explicitly rather than
   using isalnum(), which the spec's regex-class phrasing implies but does not
   say outright.

3. FORMAT-v3 §2 requires rejecting "an unknown required section", but does not
   enumerate which types a Core reader must recognize. This reader treats types
   1-10, 12, 16 and 17 as known and rejects any other type carrying the required
   flag. Retired types (11, 14, 15) are deliberately not in that set.

4. FORMAT-v3 §5.2 gives the id-index entry width as 36 bytes in prose but does
   not restate it as a named constant the way `stride` is carried in the block
   table. A reader must hardcode 36 while reading `stride` from the artifact,
   which is an asymmetry worth removing.

5. CORE §6 requires emitting an evidence envelope per result but does not say
   whether `canonical_url` is omitted or null when a pack has no base URL. This
   reader omits it.
"""

from __future__ import annotations

import json
import struct
import sys
import unicodedata
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import blake3

# ── FORMAT-v3 §1, §2 ──────────────────────────────────────────────────────────

MAGIC = b"ANNPACK3"
FORMAT_VERSION = 3
HEADER_SIZE = 128
ENTRY_SIZE = 80
MAX_SECTIONS = 16_384

CONTENT_ROOT_CONTEXT = b"ANNPACK3-CONTENT-ROOT\0"

# EVIDENCE-v1 domain separation. Leaves and interior nodes use distinct context
# strings so a leaf hash can never be reinterpreted as an interior node.
PASSAGE_EVIDENCE_CONTEXT = b"ANNPACK3-PASSAGE-EVIDENCE\0"
NODE_CONTEXT = b"ANNPACK3-EVIDENCE-NODE\0"
SIGNATURE_CONTEXT = b"ANNPACK3-SIGNATURE\0"
RECEIPT_SCHEMA = "annpack-receipt-v2"

# §2: decompression bounds.
DECOMP_RATIO_LIMIT = 256
DECOMP_RATIO_FLOOR = 16 * 1024 * 1024

# Section types this reader recognizes. Anything else carrying the required flag
# is rejected (§2). 11, 14 and 15 are retired and deliberately absent.
T_MANIFEST, T_DOCUMENTS, T_PASSAGE_INDEX, T_PASSAGE_DATA = 1, 2, 3, 4
T_LEX_DICT, T_LEX_POSTINGS = 5, 6
T_SIGNATURE = 10
T_LEXICAL_TERMS, T_PASSAGE_RECORDS = 16, 17
KNOWN_TYPES = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 17}
CORE_REQUIRED = {T_MANIFEST, T_DOCUMENTS, T_PASSAGE_INDEX, T_PASSAGE_DATA,
                 T_LEX_DICT, T_LEX_POSTINGS}

SUPPORTED_MANIFEST_FORMATS = {1, 2, 3}
SUPPORTED_LEXICAL_FORMATS = {1, 2}
SUPPORTED_PASSAGE_INDEX_FORMATS = {1, 2}

# §5.2: fixed widths. See AMBIGUITIES note 4 — `stride` is carried in the
# artifact, this one is not.
ID_ENTRY_STRIDE = 36

# §6.1: the technical punctuation set, exactly seven characters.
TECHNICAL_PUNCTUATION = frozenset("_-.:/@#")

# §6.2: BM25 profile.
BM25_K1 = 1.2
BM25_B = 0.75
TECHNICAL_BOOST = 3.0


class Invalid(Exception):
    """Any reason the artifact is not acceptable."""


# ── FORMAT-v3 §6.1: normative tokenization ────────────────────────────────────

def _is_alphanumeric(ch: str) -> bool:
    # \p{L} or \p{N}. Deliberately not str.isalnum(); see AMBIGUITIES note 2.
    return unicodedata.category(ch)[0] in ("L", "N")


def _keepable(ch: str) -> bool:
    return _is_alphanumeric(ch) or ch in TECHNICAL_PUNCTUATION


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens = []
    for raw in normalized.split():
        start, end = 0, len(raw)
        while start < end and not _keepable(raw[start]):
            start += 1
        while end > start and not _keepable(raw[end - 1]):
            end -= 1
        trimmed = raw[start:end]
        if trimmed:
            tokens.append(trimmed)
    return tokens


def _boost(term: str) -> float:
    # §6.2: digits or technical punctuation anywhere in the term.
    for ch in term:
        if ch.isdigit() and ch.isascii():
            return TECHNICAL_BOOST
        if ch in TECHNICAL_PUNCTUATION:
            return TECHNICAL_BOOST
    return 1.0


# ── Varints (§6) ──────────────────────────────────────────────────────────────

def _read_varint(data: bytes, at: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if at >= len(data):
            raise Invalid("varint runs past end of buffer")
        if shift > 63:
            raise Invalid("varint overflows 64 bits")
        byte = data[at]
        at += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, at
        shift += 7


def decode_postings(data: bytes, expected: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    at = 0
    previous = 0
    for index in range(expected):
        delta, at = _read_varint(data, at)
        frequency, at = _read_varint(data, at)
        if frequency == 0:
            raise Invalid("posting has zero term frequency")
        ordinal = delta if index == 0 else previous + delta
        if index > 0 and delta == 0:
            raise Invalid("posting ordinals must strictly increase")
        if ordinal > 0xFFFF_FFFF:
            raise Invalid("posting ordinal out of range")
        out.append((ordinal, frequency))
        previous = ordinal
    if at != len(data):
        raise Invalid("posting list has trailing bytes")
    return out


# ── Structures ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Entry:
    section_id: int
    type: int
    format_version: int
    codec: int
    flags: int
    offset: int
    stored_length: int
    logical_length: int
    item_count: int
    hash: bytes
    raw: bytes

    @property
    def required(self) -> bool:
        return bool(self.flags & 1)

    @property
    def derived(self) -> bool:
        return bool(self.flags & 2)


class Pack:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self._parse_header()
        self._parse_directory()
        self._verify_root()
        self._load_core()

    # ── §1 ────────────────────────────────────────────────────────────────────
    def _parse_header(self) -> None:
        d = self.data
        if len(d) < HEADER_SIZE:
            raise Invalid(f"file too short for header: {len(d)} < {HEADER_SIZE}")
        if d[:8] != MAGIC:
            raise Invalid(f"bad magic bytes: {d[:8]!r}")
        version, header_size = struct.unpack_from("<II", d, 8)
        if version != FORMAT_VERSION:
            raise Invalid(f"unsupported format version: {version}")
        if header_size != HEADER_SIZE:
            raise Invalid(f"unsupported header size: {header_size}")
        (self.flags, self.directory_offset, self.directory_length) = struct.unpack_from("<QQQ", d, 16)
        self.manifest_section_id, self.section_count = struct.unpack_from("<II", d, 40)
        self.root_hash = d[48:80]
        if any(d[80:128]):
            raise Invalid("reserved header bytes nonzero")
        if self.section_count > MAX_SECTIONS:
            raise Invalid(f"too many sections: {self.section_count}")
        if self.directory_length != self.section_count * ENTRY_SIZE:
            raise Invalid("directory length does not match section count")

    # ── §2 ────────────────────────────────────────────────────────────────────
    def _parse_directory(self) -> None:
        end = self.directory_offset + self.directory_length
        if end > len(self.data) or self.directory_offset < HEADER_SIZE:
            raise Invalid("directory range exceeds file")
        self.entries: list[Entry] = []
        previous_id = -1
        for i in range(self.section_count):
            at = self.directory_offset + i * ENTRY_SIZE
            raw = self.data[at:at + ENTRY_SIZE]
            (section_id,) = struct.unpack_from("<I", raw, 0)
            section_type, format_version, codec, flags = struct.unpack_from("<HHHH", raw, 4)
            offset, stored_length, logical_length, item_count = struct.unpack_from("<QQQQ", raw, 12)
            entry = Entry(section_id, section_type, format_version, codec, flags,
                          offset, stored_length, logical_length, item_count,
                          raw[44:76], raw)
            if any(raw[76:80]):
                raise Invalid("reserved directory bytes nonzero")
            if section_id <= previous_id:
                raise Invalid("directory entries must be in increasing section-ID order")
            previous_id = section_id
            if offset + stored_length > len(self.data) or offset < HEADER_SIZE:
                raise Invalid(f"section {section_id} range exceeds file")
            if entry.required and section_type not in KNOWN_TYPES:
                raise Invalid(f"unknown required section type {section_type}")
            if entry.required and codec not in (0, 1):
                raise Invalid(f"unknown required codec {codec}")
            if codec == 0 and stored_length != logical_length:
                raise Invalid(f"section {section_id} has mismatched lengths")
            if entry.derived and entry.required:
                raise Invalid("a derived section must not be required")
            self.entries.append(entry)

        # §2: v3 section types are singletons except Signature.
        seen: set[int] = set()
        for entry in self.entries:
            if entry.type in KNOWN_TYPES and entry.type != T_SIGNATURE:
                if entry.type in seen:
                    raise Invalid(f"duplicate singleton section type {entry.type}")
                seen.add(entry.type)

        for required in CORE_REQUIRED:
            if not any(e.type == required and e.required for e in self.entries):
                raise Invalid(f"core section {required} is missing or optional")

    # ── §3 ────────────────────────────────────────────────────────────────────
    def _verify_root(self) -> None:
        hasher = blake3.blake3(CONTENT_ROOT_CONTEXT)
        for entry in self.entries:
            if entry.type != T_SIGNATURE:
                hasher.update(entry.raw)
        computed = hasher.digest()
        if computed != self.root_hash:
            raise Invalid(
                f"root hash mismatch: computed={computed.hex()} stored={self.root_hash.hex()}")

    def entry_of(self, section_type: int) -> Entry | None:
        for entry in self.entries:
            if entry.type == section_type:
                return entry
        return None

    def require(self, section_type: int) -> Entry:
        entry = self.entry_of(section_type)
        if entry is None:
            raise Invalid(f"section type {section_type} is missing")
        return entry

    def _stored(self, entry: Entry) -> bytes:
        stored = self.data[entry.offset:entry.offset + entry.stored_length]
        if blake3.blake3(stored).digest() != entry.hash:
            raise Invalid(f"section {entry.section_id}: BLAKE3 hash mismatch")
        return stored

    def _inflate(self, stored: bytes, logical_length: int, what: str) -> bytes:
        # §2: bound decompression before allocating.
        if logical_length > DECOMP_RATIO_FLOOR:
            if logical_length > max(len(stored), 1) * DECOMP_RATIO_LIMIT:
                raise Invalid(f"{what} exceeds the decompression ratio limit")
        out = zlib.decompressobj().decompress(stored, logical_length + 1)
        if len(out) != logical_length:
            raise Invalid(f"{what} decompressed to {len(out)}, expected {logical_length}")
        return out

    def section(self, section_type: int) -> bytes:
        entry = self.require(section_type)
        stored = self._stored(entry)
        if entry.codec == 0:
            return stored
        if entry.codec == 1:
            return self._inflate(stored, entry.logical_length, f"section {entry.section_id}")
        raise Invalid(f"unsupported codec {entry.codec}")

    def json_section(self, section_type: int) -> Any:
        return json.loads(self.section(section_type))

    # §5.2 / §6: one block, verified against its own hash before use.
    def block(self, entry: Entry, block: dict) -> bytes:
        offset = entry.offset + int(block["offset"])
        stored_length = int(block["stored_length"])
        if offset + stored_length > len(self.data):
            raise Invalid("index block exceeds file")
        stored = self.data[offset:offset + stored_length]
        if blake3.blake3(stored).hexdigest() != block["hash"]:
            raise Invalid("index block failed verification")
        return self._inflate(stored, int(block["logical_length"]), "index block")

    # ── open: load what every query needs ─────────────────────────────────────
    def _load_core(self) -> None:
        manifest_entry = self.require(T_MANIFEST)
        if manifest_entry.format_version not in SUPPORTED_MANIFEST_FORMATS:
            raise Invalid(f"unsupported manifest section format {manifest_entry.format_version}")
        self.manifest = self.json_section(T_MANIFEST)

        postings_entry = self.require(T_LEX_POSTINGS)
        if postings_entry.format_version not in SUPPORTED_LEXICAL_FORMATS:
            raise Invalid(f"unsupported lexical format {postings_entry.format_version}")
        index_entry = self.require(T_PASSAGE_INDEX)
        if index_entry.format_version not in SUPPORTED_PASSAGE_INDEX_FORMATS:
            raise Invalid(f"unsupported passage index format {index_entry.format_version}")

        self.documents = self.json_section(T_DOCUMENTS)
        self.passage_index = self.json_section(T_PASSAGE_INDEX)
        self.dictionary = self.json_section(T_LEX_DICT)

        self.passage_count = int(self.manifest["passage_count"])
        self.passage_lengths = self.dictionary["passage_lengths"]
        if len(self.passage_lengths) != self.passage_count:
            raise Invalid("lexical index and manifest passage counts disagree")
        self.average_length = max(1.0, float(self.dictionary["average_passage_length"]))

        # §6: lexical layout.
        self.lexical_blocks = self.passage_index.get("lexical_blocks")
        if self.lexical_blocks:
            if self.entry_of(T_LEXICAL_TERMS) is None:
                raise Invalid("format 2 declares block tables but has no lexical terms section")
            self._check_tiling(self.lexical_blocks["dictionary"], self.require(T_LEXICAL_TERMS),
                               "lexical_terms", require_first=True)
            self.postings_starts = self._check_tiling(
                self.lexical_blocks["postings"], postings_entry, "lexical_postings")
            self.postings = b""
        else:
            self.postings = self.section(T_LEX_POSTINGS)
            self.postings_starts = None

        # §5.2: record layout.
        self.record_blocks = self.passage_index.get("record_blocks")
        if self.record_blocks:
            if self.entry_of(T_PASSAGE_RECORDS) is None:
                raise Invalid("format 2 declares record blocks but has no passage records section")
            self._check_record_blocks()
        elif len(self.passage_index.get("records", [])) != self.passage_count:
            raise Invalid("passage index and manifest passage counts disagree")

        self._block_cache: dict[tuple[str, int], bytes] = {}

    def _check_tiling(self, blocks: list[dict], entry: Entry, label: str,
                      require_first: bool = False) -> list[int]:
        """§6: blocks must tile their section exactly. Returns logical starts."""
        starts = []
        stored_cursor = 0
        logical_cursor = 0
        previous_first: str | None = None
        for block in blocks:
            if int(block["offset"]) != stored_cursor:
                raise Invalid(f"{label} blocks are not contiguous")
            stored_length = int(block["stored_length"])
            logical_length = int(block["logical_length"])
            if stored_length == 0 or logical_length == 0:
                raise Invalid(f"{label} block is empty")
            if len(bytes.fromhex(block["hash"])) != 32:
                raise Invalid(f"{label} block has an invalid hash")
            if require_first:
                first = block.get("first_term")
                if not isinstance(first, str):
                    raise Invalid(f"{label} block is missing its first term")
                if previous_first is not None and first <= previous_first:
                    raise Invalid(f"{label} block first terms must strictly increase")
                previous_first = first
            starts.append(logical_cursor)
            stored_cursor += stored_length
            logical_cursor += logical_length
        if stored_cursor != entry.stored_length:
            raise Invalid(f"{label} blocks do not cover their section exactly")
        return starts

    def _check_record_blocks(self) -> None:
        entry = self.require(T_PASSAGE_RECORDS)
        index = self.record_blocks
        stride = int(index["stride"])
        if stride == 0 or int(index["per_block"]) == 0:
            raise Invalid("record block index has a zero stride or block size")
        cursor = 0
        record_bytes = 0
        id_bytes = 0
        previous_first: str | None = None
        for label, blocks in (("record", index["records"]), ("id", index["ids"])):
            for block in blocks:
                if int(block["offset"]) != cursor:
                    raise Invalid(f"{label} blocks are not contiguous")
                stored_length = int(block["stored_length"])
                logical_length = int(block["logical_length"])
                if stored_length == 0 or logical_length == 0:
                    raise Invalid(f"{label} block is empty")
                if len(bytes.fromhex(block["hash"])) != 32:
                    raise Invalid(f"{label} block has an invalid hash")
                if label == "id":
                    first = block.get("first_term")
                    if not isinstance(first, str):
                        raise Invalid("id index block is missing its first id")
                    if previous_first is not None and first <= previous_first:
                        raise Invalid("id index block first ids must strictly increase")
                    previous_first = first
                    id_bytes += logical_length
                else:
                    record_bytes += logical_length
                cursor += stored_length
        if cursor != entry.stored_length:
            raise Invalid("record blocks do not cover their section exactly")
        if record_bytes != self.passage_count * stride:
            raise Invalid("record blocks do not cover every passage")
        if id_bytes != self.passage_count * ID_ENTRY_STRIDE:
            raise Invalid("id index does not cover every passage")

    def _cached_block(self, kind: str, entry: Entry, blocks: list[dict], index: int) -> bytes:
        key = (kind, index)
        if key not in self._block_cache:
            self._block_cache[key] = self.block(entry, blocks[index])
        return self._block_cache[key]

    # ── §6: term lookup ───────────────────────────────────────────────────────
    def lookup(self, term: str) -> dict | None:
        if not self.lexical_blocks:
            return self.dictionary.get("terms", {}).get(term)
        blocks = self.lexical_blocks["dictionary"]
        # The block that can contain a term is the last whose first_term is <= it.
        chosen = None
        for i, block in enumerate(blocks):
            if block["first_term"] <= term:
                chosen = i
            else:
                break
        if chosen is None:
            return None
        payload = self._cached_block("terms", self.require(T_LEXICAL_TERMS), blocks, chosen)
        return json.loads(payload).get("terms", {}).get(term)

    def posting_bytes(self, meta: dict) -> bytes:
        start = int(meta["offset"])
        length = int(meta["length"])
        end = start + length
        if not self.lexical_blocks:
            out = self.postings[start:end]
            if len(out) != length:
                raise Invalid("posting list exceeds its section")
            return out
        entry = self.require(T_LEX_POSTINGS)
        blocks = self.lexical_blocks["postings"]
        parts = []
        for i, block in enumerate(blocks):
            block_start = self.postings_starts[i]
            block_end = block_start + int(block["logical_length"])
            if block_end <= start or block_start >= end:
                continue
            payload = self._cached_block("postings", entry, blocks, i)
            parts.append(payload[max(start - block_start, 0):min(end, block_end) - block_start])
        out = b"".join(parts)
        if len(out) != length:
            raise Invalid("posting list is not covered by the postings block table")
        return out

    # ── §5.2: record access ───────────────────────────────────────────────────
    def record_at(self, ordinal: int) -> dict:
        if not 0 <= ordinal < self.passage_count:
            raise Invalid(f"passage ordinal {ordinal} is out of range")
        if not self.record_blocks:
            return self.passage_index["records"][ordinal]
        stride = int(self.record_blocks["stride"])
        per_block = int(self.record_blocks["per_block"])
        blocks = self.record_blocks["records"]
        payload = self._cached_block("records", self.require(T_PASSAGE_RECORDS),
                                     blocks, ordinal // per_block)
        at = (ordinal % per_block) * stride
        if at + stride > len(payload):
            raise Invalid("passage record exceeds its block")
        block, offset, length = struct.unpack_from("<III", payload, at)
        # §5.2: a record carries no identifier.
        return {"block": block, "offset": offset, "length": length}

    def passage(self, ordinal: int) -> dict:
        record = self.record_at(ordinal)
        blocks = self.passage_index["blocks"]
        block = blocks[record["block"]]
        entry = self.require(T_PASSAGE_DATA)
        payload = self._cached_block("passages", entry, blocks, record["block"])
        start = record["offset"]
        end = start + record["length"]
        if end > len(payload):
            raise Invalid("passage exceeds its logical block")
        passage = json.loads(payload[start:end])
        # §5.2: the payload's own ordinal is what detects a mis-seek.
        if passage.get("ordinal") != ordinal:
            raise Invalid(f"passage payload at ordinal {ordinal} reports a different ordinal")
        return passage

    # ── §6.2: BM25 ────────────────────────────────────────────────────────────
    def search(self, query: str, limit: int = 10) -> list[dict]:
        scores: dict[int, float] = {}
        seen: set[str] = set()
        for term in tokenize(query):
            if term in seen:
                continue
            seen.add(term)
            meta = self.lookup(term)
            if not meta:
                continue
            df = int(meta["document_frequency"])
            if df < 1:
                raise Invalid(f"posting metadata for {term!r} is non-canonical")
            postings = decode_postings(self.posting_bytes(meta), df)
            import math
            idf = math.log(1 + (self.passage_count - df + 0.5) / (df + 0.5)) * _boost(term)
            for ordinal, tf in postings:
                if ordinal >= self.passage_count:
                    raise Invalid("posting ordinal exceeds passage count")
                dl = float(self.passage_lengths[ordinal])
                denominator = tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / self.average_length)
                scores[ordinal] = scores.get(ordinal, 0.0) + idf * tf * (BM25_K1 + 1.0) / denominator
        # §6.2: ties resolve by ascending passage ordinal.
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
        # §5.2: the identifier comes from the payload, not the record.
        return [{"passage_id": self.passage(ordinal)["id"], "score": score}
                for ordinal, score in ranked]


# ── EVIDENCE-v1: standalone receipt verification ─────────────────────────────
#
# Verifies with no artifact and no network: the receipt carries every byte
# needed. The chain is passage record -> Merkle path -> logical content root ->
# manifest -> directory -> artifact root, with an optional signature over the
# artifact root.

def _b64(value: str, label: str) -> bytes:
    import base64
    try:
        return base64.b64decode(value, validate=True)
    except Exception as error:
        raise Invalid(f"{label} is not valid base64: {error}") from error


def verify_receipt(receipt: dict) -> None:
    """Raise Invalid if any link in the chain fails."""
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise Invalid(f"unsupported receipt schema {receipt.get('schema')!r}")

    # 1. The passage record hashes to the claimed passage hash.
    record = _b64(receipt["passage_record_b64"], "passage record")
    leaf = blake3.blake3(PASSAGE_EVIDENCE_CONTEXT + record).digest()
    if leaf.hex() != receipt["passage_hash"]:
        raise Invalid("passage record does not match passage_hash")

    # 2. The Merkle path folds the leaf to the logical content root. Sibling
    #    order is explicit rather than derived from an index, so a proof cannot
    #    be replayed against a different position.
    node = leaf
    for step in receipt["inclusion_proof"]:
        sibling = bytes.fromhex(step["sibling"])
        if len(sibling) != 32:
            raise Invalid("inclusion proof sibling is not 32 bytes")
        pair = (sibling + node) if step["sibling_is_left"] else (node + sibling)
        node = blake3.blake3(NODE_CONTEXT + pair).digest()
    if node.hex() != receipt["passage_merkle_root"]:
        raise Invalid("inclusion proof does not reach passage_merkle_root")

    # 3. The manifest commits to that logical content root.
    manifest_bytes = _b64(receipt["manifest_bytes_b64"], "manifest")
    manifest = json.loads(manifest_bytes)
    if manifest.get("passage_merkle_root") != receipt["passage_merkle_root"]:
        raise Invalid("manifest does not commit the receipt's passage_merkle_root")

    # 4. The manifest bytes match their section-directory entry.
    directory = _b64(receipt["directory_b64"], "directory")
    if len(directory) % ENTRY_SIZE != 0:
        raise Invalid("directory is not a whole number of entries")
    entries = [directory[i:i + ENTRY_SIZE] for i in range(0, len(directory), ENTRY_SIZE)]

    def entry_for(section_id: int) -> bytes:
        for raw in entries:
            if struct.unpack_from("<I", raw, 0)[0] == section_id:
                return raw
        raise Invalid(f"directory has no entry for section {section_id}")

    manifest_entry = entry_for(receipt["manifest_section_id"])
    if struct.unpack_from("<H", manifest_entry, 4)[0] != T_MANIFEST:
        raise Invalid("manifest_section_id does not reference a Manifest section")
    if blake3.blake3(manifest_bytes).digest() != manifest_entry[44:76]:
        raise Invalid("manifest bytes do not match their directory entry hash")

    # 5. The directory reproduces the artifact root, excluding signature
    #    entries exactly as the writer does.
    hasher = blake3.blake3(CONTENT_ROOT_CONTEXT)
    for raw in entries:
        if struct.unpack_from("<H", raw, 4)[0] != T_SIGNATURE:
            hasher.update(raw)
    if hasher.digest().hex() != receipt["pack_root"]:
        raise Invalid("directory does not reproduce pack_root")

    # 6. The receipt's claims about the passage match the authenticated record.
    passage = json.loads(record)
    if passage.get("id") != receipt["passage_id"]:
        raise Invalid("passage_id does not match the authenticated record")
    if "passage_ordinal" in receipt and passage.get("ordinal") != receipt["passage_ordinal"]:
        raise Invalid("passage_ordinal does not match the authenticated record")

    # 7. Pack coordinate and source revision come from the authenticated
    #    manifest, not from unauthenticated receipt fields.
    expected_pack = f"{manifest.get('name')}@{manifest.get('version')}"
    if receipt.get("pack") != expected_pack:
        raise Invalid("pack coordinate does not match the authenticated manifest")
    if receipt.get("source_revision") != manifest.get("source_revision"):
        raise Invalid("source_revision does not match the authenticated manifest")

    # 8. A canonical URL claim must be derivable from the authenticated
    #    Documents section, so a receipt cannot assert an arbitrary URL.
    if receipt.get("canonical_url") is not None:
        documents_b64 = receipt.get("documents_bytes_b64")
        section_id = receipt.get("documents_section_id")
        if documents_b64 is None or section_id is None:
            raise Invalid("canonical_url asserted without the Documents section")
        documents_bytes = _b64(documents_b64, "documents section")
        documents_entry = entry_for(section_id)
        if struct.unpack_from("<H", documents_entry, 4)[0] != T_DOCUMENTS:
            raise Invalid("documents_section_id does not reference a Documents section")
        if blake3.blake3(documents_bytes).digest() != documents_entry[44:76]:
            raise Invalid("documents bytes do not match their directory entry hash")
        codec = struct.unpack_from("<H", documents_entry, 8)[0]
        logical_length = struct.unpack_from("<Q", documents_entry, 28)[0]
        logical = documents_bytes if codec == 0 else zlib.decompressobj().decompress(
            documents_bytes, logical_length + 1)
        documents = json.loads(logical)
        document = next(
            (d for d in documents if d.get("id") == passage.get("document_id")), None)
        if document is None:
            raise Invalid("Documents section has no entry for the passage's document")
        base = document.get("url")
        anchor = passage.get("anchor")
        expected = base
        if base and anchor and "#" not in base:
            expected = f"{base}#{anchor}"
        if expected != receipt["canonical_url"]:
            raise Invalid("canonical_url is not derivable from authenticated bytes")

    # 9. A signature, when present, is over the artifact root under a
    #    domain-separated context. Its validity is a separate claim from
    #    integrity, and publisher identity is separate again: a valid signature
    #    establishes neither.
    signature = receipt.get("signature")
    if signature is not None:
        if signature.get("algorithm") != "ed25519":
            raise Invalid(f"unsupported signature algorithm {signature.get('algorithm')!r}")
        try:
            from cryptography.exceptions import InvalidSignature
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        except ImportError as error:
            raise Invalid(f"receipt is signed but Ed25519 support is unavailable: {error}") from error
        public_key = bytes.fromhex(signature["public_key"])
        if blake3.blake3(public_key).hexdigest() != signature.get("key_id"):
            raise Invalid("signature key_id does not match its public key")
        message = SIGNATURE_CONTEXT + bytes.fromhex(receipt["pack_root"])
        try:
            Ed25519PublicKey.from_public_bytes(public_key).verify(
                bytes.fromhex(signature["signature"]), message)
        except InvalidSignature as error:
            raise Invalid("signature is not valid over the artifact root") from error


def verify_sections(pack: Pack) -> None:
    """Every section's stored bytes must match its directory hash."""
    for entry in pack.entries:
        pack._stored(entry)


# ── Adapter contract (spec/conformance/README.md) ─────────────────────────────

def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: annpack_reader.py <tokenize|search|open|verify-receipt> ...", file=sys.stderr)
        return 2
    verb = argv[1]
    try:
        if verb == "tokenize":
            print(json.dumps(tokenize(argv[2])))
            return 0
        if verb == "open":
            pack = Pack(Path(argv[2]).read_bytes())
            verify_sections(pack)
            return 0
        if verb == "search":
            pack = Pack(Path(argv[2]).read_bytes())
            print(json.dumps({"results": pack.search(argv[3], limit=10)}))
            return 0
        if verb == "verify-receipt":
            verify_receipt(json.loads(Path(argv[2]).read_text()))
            return 0
    except Invalid as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except (KeyError, IndexError, ValueError, struct.error, zlib.error) as error:
        # A malformed artifact must be an error, never a crash.
        print(f"error: malformed artifact: {error}", file=sys.stderr)
        return 1
    print(f"unknown verb: {verb}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
