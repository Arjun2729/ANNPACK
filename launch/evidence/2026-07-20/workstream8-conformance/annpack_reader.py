#!/usr/bin/env python3
"""
ANNPack Core v1.0-draft — Independent Python Reader
Written from spec only (CORE-v1.0-draft.md, FORMAT-v3.md, SECURITY.md, MEDIA-TYPES.md, PROTOCOL-v1.md).
No Rust source was read or referenced.
"""

import json
import math
import os
import struct
import zlib
from dataclasses import dataclass, field
from typing import Optional

import blake3
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature

# ── Format constants ──────────────────────────────────────────────────────────

MAGIC = b"ANNPACK3"
FORMAT_VERSION = 3
HEADER_SIZE = 128
SECTION_ENTRY_SIZE = 80

MAX_SECTIONS = 16_384
MAX_MANIFEST_BYTES = 4 * 1024 * 1024       # 4 MiB
MAX_SECTION_BYTES = 64 * 1024 * 1024 * 1024  # 64 GiB
MAX_PASSAGE_BLOCK_LOGICAL = 1 * 1024 * 1024  # 1 MiB per spec §9
DECOMP_RATIO = 256
DECOMP_RATIO_FLOOR = 16 * 1024 * 1024       # 16 MiB
MAX_VARINT_BYTES = 10

# Section types
TYPE_MANIFEST         = 1
TYPE_DOCUMENTS        = 2
TYPE_PASSAGE_INDEX    = 3
TYPE_PASSAGE_DATA     = 4
TYPE_LEXICAL_DICT     = 5
TYPE_LEXICAL_POSTINGS = 6
TYPE_SIGNATURE        = 10

KNOWN_SECTION_TYPES = set(range(1, 13))
CORE_REQUIRED_TYPES = {
    TYPE_MANIFEST, TYPE_DOCUMENTS, TYPE_PASSAGE_INDEX,
    TYPE_PASSAGE_DATA, TYPE_LEXICAL_DICT, TYPE_LEXICAL_POSTINGS,
}

# Codecs
CODEC_UNCOMPRESSED = 0
CODEC_ZLIB         = 1

FLAG_REQUIRED = 0x0001

# BM25
BM25_K1 = 1.2
BM25_B  = 0.75
TECH_BOOST = 2.0  # for terms with digits or technical punctuation


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class SectionEntry:
    section_id:      int
    section_type:    int
    fmt_version:     int
    codec:           int
    flags:           int
    stored_offset:   int
    stored_length:   int
    logical_length:  int
    item_count:      int
    stored_hash:     bytes   # 32 bytes
    raw_bytes:       bytes   # full 80-byte entry for root computation


@dataclass
class PassageRecord:
    passage_id:    str
    block_ordinal: int
    offset:        int
    length:        int


@dataclass
class BlockEntry:
    offset:         int   # relative to passage-data section stored bytes
    stored_length:  int
    logical_length: int
    hash_hex:       str


@dataclass
class SearchResult:
    passage_id:    str
    score:         float
    ordinal:       int
    record:        dict
    record_bytes:  bytes


# ── Errors ────────────────────────────────────────────────────────────────────

class AnnPackError(Exception):
    pass


# ── Varint decoder ────────────────────────────────────────────────────────────

def decode_varint(data: bytes, pos: int) -> tuple[int, int]:
    """LEB128 unsigned varint; terminates after MAX_VARINT_BYTES."""
    value = 0
    shift = 0
    for _ in range(MAX_VARINT_BYTES):
        if pos >= len(data):
            raise AnnPackError("varint: unexpected end of data")
        b = data[pos]; pos += 1
        value |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return value, pos
        shift += 7
    raise AnnPackError("varint: not terminated within 10 bytes")


# ── Main reader ───────────────────────────────────────────────────────────────

class AnnPackReader:
    """
    Read-only Core reader.  Security invariants from SECURITY.md are implemented
    throughout; see inline comments.
    """

    def __init__(self, path: str):
        with open(path, "rb") as fh:
            self._data = fh.read()
        self._size = len(self._data)

        # Populated by _parse()
        self.header_root:    bytes = b""
        self.computed_root:  bytes = b""
        self.sections:       list[SectionEntry] = []
        self._by_type:       dict[int, list[SectionEntry]] = {}

        self.manifest:    dict = {}
        self.documents:   list = []
        self._doc_by_id:  dict[str, dict] = {}

        self._passage_records: list[PassageRecord] = []
        self._block_table:     list[BlockEntry] = []
        self._pd_sec:          Optional[SectionEntry] = None
        self._pd_raw:          bytes = b""   # stored bytes of passage-data section

        self._lex_dict:          dict = {}
        self._postings_data:     bytes = b""

        self.signatures: list[dict] = []

        self._parse()

    # ── Low-level I/O ────────────────────────────────────────────────────────

    def _read(self, offset: int, length: int) -> bytes:
        """Bounds-checked read.  SECURITY: rejects out-of-bounds or overflow."""
        if length == 0:
            return b""
        if offset < 0 or length < 0:
            raise AnnPackError("negative offset or length")
        end = offset + length
        if end < offset:   # addition overflow (Python ints don't overflow, but check semantics)
            raise AnnPackError("offset+length overflow")
        if end > self._size:
            raise AnnPackError(
                f"read [{offset},{end}) exceeds file size {self._size}"
            )
        return self._data[offset:end]

    # ── Parse pipeline ───────────────────────────────────────────────────────

    def _parse(self):
        self._parse_header()
        self._parse_directory()
        # SECURITY invariant 9: verify root BEFORE interpreting any section
        self._verify_root()
        self._load_sections()

    def _parse_header(self):
        # SECURITY: reject if too short for header
        if self._size < HEADER_SIZE:
            raise AnnPackError(
                f"file too short for header: {self._size} < {HEADER_SIZE}"
            )

        hdr = self._data[:HEADER_SIZE]

        # Magic
        if hdr[0:8] != MAGIC:
            raise AnnPackError(f"bad magic bytes: {hdr[0:8]!r}")

        # Format version
        version = struct.unpack_from("<I", hdr, 8)[0]
        if version != FORMAT_VERSION:
            raise AnnPackError(f"unsupported format version: {version}")

        # Header size field must equal 128
        hdr_size = struct.unpack_from("<I", hdr, 12)[0]
        if hdr_size != HEADER_SIZE:
            raise AnnPackError(f"unexpected header_size field: {hdr_size}")

        self._container_flags = struct.unpack_from("<Q", hdr, 16)[0]

        self._dir_offset = struct.unpack_from("<Q", hdr, 24)[0]
        self._dir_length = struct.unpack_from("<Q", hdr, 32)[0]

        self._manifest_section_id = struct.unpack_from("<I", hdr, 40)[0]
        self._section_count       = struct.unpack_from("<I", hdr, 44)[0]

        # SECURITY: section count limit
        if self._section_count > MAX_SECTIONS:
            raise AnnPackError(
                f"section count {self._section_count} exceeds limit {MAX_SECTIONS}"
            )

        # directory length MUST equal section_count × 80
        expected = self._section_count * SECTION_ENTRY_SIZE
        if self._dir_length != expected:
            raise AnnPackError(
                f"directory length {self._dir_length} != section_count*80 = {expected}"
            )

        # Content root
        self.header_root = bytes(hdr[48:80])

        # SECURITY: reserved bytes MUST be zero
        if any(b != 0 for b in hdr[80:128]):
            raise AnnPackError("reserved header bytes nonzero")

        # SECURITY: directory range must fit in file (overflow-checked)
        dir_end = self._dir_offset + self._dir_length
        if dir_end < self._dir_offset or dir_end > self._size:
            raise AnnPackError("directory range exceeds file")

    def _parse_directory(self):
        dir_data = self._read(self._dir_offset, self._dir_length)

        prev_id  = -1
        seen_ids:    set[int]          = set()
        seen_ranges: list[tuple[int,int]] = []

        for i in range(self._section_count):
            e_start = i * SECTION_ENTRY_SIZE
            entry   = dir_data[e_start : e_start + SECTION_ENTRY_SIZE]

            section_id   = struct.unpack_from("<I", entry,  0)[0]
            section_type = struct.unpack_from("<H", entry,  4)[0]
            fmt_version  = struct.unpack_from("<H", entry,  6)[0]
            codec        = struct.unpack_from("<H", entry,  8)[0]
            flags        = struct.unpack_from("<H", entry, 10)[0]
            stored_off   = struct.unpack_from("<Q", entry, 12)[0]
            stored_len   = struct.unpack_from("<Q", entry, 20)[0]
            logical_len  = struct.unpack_from("<Q", entry, 28)[0]
            item_count   = struct.unpack_from("<Q", entry, 36)[0]
            stored_hash  = bytes(entry[44:76])
            reserved     = struct.unpack_from("<I", entry, 76)[0]

            # SECURITY: reserved directory bytes MUST be zero
            if reserved != 0:
                raise AnnPackError(
                    f"section {section_id}: reserved directory bytes nonzero"
                )

            # SECURITY: strictly increasing IDs (catches noncanonical order and duplicates)
            if section_id in seen_ids:
                raise AnnPackError(f"duplicate section ID {section_id}")
            if section_id <= prev_id:
                raise AnnPackError(
                    f"section IDs not strictly increasing: {prev_id} -> {section_id}"
                )
            prev_id = section_id
            seen_ids.add(section_id)

            # SECURITY: per-section size limit
            if stored_len > MAX_SECTION_BYTES:
                raise AnnPackError(
                    f"section {section_id}: stored_length {stored_len} exceeds 64 GiB"
                )

            # SECURITY: section must fit in file (overflow-checked)
            if stored_len > 0:
                sec_end = stored_off + stored_len
                if sec_end < stored_off or sec_end > self._size:
                    raise AnnPackError(
                        f"section {section_id}: stored range exceeds file"
                    )

                # SECURITY: no overlapping sections
                for (rs, re) in seen_ranges:
                    if stored_off < re and sec_end > rs:
                        raise AnnPackError(
                            f"section {section_id}: overlaps another section"
                        )
                seen_ranges.append((stored_off, sec_end))

            # SECURITY: validate codec
            is_required = bool(flags & FLAG_REQUIRED)
            if codec not in (CODEC_UNCOMPRESSED, CODEC_ZLIB):
                if is_required:
                    raise AnnPackError(
                        f"section {section_id}: unknown required codec {codec}"
                    )

            # Uncompressed sections: stored == logical
            if codec == CODEC_UNCOMPRESSED and stored_len != logical_len:
                raise AnnPackError(
                    f"section {section_id}: codec=0 but stored_length != logical_length"
                )

            # SECURITY: decompression ratio check (pre-allocation)
            if codec == CODEC_ZLIB and stored_len > 0:
                ceiling = max(stored_len * DECOMP_RATIO, DECOMP_RATIO_FLOOR)
                if logical_len > ceiling:
                    raise AnnPackError(
                        f"section {section_id}: logical_length exceeds 256× ratio limit"
                    )

            # SECURITY: unknown required section types are rejected
            if section_type not in KNOWN_SECTION_TYPES and is_required:
                raise AnnPackError(
                    f"unknown required section type {section_type}"
                )

            sec = SectionEntry(
                section_id=section_id, section_type=section_type,
                fmt_version=fmt_version, codec=codec, flags=flags,
                stored_offset=stored_off, stored_length=stored_len,
                logical_length=logical_len, item_count=item_count,
                stored_hash=stored_hash, raw_bytes=bytes(entry),
            )
            self.sections.append(sec)
            self._by_type.setdefault(section_type, []).append(sec)

    def _verify_root(self):
        """
        Compute BLAKE3("ANNPACK3-CONTENT-ROOT\\0" || <non-sig entries>)
        and assert it matches the stored header root.
        SECURITY invariant 9.
        """
        h = blake3.blake3()
        h.update(b"ANNPACK3-CONTENT-ROOT\x00")
        for sec in self.sections:
            if sec.section_type != TYPE_SIGNATURE:
                h.update(sec.raw_bytes)
        self.computed_root = h.digest()

        if self.computed_root != self.header_root:
            raise AnnPackError(
                f"root hash mismatch: computed={self.computed_root.hex()}"
                f" stored={self.header_root.hex()}"
            )

    # ── Section data ─────────────────────────────────────────────────────────

    def _section_data(self, sec: SectionEntry) -> bytes:
        """
        Read stored bytes, verify BLAKE3 hash, then decompress.
        SECURITY invariant 10: hash verified BEFORE decoding.
        """
        raw = self._read(sec.stored_offset, sec.stored_length)

        # SECURITY: verify stored-byte hash before any decoding
        h = blake3.blake3(raw).digest()
        if h != sec.stored_hash:
            raise AnnPackError(
                f"section {sec.section_id}: BLAKE3 hash mismatch"
            )

        if sec.codec == CODEC_UNCOMPRESSED:
            return raw

        if sec.codec == CODEC_ZLIB:
            # SECURITY: manifest size limit checked by caller; ratio already pre-checked
            decompressed = zlib.decompress(raw)
            if len(decompressed) != sec.logical_length:
                raise AnnPackError(
                    f"section {sec.section_id}: decompressed size {len(decompressed)}"
                    f" != logical_length {sec.logical_length}"
                )
            return decompressed

        raise AnnPackError(f"section {sec.section_id}: unknown codec {sec.codec}")

    def _require_section(self, stype: int, name: str) -> SectionEntry:
        lst = self._by_type.get(stype)
        if not lst:
            raise AnnPackError(f"missing required {name} section (type {stype})")
        return lst[0]

    # ── Section loading ───────────────────────────────────────────────────────

    def _load_sections(self):
        # ── Manifest ─────────────────────────────────────────────────────────
        msec = self._require_section(TYPE_MANIFEST, "Manifest")
        # SECURITY: manifest size limit
        if msec.logical_length > MAX_MANIFEST_BYTES:
            raise AnnPackError(
                f"manifest too large: {msec.logical_length} > 4 MiB"
            )
        self.manifest = json.loads(self._section_data(msec).decode("utf-8"))

        # ── Documents ────────────────────────────────────────────────────────
        dsec = self._require_section(TYPE_DOCUMENTS, "Documents")
        self.documents = json.loads(self._section_data(dsec).decode("utf-8"))
        self._doc_by_id = {d["id"]: d for d in self.documents}

        # ── Passage Index ────────────────────────────────────────────────────
        pisec = self._require_section(TYPE_PASSAGE_INDEX, "Passage Index")
        pi = json.loads(self._section_data(pisec).decode("utf-8"))

        self._passage_records = [
            PassageRecord(
                passage_id=r["id"],
                block_ordinal=r["block"],
                offset=r["offset"],
                length=r["length"],
            )
            for r in pi.get("records", [])
        ]
        self._block_table = [
            BlockEntry(
                offset=b["offset"],
                stored_length=b["stored_length"],
                logical_length=b["logical_length"],
                hash_hex=b["hash"],
            )
            for b in pi.get("blocks", [])
        ]

        # ── Passage Data (store raw bytes for lazy block reads) ───────────────
        self._pd_sec = self._require_section(TYPE_PASSAGE_DATA, "Passage Data")
        praw = self._read(self._pd_sec.stored_offset, self._pd_sec.stored_length)
        # SECURITY: verify section hash before any block extraction
        h = blake3.blake3(praw).digest()
        if h != self._pd_sec.stored_hash:
            raise AnnPackError("Passage Data section: BLAKE3 hash mismatch")
        self._pd_raw = praw

        # ── Lexical Dictionary ────────────────────────────────────────────────
        ldsec = self._require_section(TYPE_LEXICAL_DICT, "Lexical Dictionary")
        self._lex_dict = json.loads(self._section_data(ldsec).decode("utf-8"))

        # ── Lexical Postings ─────────────────────────────────────────────────
        lpsec = self._require_section(TYPE_LEXICAL_POSTINGS, "Lexical Postings")
        self._postings_data = self._section_data(lpsec)

        # ── Signatures (optional) ─────────────────────────────────────────────
        for ssec in self._by_type.get(TYPE_SIGNATURE, []):
            self.signatures.append(
                json.loads(self._section_data(ssec).decode("utf-8"))
            )

    # ── Passage block access ──────────────────────────────────────────────────

    def _get_block(self, blk: BlockEntry) -> bytes:
        """
        Slice a compressed block from passage-data raw bytes, verify its hash,
        decompress it, and check the decompressed length.
        SECURITY: hash verified before decode; ratio pre-checked.
        """
        start = blk.offset
        end   = start + blk.stored_length

        # SECURITY: overflow + bounds check
        if end < start or end > len(self._pd_raw):
            raise AnnPackError("block range exceeds passage-data section")

        raw = self._pd_raw[start:end]

        # SECURITY: verify block hash before decompressing
        h = blake3.blake3(raw).hexdigest()
        if h != blk.hash_hex:
            raise AnnPackError("passage block hash mismatch")

        # SECURITY: passage block size limit (1 MiB logical)
        if blk.logical_length > MAX_PASSAGE_BLOCK_LOGICAL:
            raise AnnPackError(
                f"passage block logical length {blk.logical_length} exceeds 1 MiB"
            )

        decompressed = zlib.decompress(raw)
        if len(decompressed) != blk.logical_length:
            raise AnnPackError(
                f"block decompressed size {len(decompressed)} != {blk.logical_length}"
            )
        return decompressed

    def _fetch_passage(self, pr: PassageRecord) -> tuple[dict, bytes]:
        """Return (parsed record, raw JSON bytes) for a passage record."""
        blk = self._block_table[pr.block_ordinal]
        block_data = self._get_block(blk)

        end = pr.offset + pr.length
        if end < pr.offset or end > len(block_data):
            raise AnnPackError(
                f"passage {pr.passage_id}: byte range exceeds block"
            )
        rec_bytes = block_data[pr.offset:end]
        return json.loads(rec_bytes.decode("utf-8")), rec_bytes

    # ── Tokenization ──────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Simple tokenizer: lower-case, split on whitespace/punctuation while
        preserving alphanumeric sequences that include hyphens or dots
        (e.g. 'ap-104').
        """
        import re
        return re.findall(r"[a-z0-9]+(?:[-\.][a-z0-9]+)*", text.lower())

    @staticmethod
    def _is_technical(token: str) -> bool:
        """True if token contains a digit or technical punctuation (→ boost)."""
        return any(c.isdigit() or c in "-._" for c in token)

    # ── Postings decoder ──────────────────────────────────────────────────────

    def _decode_postings(self, term_info: dict) -> list[tuple[int, int]]:
        """
        Decode delta-encoded (ordinal_delta, tf) varint pairs.
        First ordinal is stored directly; subsequent are positive deltas.
        SECURITY: rejects zero-frequency, trailing bytes, out-of-range ordinals.
        """
        offset = term_info["offset"]
        length = term_info["length"]
        pdata  = self._postings_data[offset : offset + length]

        postings: list[tuple[int, int]] = []
        pos    = 0
        ordinal = 0
        first  = True
        n      = len(self._passage_records)

        while pos < len(pdata):
            delta, pos = decode_varint(pdata, pos)
            tf,    pos = decode_varint(pdata, pos)

            # SECURITY: zero-frequency posting is invalid
            if tf == 0:
                raise AnnPackError("zero-frequency posting in postings list")

            if first:
                ordinal = delta
                first = False
            else:
                # SECURITY: delta must be positive (strictly increasing ordinals)
                if delta == 0:
                    raise AnnPackError("non-positive delta in postings list")
                ordinal += delta

            # SECURITY: ordinal must be in range
            if ordinal >= n:
                raise AnnPackError(
                    f"posting ordinal {ordinal} out of range (n={n})"
                )

            postings.append((ordinal, tf))

        # SECURITY: no trailing bytes allowed
        if pos != len(pdata):
            raise AnnPackError("trailing bytes in postings data")

        return postings

    # ── BM25 search ───────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """BM25 lexical search (k1=1.2, b=0.75) with technical-token boost."""
        terms    = self._lex_dict.get("terms", {})
        lengths  = self._lex_dict.get("passage_lengths", [])
        n        = len(self._passage_records)

        if n == 0 or not lengths:
            return []

        avg_dl = sum(lengths) / len(lengths)

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores: dict[int, float] = {}

        for token in set(query_tokens):
            if token not in terms:
                continue

            term_info = terms[token]
            df        = term_info["document_frequency"]
            if df == 0:
                continue

            idf   = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            boost = TECH_BOOST if self._is_technical(token) else 1.0

            for ordinal, tf in self._decode_postings(term_info):
                dl = lengths[ordinal] if ordinal < len(lengths) else avg_dl
                tf_norm = (tf * (BM25_K1 + 1.0)) / (
                    tf + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / avg_dl)
                )
                scores[ordinal] = scores.get(ordinal, 0.0) + boost * idf * tf_norm

        # Sort: descending score, then ascending ordinal for deterministic ties
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))

        results: list[SearchResult] = []
        for ordinal, score in ranked[:top_k]:
            pr = self._passage_records[ordinal]
            record, rec_bytes = self._fetch_passage(pr)
            results.append(SearchResult(
                passage_id=pr.passage_id,
                score=score,
                ordinal=ordinal,
                record=record,
                record_bytes=rec_bytes,
            ))
        return results

    # ── Evidence envelope ─────────────────────────────────────────────────────

    def _passage_hash(self, rec_bytes: bytes) -> str:
        """
        BLAKE3("ANNPACK3-PASSAGE-EVIDENCE\\0" || deterministic_passage_json)
        rec_bytes are the exact stored bytes from the passage block.
        """
        return blake3.blake3(
            b"ANNPACK3-PASSAGE-EVIDENCE\x00" + rec_bytes
        ).hexdigest()

    def _canonical_url(self, record: dict) -> str:
        doc = self._doc_by_id.get(record.get("document_id", ""), {})
        base = doc.get("url", "")
        anchor = record.get("anchor", "")
        return f"{base}#{anchor}" if anchor else base

    def evidence_envelope(self, result: SearchResult) -> dict:
        """Build the evidence envelope for a search result."""
        pack_name    = self.manifest.get("name", "")
        pack_version = self.manifest.get("version", "")
        source_rev   = self.manifest.get("source_revision", "")

        # Determine publisher status
        pub_status   = "unsigned"
        key_ids:     list[str] = []
        identities:  list[str] = []

        if self.signatures:
            sig = self.signatures[0]
            try:
                raw_pub = bytes.fromhex(sig["public_key"])
                pub_key = Ed25519PublicKey.from_public_bytes(raw_pub)
                sig_bytes = bytes.fromhex(sig["signature"])
                msg = b"ANNPACK3-SIGNATURE\x00" + self.header_root
                pub_key.verify(sig_bytes, msg)
                pub_status = "cryptographically_verified"
                key_ids    = [blake3.blake3(raw_pub).hexdigest()]
                if sig.get("identity"):
                    identities = [sig["identity"]]
            except (InvalidSignature, Exception):
                pub_status = "not_verified"

        return {
            "schema":          "annpack-evidence-v1",
            "pack":            f"{pack_name}@{pack_version}",
            "pack_root":       self.header_root.hex(),
            "source_revision": source_rev,
            "passage_id":      result.passage_id,
            "passage_hash":    self._passage_hash(result.record_bytes),
            "canonical_url":   self._canonical_url(result.record),
            "publisher": {
                "status":              pub_status,
                "key_ids":             key_ids,
                "asserted_identities": identities,
                "identity_trusted":    False,
            },
        }

    # ── Signature verification ────────────────────────────────────────────────

    def verify_signature(self, pub_key_path: Optional[str] = None) -> dict:
        """
        Verify the first Ed25519 signature section (if present).
        Returns a status dict.  identity_trusted is always False
        (external key binding is the caller's responsibility).
        """
        if not self.signatures:
            return {"status": "unsigned"}

        sig = self.signatures[0]
        try:
            raw_pub = bytes.fromhex(sig["public_key"])

            # If a reference public key file is supplied, verify it matches
            if pub_key_path:
                with open(pub_key_path, "rb") as fh:
                    expected_hex = fh.read().decode("ascii").strip()
                expected_raw = bytes.fromhex(expected_hex)
                if raw_pub != expected_raw:
                    return {"status": "not_verified", "reason": "public key mismatch"}

            pub_key = Ed25519PublicKey.from_public_bytes(raw_pub)

            # signed_root must match our computed root
            signed_root = sig.get("signed_root", "")
            if signed_root != self.header_root.hex():
                return {
                    "status": "not_verified",
                    "reason": f"signed_root {signed_root!r} != content root",
                }

            sig_bytes = bytes.fromhex(sig["signature"])
            msg = b"ANNPACK3-SIGNATURE\x00" + self.header_root
            pub_key.verify(sig_bytes, msg)

            key_id = blake3.blake3(raw_pub).hexdigest()
            return {
                "status":              "cryptographically_verified",
                "key_id":              key_id,
                "signed_root":         signed_root,
                "asserted_identities": [sig["identity"]] if sig.get("identity") else [],
                "identity_trusted":    False,
            }

        except InvalidSignature:
            return {"status": "not_verified", "reason": "invalid signature"}
        except AnnPackError as exc:
            return {"status": "not_verified", "reason": str(exc)}
        except Exception as exc:
            return {"status": "not_verified", "reason": str(exc)}


# ── Conformance harness ───────────────────────────────────────────────────────

def try_open(path: str) -> str:
    """Return '' on success or 'error: <reason>' on rejection."""
    try:
        AnnPackReader(path)
        return ""
    except AnnPackError as exc:
        return f"error: {exc}"
    except Exception as exc:
        return f"error: {exc}"


def count_lines(path: str) -> int:
    with open(path, encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def run_conformance(base_dir: str, pub_key_path: str) -> dict:
    """
    Run the full conformance suite and return the report dict.
    base_dir should contain:
        golden-v1.annpack
        golden-v1-signed.annpack
        test.pub
        invalid-corpus/*.annpack
    """
    golden_path  = os.path.join(base_dir, "golden-v1.annpack")
    signed_path  = os.path.join(base_dir, "golden-v1-signed.annpack")
    invalid_dir  = os.path.join(base_dir, "invalid-corpus")

    # ── Golden pack ───────────────────────────────────────────────────────────
    reader = AnnPackReader(golden_path)

    # AP-104 search
    results_ap104 = reader.search("AP-104")
    ap104_ok = False
    ap104_pid = ""
    if results_ap104:
        top = results_ap104[0]
        ap104_pid = top.passage_id
        ap104_ok = (ap104_pid == "073b6867886b39c069a287c9ea426dbada5275b76948257b201836e0878f7c2e")

    # cache rotation search
    results_cache = reader.search("cache rotation")
    cache_ok = False
    if results_cache:
        top = results_cache[0]
        doc = reader._doc_by_id.get(top.record.get("document_id", ""), {})
        cache_ok = doc.get("source_path") == "rotation.md"

    # Evidence envelope for AP-104
    ap104_envelope = reader.evidence_envelope(results_ap104[0]) if results_ap104 else {}

    # ── Signed pack ───────────────────────────────────────────────────────────
    signed_reader = AnnPackReader(signed_path)
    sig_result    = signed_reader.verify_signature(pub_key_path)
    signed_root_unchanged = (signed_reader.header_root.hex() == reader.header_root.hex())

    # ── Invalid corpus ────────────────────────────────────────────────────────
    invalid_files = [
        "empty.annpack",
        "magic-only.annpack",
        "wrong-magic.annpack",
        "wrong-version.annpack",
        "truncated-at-header.annpack",
        "directory-bit-flip.annpack",
        "section-hash-mismatch.annpack",
        "reserved-header-set.annpack",
    ]

    invalid_results: dict[str, str] = {}
    for fname in invalid_files:
        fpath = os.path.join(invalid_dir, fname)
        res = try_open(fpath)
        invalid_results[fname] = res if res else "UNEXPECTEDLY ACCEPTED"

    all_invalid_rejected = all(
        v.startswith("error:") for v in invalid_results.values()
    )

    self_loc = os.path.abspath(__file__)
    loc = count_lines(self_loc)

    return {
        "implementation":                      "python/annpack-reader-independent",
        "golden_root_computed":                reader.computed_root.hex(),
        "golden_root_matches":                 reader.computed_root == reader.header_root,
        "golden_search_ap104_first_passage_id": ap104_pid,
        "golden_search_ap104_correct":          ap104_ok,
        "golden_search_cache_rotation_from_rotation_md": cache_ok,
        "ap104_evidence_envelope":              ap104_envelope,
        "signed_root_unchanged":               signed_root_unchanged,
        "signed_signature_result":             sig_result,
        "invalid_corpus_results":              invalid_results,
        "all_invalid_rejected":                all_invalid_rejected,
        "lines_of_code":                       loc,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    base_dir = os.path.dirname(os.path.abspath(__file__))
    pub_key  = os.path.join(base_dir, "test.pub")

    report = run_conformance(base_dir, pub_key)
    print(json.dumps(report, indent=2))

    # Summary
    print("\n── Summary ──────────────────────────────────────────────────────────", file=sys.stderr)
    print(f"Root match:          {report['golden_root_matches']}", file=sys.stderr)
    print(f"AP-104 correct:      {report['golden_search_ap104_correct']}", file=sys.stderr)
    print(f"Cache rotation ok:   {report['golden_search_cache_rotation_from_rotation_md']}", file=sys.stderr)
    print(f"Signed root same:    {report['signed_root_unchanged']}", file=sys.stderr)
    print(f"All invalid rej.:    {report['all_invalid_rejected']}", file=sys.stderr)
    print(f"LoC:                 {report['lines_of_code']}", file=sys.stderr)
