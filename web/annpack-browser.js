const HEADER_SIZE = 128;
const DIRECTORY_ENTRY_SIZE = 80;
const MAX_SECTIONS = 16384;
const MAX_MANIFEST_SIZE = 4 * 1024 * 1024;
// Manifest schema versions this reader understands. See FORMAT-v3 §4.2.
const SUPPORTED_MANIFEST_FORMAT_VERSIONS = Object.freeze([1, 2, 3, 4]);
const MAX_LOGICAL_SECTION_SIZE = 64 * 1024 * 1024 * 1024;
const MAX_PASSAGE_BLOCK_SIZE = 1024 * 1024;
const DECOMPRESSION_RATIO_LIMIT = 256;
const MAGIC = new TextEncoder().encode('ANNPACK3');
const ROOT_CONTEXT = new TextEncoder().encode('ANNPACK3-CONTENT-ROOT\0');
const EVIDENCE_CONTEXT = new TextEncoder().encode('ANNPACK3-PASSAGE-EVIDENCE\0');
const CORE_CAPABILITIES = Object.freeze([
  'citations',
  'content',
  'lexical-bm25',
  'range-addressable-passages',
  'section-integrity',
]);
const SECTION = Object.freeze({
  MANIFEST: 1,
  DOCUMENTS: 2,
  PASSAGE_INDEX: 3,
  PASSAGE_DATA: 4,
  LEXICAL_DICTIONARY: 5,
  LEXICAL_POSTINGS: 6,
  VECTOR_PROFILE: 7,
  VECTOR_DATA: 8,
  VECTOR_INDEX: 9,
  SIGNATURE: 10,
  LEXICAL_TERMS: 16,
  PASSAGE_RECORDS: 17,
});

// Lexical index section format versions this reader accepts. 1 is the original
// monolithic layout; 2 partitions the term table and posting stream into
// independently hashed blocks so a term costs a bounded range read. Mirrors
// SUPPORTED_LEXICAL_FORMAT_VERSIONS in rust/src/format.rs.
const SUPPORTED_LEXICAL_FORMAT_VERSIONS = Object.freeze([1, 2]);
// Passage index format versions. 1 stored records inline in the passage index
// as JSON; 2 moves them to fixed-width blocks addressed by ordinal, plus an
// id-sorted index. Mirrors rust/src/build.rs.
const SUPPORTED_PASSAGE_INDEX_FORMAT_VERSIONS = Object.freeze([1, 2]);
// Fixed-width record: block, offset, length as u32 LE. No passage id -- it is
// already in the id index and in the payload, and a third copy made packs
// larger than their source. See FORMAT-v3 §5.2.
const RECORD_STRIDE = 12;
// Fixed-width id-index entry: 32-byte id, then a u32 ordinal.
const ID_ENTRY_STRIDE = 36;

// Section types that may be marked required, and that appear at most once.
// LEXICAL_TERMS joins the set with lexical index format 2: a reader that cannot
// read it cannot resolve any term, so it must refuse the pack rather than
// search an index it only partly understands.
// 11, 14 and 15 are retired (AN-5, AN-9) and deliberately absent.
const KNOWN_REQUIRED_TYPES = new Set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12,
  SECTION.LEXICAL_TERMS, SECTION.PASSAGE_RECORDS]);
const KNOWN_SINGLETON_TYPES = KNOWN_REQUIRED_TYPES;

export class ANNPackBrowser {
  constructor(source, blake3, inflate, onRequest = null, originUrl = null) {
    if (typeof blake3 !== 'function') {
      throw new TypeError('ANNPackBrowser requires an async BLAKE3 function');
    }
    if (typeof source !== 'string' && !(source instanceof Uint8Array)) {
      throw new TypeError('ANNPackBrowser source must be a URL or Uint8Array');
    }
    this.url = typeof source === 'string' ? source : originUrl;
    this.memory = source instanceof Uint8Array ? source : null;
    this.mode = this.memory ? 'offline-memory' : 'remote-range';
    this.blake3 = blake3;
    this.inflate = inflate;
    this.onRequest = typeof onRequest === 'function' ? onRequest : null;
    this.requestLog = [];
    this.length = 0;
    this.etag = null;
    this.header = null;
    this.entries = [];
    this.entryByType = new Map();
    this.manifest = null;
    this.documents = [];
    this.documentById = new Map();
    this.passageIndex = null;
    this.dictionary = null;
    this.postings = null;
    this.vectorRuntime = null;
    this.conformance = null;
    this.publisher = null;
    this.passageBlockCache = new Map();
    this.stats = {
      requests: 0,
      rangeRequests: 0,
      memoryReads: 0,
      bytes: 0,
      installedBytes: this.memory?.length || 0,
    };
  }

  static async open(url, { blake3, inflate, onRequest = null }) {
    if (typeof inflate !== 'function') {
      throw new TypeError('ANNPackBrowser requires a bounded zlib inflate function');
    }
    const pack = new ANNPackBrowser(url, blake3, inflate, onRequest);
    await pack.open();
    return pack;
  }

  static async openBytes(bytes, {
    blake3,
    inflate,
    onRequest = null,
    originUrl = null,
    verifyAll = false,
  }) {
    if (typeof inflate !== 'function') {
      throw new TypeError('ANNPackBrowser requires a bounded zlib inflate function');
    }
    const source = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    const pack = new ANNPackBrowser(source, blake3, inflate, onRequest, originUrl);
    await pack.open();
    if (verifyAll) await pack.verifyAllSections();
    return pack;
  }

  async open() {
    if (this.memory) {
      this.length = this.memory.length;
    } else {
      const started = Date.now();
      const head = await fetch(this.url, { method: 'HEAD', cache: 'no-store', headers: { 'Accept-Encoding': 'identity' } });
      this.stats.requests += 1;
      this.recordRequest({
        kind: 'head',
        method: 'HEAD',
        status: head.status,
        bytes: 0,
        duration_ms: Date.now() - started,
      });
      if (!head.ok) throw new Error(`HEAD failed with HTTP ${head.status}`);
      this.length = parseSafeInteger(head.headers.get('content-length'), 'Content-Length');
      this.etag = head.headers.get('etag');
    }

    const headerBytes = await this.readRange(0, HEADER_SIZE);
    this.header = parseHeader(headerBytes);
    const expectedDirectoryLength = this.header.sectionCount * DIRECTORY_ENTRY_SIZE;
    if (this.header.directoryLength !== expectedDirectoryLength) {
      throw new Error('Directory length does not match section count');
    }
    const directoryBytes = await this.readRange(
      this.header.directoryOffset,
      this.header.directoryLength,
    );
    this.entries = parseDirectory(directoryBytes, this.length, this.header);
    const rootedDirectoryParts = [ROOT_CONTEXT];
    for (let index = 0; index < this.entries.length; index += 1) {
      if (this.entries[index].type !== SECTION.SIGNATURE) {
        rootedDirectoryParts.push(directoryBytes.slice(
          index * DIRECTORY_ENTRY_SIZE,
          (index + 1) * DIRECTORY_ENTRY_SIZE,
        ));
      }
      if (!this.entryByType.has(this.entries[index].type)) {
        this.entryByType.set(this.entries[index].type, this.entries[index]);
      }
    }
    const root = await this.hash(concat(rootedDirectoryParts));
    if (root !== this.header.rootHash) throw new Error('Content root does not match directory');

    const [manifest, documents, passageIndex, dictionary] = await Promise.all([
      this.readJsonSection(SECTION.MANIFEST),
      this.readJsonSection(SECTION.DOCUMENTS),
      this.readJsonSection(SECTION.PASSAGE_INDEX),
      this.readJsonSection(SECTION.LEXICAL_DICTIONARY),
    ]);
    this.manifest = manifest;
    this.documents = documents;
    this.passageIndex = passageIndex;
    this.dictionary = dictionary;

    // Resolve the lexical layout before reading anything large. Format 2 keeps
    // the term table and posting stream as independently hashed blocks and must
    // never read either section whole -- that read is the cost the layout
    // removes. Format 1 is read as before. Mirrors rust/src/search.rs.
    // Passage record layout. Format 2 keeps the table out of the open path;
    // format 1 packs still carry it inline. Mirrors rust/src/search.rs.
    this.recordBlocks = passageIndex.record_blocks || null;
    this.recordBlockCache = new Map();
    this.idBlockCache = new Map();
    this.passageCount = this.recordBlocks ? manifest.passage_count : passageIndex.records.length;
    if (this.recordBlocks) {
      if (!this.entryByType.has(SECTION.PASSAGE_RECORDS)) {
        throw new Error('Pack declares record block tables but carries no passage records section');
      }
      validateRecordBlocks(this.recordBlocks, this.requireSection(SECTION.PASSAGE_RECORDS), manifest.passage_count);
    }
    this.lexicalBlocks = passageIndex.lexical_blocks || null;
    this.termBlockCache = new Map();
    this.postingsBlockCache = new Map();
    if (this.lexicalBlocks) {
      if (!this.entryByType.has(SECTION.LEXICAL_TERMS)) {
        throw new Error('Pack declares lexical block tables but carries no lexical terms section');
      }
      this.postings = null;
      this.postingsStarts = validateLexicalBlocks(this.lexicalBlocks, {
        terms: this.requireSection(SECTION.LEXICAL_TERMS),
        postings: this.requireSection(SECTION.LEXICAL_POSTINGS),
      });
    } else {
      this.postings = await this.readSection(this.requireSection(SECTION.LEXICAL_POSTINGS));
      this.postingsStarts = null;
    }
    this.documentById = new Map(documents.map((document) => [document.id, document]));
    this.conformance = inspectConformance(this.entries, manifest);
    if (!this.conformance.core_conformant) {
      throw new Error(`Pack is not ${this.conformance.core_profile} conformant: ${this.conformance.issues.join('; ')}`);
    }
    this.publisher = {
      status: this.entries.some((entry) => entry.type === SECTION.SIGNATURE)
        ? 'not_verified'
        : 'unsigned',
      key_ids: [],
      asserted_identities: [],
      identity_trusted: false,
    };
    if (this.documentById.size !== documents.length || documents.length !== manifest.document_count) {
      throw new Error('Document identities or manifest document count are invalid');
    }
    if (dictionary.passage_lengths.length !== manifest.passage_count) {
      throw new Error('Passage and lexical index counts disagree');
    }
    if (!this.recordBlocks && passageIndex.records.length !== manifest.passage_count) {
      throw new Error('Passage index and manifest passage count disagree');
    }
    if (passageIndex.codec !== 'deflate-zlib') throw new Error('Unsupported passage block codec');
    if (!Number.isFinite(dictionary.average_passage_length) || dictionary.average_passage_length < 0) {
      throw new Error('Invalid average passage length');
    }
    this.validateSearchIndexes();
    return this;
  }

  async installOffline() {
    if (this.memory) {
      await this.verifyAllSections();
      return this;
    }
    const headers = {};
    if (this.etag) headers['If-Match'] = this.etag;
    const started = Date.now();
    const response = await fetch(this.url, { headers, cache: 'no-store' });
    this.stats.requests += 1;
    if (!response.ok) throw new Error(`Offline install failed with HTTP ${response.status}`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    this.stats.bytes += bytes.length;
    this.stats.installedBytes = bytes.length;
    this.recordRequest({
      kind: 'install',
      method: 'GET',
      status: response.status,
      bytes: bytes.length,
      duration_ms: Date.now() - started,
    });
    if (bytes.length !== this.length) {
      throw new Error(`Offline install returned ${bytes.length} bytes, expected ${this.length}`);
    }
    if (this.etag && response.headers.get('etag') && response.headers.get('etag') !== this.etag) {
      throw new Error('ETag changed during offline installation');
    }
    const installed = await ANNPackBrowser.openBytes(bytes, {
      blake3: this.blake3,
      inflate: this.inflate,
      onRequest: this.onRequest,
      originUrl: this.url,
      verifyAll: true,
    });
    if (installed.header.rootHash !== this.header.rootHash) {
      throw new Error('Offline installation root does not match the remote artifact');
    }
    return installed;
  }

  async verifyAllSections() {
    await Promise.all(this.entries.map((entry) => this.readSection(entry)));
    return {
      root_hash: this.header.rootHash,
      sections: this.entries.length,
      bytes: this.length,
    };
  }

  recordRequest(event) {
    const entry = Object.freeze({ sequence: this.requestLog.length + 1, mode: this.mode, ...event });
    this.requestLog.push(entry);
    if (this.onRequest) this.onRequest(entry);
  }

  validateSearchIndexes() {
    const data = this.requireSection(SECTION.PASSAGE_DATA);
    if (data.codec !== 0) throw new Error('Passage data must use independently compressed blocks');
    const ranges = [];
    this.passageIndex.blocks.forEach((block, index) => {
      const offset = toSafeNumber(block.offset, 'passage block offset');
      const storedLength = toSafeNumber(block.stored_length, 'passage block length');
      const logicalLength = toSafeNumber(block.logical_length, 'passage block logical length');
      if (logicalLength > MAX_PASSAGE_BLOCK_SIZE) throw new Error(`Passage block ${index} is too large`);
      if (storedLength === 0 && logicalLength !== 0) throw new Error(`Passage block ${index} is empty`);
      if (!/^[0-9a-f]{64}$/u.test(block.hash)) throw new Error(`Passage block ${index} has an invalid hash`);
      const end = offset + storedLength;
      if (!Number.isSafeInteger(end) || end > data.storedLength) throw new Error(`Passage block ${index} exceeds passage data`);
      ranges.push([offset, end, index]);
    });
    ranges.sort((left, right) => left[0] - right[0]);
    for (let index = 1; index < ranges.length; index += 1) {
      if (ranges[index - 1][1] > ranges[index][0]) throw new Error('Passage blocks overlap');
    }
    // Only the inline layout can be walked here; the blocked layout's coverage
    // is checked against the declared passage count in validateRecordBlocks.
    const passageIds = new Set();
    (this.recordBlocks ? [] : this.passageIndex.records).forEach((record) => {
      if (!/^[0-9a-f]{64}$/u.test(record.id) || passageIds.has(record.id)) throw new Error('Invalid or duplicate passage ID');
      passageIds.add(record.id);
      const block = this.passageIndex.blocks[record.block];
      if (!block) throw new Error(`Passage ${record.id} references a missing block`);
      const end = toSafeNumber(record.offset, 'passage offset') + toSafeNumber(record.length, 'passage length');
      if (!Number.isSafeInteger(end) || end > toSafeNumber(block.logical_length, 'passage block logical length')) {
        throw new Error(`Passage ${record.id} exceeds its logical block`);
      }
    });
    // Exhaustive validation is affordable only in the inline layout, where the
    // whole index is already resident. In the blocked layout it would fetch
    // every block at open -- exactly the cost the layout exists to avoid -- so
    // the block tables are validated instead (already done above) and per-posting
    // ordinals are bounds-checked at the point of use, in scoreLexical.
    if (!this.lexicalBlocks) {
      let postingCursor = 0;
      for (const [term, meta] of Object.entries(this.dictionary.terms).sort((a, b) => a[1].offset - b[1].offset)) {
        const offset = toSafeNumber(meta.offset, 'posting offset');
        const length = toSafeNumber(meta.length, 'posting length');
        if (offset !== postingCursor || !Number.isSafeInteger(meta.document_frequency) || meta.document_frequency < 1) {
          throw new Error(`Posting metadata for ${JSON.stringify(term)} is non-canonical`);
        }
        const end = offset + length;
        if (!Number.isSafeInteger(end) || end > this.postings.length) throw new Error(`Posting list for ${JSON.stringify(term)} exceeds its section`);
        for (const [ordinal] of decodePostings(this.postings.slice(offset, end), meta.document_frequency)) {
          if (ordinal >= this.passageCount) throw new Error(`Posting ordinal for ${JSON.stringify(term)} is invalid`);
        }
        postingCursor = end;
      }
      if (postingCursor !== this.postings.length) throw new Error('Dictionary does not cover postings exactly');
    }
  }

  async search(query, {
    limit = 10,
    mode = 'hybrid',
    queryVector = null,
    embed = null,
    vectorProfile = null,
    vectorProbes = 4,
    candidateDepth = 50,
    debug = false,
  } = {}) {
    if (!query.trim()) throw new Error('Query must not be empty');
    if (!Number.isSafeInteger(limit) || limit < 1 || limit > 1000) {
      throw new Error('Result limit must be between 1 and 1000');
    }
    if (!['lexical', 'vector', 'hybrid'].includes(mode)) throw new Error(`Unsupported search mode ${mode}`);
    if (!Number.isSafeInteger(candidateDepth) || candidateDepth < 1) throw new Error('Candidate depth must be positive');
    if (!Number.isSafeInteger(vectorProbes) || vectorProbes < 1 || vectorProbes > 1024) {
      throw new Error('Vector probes must be between 1 and 1024');
    }
    const terms = [...new Set(tokenize(query))];
    if (terms.length > 256) throw new Error('Query contains more than 256 terms');
    if (queryVector === null && embed !== null && mode !== 'lexical') {
      const runtime = await this.loadVectorRuntime();
      queryVector = await invokeEmbedding(embed, query, runtime.profile.profile);
    }
    if (mode === 'vector' && queryVector === null) {
      throw new Error('Vector mode requires queryVector or an embedding adapter');
    }
    const depth = Math.max(limit, candidateDepth);
    const lexicalResult = mode === 'vector'
      ? { candidates: [], achievable: 0 }
      : await this.lexicalCandidates(terms, depth);
    const lexical = lexicalResult.candidates;
    const vector = mode === 'lexical' || queryVector === null
      ? []
      : await this.vectorCandidates(queryVector, vectorProfile, vectorProbes, depth);
    const effectiveMode = lexical.length && vector.length
      ? 'hybrid'
      : (vector.length ? 'vector' : 'lexical');
    const candidates = fuseCandidates(
      lexical, lexicalResult.achievable, vector, effectiveMode,
    ).slice(0, limit);
    const results = await Promise.all(candidates.map(async ([ordinal, candidate], index) => {
      const passage = await this.getPassageByOrdinal(ordinal);
      const document = this.documentById.get(passage.document_id);
      if (!document) throw new Error(`Passage references unknown document ${passage.document_id}`);
      const url = citationUrl(document, passage);
      const passageHash = await this.hash(concat([
        EVIDENCE_CONTEXT,
        new TextEncoder().encode(JSON.stringify(passage)),
      ]));
      const evidence = {
        schema: 'annpack-evidence-v1',
        pack: `${this.manifest.name}@${this.manifest.version}`,
        pack_root: this.header.rootHash,
        source_revision: this.manifest.source_revision,
        passage_id: passage.id,
        passage_hash: passageHash,
        canonical_url: url,
        publisher: { ...this.publisher },
      };
      return {
        rank: index + 1,
        score: candidate.fusedScore,
        lexical_score: candidate.lexicalScore,
        vector_score: candidate.vectorScore,
        lexical_rank: candidate.lexicalRank,
        vector_rank: candidate.vectorRank,
        document_id: document.id,
        passage_id: passage.id,
        title: document.title,
        heading_path: passage.heading_path,
        url,
        source_path: document.source_path,
        text: passage.text,
        citation: {
          canonical_url: url,
          pack: `${this.manifest.name}@${this.manifest.version}`,
          pack_root: this.header.rootHash,
          passage_hash: passageHash,
          source_revision: this.manifest.source_revision,
        },
        evidence,
      };
    }));
    return {
      pack: {
        name: this.manifest.name,
        version: this.manifest.version,
        root_hash: this.header.rootHash,
        source_revision: this.manifest.source_revision,
        publisher: { ...this.publisher },
        conformance: this.conformance,
      },
      query,
      requested_mode: mode,
      effective_mode: effectiveMode,
      results,
      diagnostics: debug ? {
        query_terms: terms,
        lexical_candidates: lexical.length,
        vector_candidates: vector.length,
        vector_profile: vectorProfile,
        transfer: { ...this.stats },
      } : null,
    };
  }

  async lexicalCandidates(terms, depth) {
    const scores = new Map();
    // The score a passage answering every query term fully would earn. Used by
    // fuseCandidates to put BM25 on an absolute scale. Mirrors rust/src/search.rs.
    let achievable = 0;
    const passageCount = this.dictionary.passage_lengths.length;
    const averageLength = Math.max(1, this.dictionary.average_passage_length);
    // Two parallel rounds, not one serial pass per term. Resolving N terms
    // serially costs 2N sequential round trips; over a CDN that is the dominant
    // cost of a query. Block caches hold in-flight promises, so terms sharing a
    // block still issue exactly one fetch.
    const metas = await Promise.all(terms.map(async (term) => [term, await this.lookupTerm(term)]));
    const resolved = metas.filter(([, meta]) => meta);
    for (const [term, meta] of resolved) {
      if (!Number.isSafeInteger(meta.document_frequency) || meta.document_frequency < 1) {
        throw new Error(`Posting metadata for ${JSON.stringify(term)} is non-canonical`);
      }
    }
    const lists = await Promise.all(resolved.map(([, meta]) => this.postingBytes(meta)));

    for (let i = 0; i < resolved.length; i += 1) {
      const [term, meta] = resolved[i];
      const postings = decodePostings(lists[i], meta.document_frequency);
      const df = meta.document_frequency;
      const idf = Math.log(1 + (passageCount - df + 0.5) / (df + 0.5))
        * technicalBoost(term);
      achievable += idf * 2.2;
      for (const [ordinal, frequency] of postings) {
        if (ordinal >= passageCount) throw new Error('Posting ordinal exceeds passage count');
        const passageLength = this.dictionary.passage_lengths[ordinal];
        const denominator = frequency + 1.2 * (
          1 - 0.75 + 0.75 * passageLength / averageLength
        );
        const score = idf * frequency * 2.2 / denominator;
        scores.set(ordinal, (scores.get(ordinal) || 0) + score);
      }
    }
    return {
      candidates: [...scores.entries()]
        .sort((left, right) => right[1] - left[1] || left[0] - right[0])
        .slice(0, depth),
      achievable,
    };
  }

  async loadVectorRuntime() {
    if (this.vectorRuntime) return this.vectorRuntime;
    const profileEntry = this.entryByType.get(SECTION.VECTOR_PROFILE);
    const dataEntry = this.entryByType.get(SECTION.VECTOR_DATA);
    if (!profileEntry || !dataEntry) throw new Error('Pack has no complete vector representation');
    const indexEntry = this.entryByType.get(SECTION.VECTOR_INDEX);
    const [profile, data, index] = await Promise.all([
      this.readJsonSection(SECTION.VECTOR_PROFILE),
      this.readSection(dataEntry),
      indexEntry ? this.readJsonSection(SECTION.VECTOR_INDEX) : null,
    ]);
    const dimensions = profile.profile?.dimensions;
    if (!Number.isSafeInteger(dimensions) || dimensions < 1 || dimensions > 65536) {
      throw new Error('Vector profile has invalid dimensions');
    }
    if (profile.profile.dtype !== 'float32'
      || profile.passage_ids.length !== this.passageCount) {
      throw new Error('Vector profile does not match passage identities');
    }
    for (const field of ['id', 'model', 'revision', 'pooling']) {
      if (typeof profile.profile[field] !== 'string' || !profile.profile[field].trim()) {
        throw new Error(`Vector profile has invalid ${field}`);
      }
    }
    if (profile.profile.runtime) {
      for (const field of ['library', 'library_version', 'weights_dtype']) {
        if (typeof profile.profile.runtime[field] !== 'string'
          || !profile.profile.runtime[field].trim()) {
          throw new Error(`Vector runtime has invalid ${field}`);
        }
      }
      if (!Number.isSafeInteger(profile.profile.runtime.max_tokens)
        || profile.profile.runtime.max_tokens < 1) {
        throw new Error('Vector runtime has invalid max_tokens');
      }
    }
    const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
    if (data.length < 8) throw new Error('Truncated vector data');
    const count = view.getUint32(0, true);
    const storedDimensions = view.getUint32(4, true);
    const expectedLength = 8 + count * storedDimensions * 4;
    if (!Number.isSafeInteger(expectedLength) || count !== this.passageCount
      || storedDimensions !== dimensions || data.length !== expectedLength) {
      throw new Error('Vector data shape does not match its profile');
    }
    this.vectorRuntime = { profile, data, view, index, count, dimensions };
    return this.vectorRuntime;
  }

  async vectorCandidates(queryVector, requestedProfile, probes, depth) {
    const runtime = await this.loadVectorRuntime();
    if (requestedProfile !== null && requestedProfile !== runtime.profile.profile.id) {
      throw new Error(`Vector profile ${JSON.stringify(requestedProfile)} is unavailable`);
    }
    const query = Array.from(queryVector);
    if (query.length !== runtime.dimensions || query.some((value) => !Number.isFinite(value))) {
      throw new Error('Query vector has invalid dimensions or values');
    }
    const ordinals = runtime.index
      ? selectIvfOrdinals(runtime.index, query, runtime.count, runtime.dimensions, probes)
      : Array.from({ length: runtime.count }, (_, ordinal) => ordinal);
    const candidates = ordinals.map((ordinal) => {
      let score = 0;
      const base = 8 + ordinal * runtime.dimensions * 4;
      for (let dimension = 0; dimension < runtime.dimensions; dimension += 1) {
        const value = runtime.view.getFloat32(base + dimension * 4, true);
        if (!Number.isFinite(value)) throw new Error(`Stored vector ${ordinal} is non-finite`);
        score += query[dimension] * value;
      }
      return [ordinal, score];
    });
    return candidates
      .sort((left, right) => right[1] - left[1] || left[0] - right[0])
      .slice(0, depth);
  }

  // The record at a passage ordinal. In the blocked layout the containing block
  // is arithmetic, not a search: records are fixed width and uniformly packed.
  async recordAt(ordinal) {
    if (!this.recordBlocks) {
      const record = this.passageIndex.records[ordinal];
      if (!record) throw new Error(`Passage ordinal ${ordinal} is out of range`);
      return record;
    }
    if (!Number.isSafeInteger(ordinal) || ordinal < 0 || ordinal >= this.passageCount) {
      throw new Error(`Passage ordinal ${ordinal} is out of range`);
    }
    const perBlock = toSafeNumber(this.recordBlocks.per_block, 'records per block');
    const stride = toSafeNumber(this.recordBlocks.stride, 'record stride');
    const blockIndex = Math.floor(ordinal / perBlock);
    const block = this.recordBlocks.records[blockIndex];
    if (!block) throw new Error('Passage record block is missing');
    let pending = this.recordBlockCache.get(blockIndex);
    if (!pending) {
      pending = this.readIndexBlock(this.requireSection(SECTION.PASSAGE_RECORDS), block);
      this.recordBlockCache.set(blockIndex, pending);
    }
    const bytes = await pending;
    const at = (ordinal % perBlock) * stride;
    if (at + stride > bytes.length) throw new Error('Passage record exceeds its block');
    const view = new DataView(bytes.buffer, bytes.byteOffset + at, stride);
    return {
      id: null,
      block: view.getUint32(0, true),
      offset: view.getUint32(4, true),
      length: view.getUint32(8, true),
    };
  }

  async getPassage(id) {
    let ordinal;
    if (!this.recordBlocks) {
      ordinal = this.passageIndex.records.findIndex((record) => record.id === id);
    } else {
      ordinal = await this.ordinalOf(id);
    }
    if (ordinal === null || ordinal < 0) throw new Error(`Unknown passage ID ${id}`);
    return this.getPassageByOrdinal(ordinal);
  }

  // Binary search the id-sorted index: one sparse lookup to pick the block,
  // then a binary search within it, because entries are fixed width and sorted.
  async ordinalOf(id) {
    if (!/^[0-9a-f]{64}$/u.test(id)) return null;
    const blockIndex = sparseBlockForTerm(this.recordBlocks.ids, id);
    if (blockIndex === null) return null;
    let pending = this.idBlockCache.get(blockIndex);
    if (!pending) {
      pending = this.readIndexBlock(
        this.requireSection(SECTION.PASSAGE_RECORDS),
        this.recordBlocks.ids[blockIndex],
      );
      this.idBlockCache.set(blockIndex, pending);
    }
    const bytes = await pending;
    if (bytes.length % ID_ENTRY_STRIDE !== 0) {
      throw new Error('Passage id index block is not a whole number of entries');
    }
    const target = id;
    let low = 0;
    let high = bytes.length / ID_ENTRY_STRIDE;
    while (low < high) {
      const middle = (low + high) >> 1;
      const at = middle * ID_ENTRY_STRIDE;
      const candidate = toHex(bytes.subarray(at, at + 32));
      if (candidate < target) low = middle + 1;
      else if (candidate > target) high = middle;
      else return new DataView(bytes.buffer, bytes.byteOffset + at + 32, 4).getUint32(0, true);
    }
    return null;
  }

  async getPassageByOrdinal(ordinal) {
    const record = await this.recordAt(ordinal);
    if (!record) throw new Error(`Passage ordinal ${ordinal} is out of range`);
    const data = this.requireSection(SECTION.PASSAGE_DATA);
    const block = this.passageIndex.blocks[record.block];
    if (!block) throw new Error(`Passage ${record.id} references a missing block`);
    let logical = this.passageBlockCache.get(record.block);
    if (!logical) {
      const compressed = await this.readRange(
        data.offset + toSafeNumber(block.offset, 'passage block offset'),
        toSafeNumber(block.stored_length, 'passage block length'),
      );
      if (await this.hash(compressed) !== block.hash) throw new Error(`Passage block ${record.block} hash mismatch`);
      const logicalLength = toSafeNumber(block.logical_length, 'passage block logical length');
      logical = this.inflate(compressed, logicalLength);
      if (!(logical instanceof Uint8Array)) logical = new Uint8Array(logical);
      if (logical.length !== logicalLength) throw new Error(`Passage block ${record.block} inflated to the wrong length`);
      this.passageBlockCache.set(record.block, logical);
    }
    const start = toSafeNumber(record.offset, 'passage offset');
    const length = toSafeNumber(record.length, 'passage length');
    const bytes = logical.slice(start, start + length);
    if (bytes.length !== length) throw new Error(`Passage at ordinal ${ordinal} exceeds its block`);
    const passage = JSON.parse(new TextDecoder().decode(bytes));
    // A format-2 record carries no id, so the payload's own ordinal is what
    // detects a mis-seek. Compare the id only when the record actually has one.
    if ((record.id !== null && record.id !== undefined && passage.id !== record.id)
      || passage.ordinal !== ordinal) {
      throw new Error('Passage payload does not match its verified index');
    }
    return passage;
  }

  async verifySignatures({ trustedPublicKey = null } = {}) {
    if (!globalThis.crypto?.subtle) throw new Error('WebCrypto is unavailable');
    const reports = [];
    let trustedSignatureFound = false;
    for (const entry of this.entries.filter((value) => value.type === SECTION.SIGNATURE)) {
      const envelope = JSON.parse(new TextDecoder().decode(await this.readSection(entry)));
      if (envelope.algorithm !== 'Ed25519' || envelope.signed_root !== this.header.rootHash) {
        throw new Error('Signature envelope targets another root or algorithm');
      }
      const publicKey = fromHex(envelope.public_key);
      const signature = fromHex(envelope.signature);
      if (await this.hash(publicKey) !== envelope.key_id) throw new Error('Signature key ID mismatch');
      const key = await globalThis.crypto.subtle.importKey('raw', publicKey, 'Ed25519', false, ['verify']);
      const message = concat([
        new TextEncoder().encode('ANNPACK3-SIGNATURE\0'),
        fromHex(this.header.rootHash),
      ]);
      const valid = await globalThis.crypto.subtle.verify('Ed25519', key, signature, message);
      if (!valid) throw new Error(`Invalid signature in section ${entry.id}`);
      const trusted = trustedPublicKey !== null
        && trustedPublicKey.toLowerCase() === envelope.public_key.toLowerCase();
      trustedSignatureFound ||= trusted;
      reports.push({
        section_id: entry.id,
        key_id: envelope.key_id,
        identity: envelope.identity,
        cryptographically_valid: true,
        identity_trusted: trusted,
      });
    }
    if (trustedPublicKey !== null && !trustedSignatureFound) {
      throw new Error('No valid signature matches the trusted public key');
    }
    this.publisher = {
      status: reports.length ? 'cryptographically_verified' : 'unsigned',
      key_ids: reports.map((report) => report.key_id),
      asserted_identities: reports.map((report) => report.identity).filter(Boolean),
      identity_trusted: reports.some((report) => report.identity_trusted),
    };
    return reports;
  }

  async readJsonSection(type) {
    const bytes = await this.readSection(this.requireSection(type));
    return JSON.parse(new TextDecoder().decode(bytes));
  }

  requireSection(type) {
    const entry = this.entryByType.get(type);
    if (!entry) throw new Error(`Required section type ${type} is missing`);
    return entry;
  }

  // Fetch one stored block by range, verify it against the hash recorded in the
  // block table, then inflate it. The hash check is what makes a partial read
  // safe: a section hash only authenticates the section in full, so a block's
  // authenticity comes from the block table, which was itself read from a
  // hash-verified section. Mirrors read_index_block in rust/src/search.rs.
  async readIndexBlock(entry, block) {
    const offset = toSafeNumber(entry.offset, 'section offset')
      + toSafeNumber(block.offset, 'index block offset');
    const storedLength = toSafeNumber(block.stored_length, 'index block stored length');
    const logicalLength = toSafeNumber(block.logical_length, 'index block logical length');
    const stored = await this.readRange(offset, storedLength);
    if (await this.hash(stored) !== block.hash) {
      throw new Error(`Index block at offset ${block.offset} failed verification`);
    }
    const logical = this.inflate(stored, logicalLength);
    const bytes = logical instanceof Uint8Array ? logical : new Uint8Array(logical);
    if (bytes.length !== logicalLength) {
      throw new Error('Index block inflated to the wrong length');
    }
    return bytes;
  }

  // Posting metadata for one term, or null if absent. In the blocked layout
  // this costs at most one block read.
  async lookupTerm(term) {
    if (!this.lexicalBlocks) return this.dictionary.terms[term] || null;
    const index = sparseBlockForTerm(this.lexicalBlocks.dictionary, term);
    if (index === null) return null;
    let pending = this.termBlockCache.get(index);
    if (!pending) {
      pending = this.readIndexBlock(
        this.requireSection(SECTION.LEXICAL_TERMS),
        this.lexicalBlocks.dictionary[index],
      ).then((bytes) => JSON.parse(new TextDecoder().decode(bytes)).terms || {});
      this.termBlockCache.set(index, pending);
    }
    const block = await pending;
    return block[term] || null;
  }

  // The exact posting-list bytes for `meta`, reassembled from however many
  // postings blocks its byte range touches.
  async postingBytes(meta) {
    const start = toSafeNumber(meta.offset, 'posting offset');
    const length = toSafeNumber(meta.length, 'posting length');
    const end = start + length;
    if (!Number.isSafeInteger(end)) throw new Error('Posting range overflow');
    if (!this.lexicalBlocks) {
      const bytes = this.postings.slice(start, end);
      if (bytes.length !== length) throw new Error('Posting list exceeds its section');
      return bytes;
    }
    const entry = this.requireSection(SECTION.LEXICAL_POSTINGS);
    const parts = [];
    for (let index = 0; index < this.lexicalBlocks.postings.length; index += 1) {
      const block = this.lexicalBlocks.postings[index];
      const blockStart = this.postingsStarts[index];
      const blockEnd = blockStart + toSafeNumber(block.logical_length, 'postings block logical length');
      if (blockEnd <= start || blockStart >= end) continue;
      let pending = this.postingsBlockCache.get(index);
      if (!pending) {
        pending = this.readIndexBlock(entry, block);
        this.postingsBlockCache.set(index, pending);
      }
      const bytes = await pending;
      parts.push(bytes.slice(Math.max(start - blockStart, 0), Math.min(end, blockEnd) - blockStart));
    }
    const joined = concat(parts);
    if (joined.length !== length) {
      throw new Error('Posting list is not covered by the postings block table');
    }
    return joined;
  }

  async readSection(entry) {
    const stored = await this.readRange(entry.offset, entry.storedLength);
    if (await this.hash(stored) !== entry.hash) {
      throw new Error(`Section ${entry.id} hash mismatch`);
    }
    if (entry.codec === 0) return stored;
    if (entry.codec === 1) {
      const logical = this.inflate(stored, entry.logicalLength);
      const bytes = logical instanceof Uint8Array ? logical : new Uint8Array(logical);
      if (bytes.length !== entry.logicalLength) throw new Error(`Section ${entry.id} inflated to the wrong length`);
      return bytes;
    }
    throw new Error(`Unsupported section codec ${entry.codec}`);
  }

  async readRange(offset, length) {
    offset = toSafeNumber(offset, 'range offset');
    length = toSafeNumber(length, 'range length');
    const endExclusive = offset + length;
    if (!Number.isSafeInteger(endExclusive) || endExclusive > this.length) {
      throw new Error(`Range ${offset}..${endExclusive} exceeds artifact length ${this.length}`);
    }
    if (length === 0) return new Uint8Array();
    if (this.memory) {
      const bytes = this.memory.slice(offset, endExclusive);
      this.stats.memoryReads += 1;
      this.recordRequest({
        kind: 'memory',
        method: 'READ',
        start: offset,
        end: endExclusive - 1,
        status: 0,
        bytes: bytes.length,
        duration_ms: 0,
      });
      return bytes;
    }
    const end = endExclusive - 1;
    const headers = { Range: `bytes=${offset}-${end}` };
    const started = Date.now();
    const response = await fetch(this.url, { headers, cache: 'no-store' });
    this.stats.requests += 1;
    this.stats.rangeRequests += 1;
    if (response.status !== 206) {
      throw new Error(`Range request returned HTTP ${response.status}, expected 206`);
    }
    const expected = `bytes ${offset}-${end}/${this.length}`;
    if (response.headers.get('content-range') !== expected) {
      throw new Error(`Incorrect Content-Range; expected ${expected}`);
    }
    if (this.etag && response.headers.get('etag') && response.headers.get('etag') !== this.etag) {
      throw new Error('ETag changed during the read session');
    }
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes.length !== length) throw new Error(`Truncated range: got ${bytes.length}, expected ${length}`);
    this.stats.bytes += bytes.length;
    this.recordRequest({
      kind: 'range',
      method: 'GET',
      start: offset,
      end,
      status: response.status,
      bytes: bytes.length,
      duration_ms: Date.now() - started,
    });
    return bytes;
  }

  async hash(bytes) {
    const value = await this.blake3(bytes);
    return typeof value === 'string' ? value.toLowerCase() : toHex(value);
  }
}

function parseHeader(bytes) {
  if (bytes.length !== HEADER_SIZE || !MAGIC.every((value, index) => bytes[index] === value)) {
    throw new Error('Invalid ANNPack v3 header');
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(8, true) !== 3 || view.getUint32(12, true) !== HEADER_SIZE) {
    throw new Error('Unsupported ANNPack version or header size');
  }
  if (bytes.slice(80).some((value) => value !== 0)) throw new Error('Reserved header bytes must be zero');
  const sectionCount = view.getUint32(44, true);
  if (sectionCount < 1 || sectionCount > MAX_SECTIONS) throw new Error('Invalid section count');
  return {
    flags: getSafeUint64(view, 16, 'flags'),
    directoryOffset: getSafeUint64(view, 24, 'directory offset'),
    directoryLength: getSafeUint64(view, 32, 'directory length'),
    manifestSectionId: view.getUint32(40, true),
    sectionCount,
    rootHash: toHex(bytes.slice(48, 80)),
  };
}

function parseDirectory(bytes, fileLength, header) {
  if (bytes.length !== header.sectionCount * DIRECTORY_ENTRY_SIZE) throw new Error('Invalid directory');
  const entries = [];
  const ids = new Set();
  const singletonTypes = new Set();
  let previousId = -1;
  const directoryEnd = header.directoryOffset + header.directoryLength;
  for (let offset = 0; offset < bytes.length; offset += DIRECTORY_ENTRY_SIZE) {
    const view = new DataView(bytes.buffer, bytes.byteOffset + offset, DIRECTORY_ENTRY_SIZE);
    const entry = {
      id: view.getUint32(0, true),
      type: view.getUint16(4, true),
      formatVersion: view.getUint16(6, true),
      codec: view.getUint16(8, true),
      flags: view.getUint16(10, true),
      offset: getSafeUint64(view, 12, 'section offset'),
      storedLength: getSafeUint64(view, 20, 'stored length'),
      logicalLength: getSafeUint64(view, 28, 'logical length'),
      itemCount: getSafeUint64(view, 36, 'item count'),
      hash: toHex(new Uint8Array(view.buffer, view.byteOffset + 44, 32)),
    };
    if ([76, 77, 78, 79].some((index) => view.getUint8(index) !== 0)) {
      throw new Error(`Section ${entry.id} has nonzero reserved directory bytes`);
    }
    if (ids.has(entry.id)) throw new Error(`Duplicate section ID ${entry.id}`);
    if (entry.id <= previousId) throw new Error('Directory is not in strictly increasing section-ID order');
    previousId = entry.id;
    ids.add(entry.id);
    if (KNOWN_SINGLETON_TYPES.has(entry.type) && entry.type !== SECTION.SIGNATURE) {
      if (singletonTypes.has(entry.type)) throw new Error(`Duplicate singleton section type ${entry.type}`);
      singletonTypes.add(entry.type);
    }
    if (![0, 1].includes(entry.codec) && (entry.flags & 1)) throw new Error(`Unsupported required codec ${entry.codec}`);
    // A required section this reader does not know is a hard stop: it may carry
    // meaning that changes results, so serving a partial answer would be wrong.
    if (!KNOWN_REQUIRED_TYPES.has(entry.type) && (entry.flags & 1)) throw new Error(`Unsupported required section type ${entry.type}`);
    if (entry.logicalLength > MAX_LOGICAL_SECTION_SIZE || entry.storedLength > MAX_LOGICAL_SECTION_SIZE) throw new Error(`Section ${entry.id} is too large`);
    if (entry.codec === 0 && entry.storedLength !== entry.logicalLength) throw new Error(`Section ${entry.id} has mismatched lengths`);
    if (entry.codec === 1 && entry.logicalLength > 16 * 1024 * 1024
      && (entry.storedLength === 0
        || entry.logicalLength / entry.storedLength > DECOMPRESSION_RATIO_LIMIT)) {
      throw new Error(`Section ${entry.id} exceeds the decompression-ratio limit`);
    }
    if (entry.offset < HEADER_SIZE || entry.offset + entry.storedLength > fileLength) {
      throw new Error(`Section ${entry.id} exceeds artifact bounds`);
    }
    if (entry.offset < directoryEnd && entry.offset + entry.storedLength > header.directoryOffset) {
      throw new Error(`Section ${entry.id} overlaps the directory`);
    }
    entries.push(entry);
  }
  const ranges = entries.map((entry) => [entry.offset, entry.offset + entry.storedLength, entry.id])
    .sort((left, right) => left[0] - right[0]);
  for (let index = 1; index < ranges.length; index += 1) {
    if (ranges[index - 1][1] > ranges[index][0]) throw new Error('Sections overlap');
  }
  const manifest = entries.find((entry) => entry.id === header.manifestSectionId);
  if (!manifest || manifest.type !== SECTION.MANIFEST || !(manifest.flags & 1)
    || manifest.logicalLength > MAX_MANIFEST_SIZE) {
    throw new Error('Manifest section reference is invalid');
  }
  // The manifest schema is versioned independently of the ANNPACK3 wire format
  // (FORMAT-v3 §4.2). Version 2 dropped `builder` and added the logical content
  // root. Refuse an unknown version explicitly rather than mis-parsing it.
  if (!SUPPORTED_MANIFEST_FORMAT_VERSIONS.includes(manifest.formatVersion)) {
    throw new Error(
      `Unsupported manifest section format version ${manifest.formatVersion}; `
      + `this reader supports ${SUPPORTED_MANIFEST_FORMAT_VERSIONS.join(', ')}`,
    );
  }
  for (const type of [SECTION.DOCUMENTS, SECTION.PASSAGE_INDEX,
    SECTION.PASSAGE_DATA, SECTION.LEXICAL_DICTIONARY, SECTION.LEXICAL_POSTINGS]) {
    const entry = entries.find((value) => value.type === type);
    // The postings section carries its own schema version; every other
    // required section is v1 only.
    let accepted = [1];
    if (type === SECTION.LEXICAL_POSTINGS) accepted = SUPPORTED_LEXICAL_FORMAT_VERSIONS;
    else if (type === SECTION.PASSAGE_INDEX) accepted = SUPPORTED_PASSAGE_INDEX_FORMAT_VERSIONS;
    if (!entry || !(entry.flags & 1) || !accepted.includes(entry.formatVersion)) {
      throw new Error(`Required profile section ${type} is missing, optional, or at an unsupported format version`);
    }
  }
  return entries;
}

// Validate the lexical block tables and return each postings block's logical
// start offset. Everything checked here comes from the section directory, which
// the artifact root already authenticates, so a malformed table is rejected
// before any block is fetched. Blocks must tile their section exactly, and
// dictionary first terms must be strictly increasing because the sparse search
// assumes that ordering. Mirrors validate_lexical_blocks in rust/src/search.rs.
function validateLexicalBlocks(blocks, sections) {
  const tile = (list, entry, label) => {
    const starts = [];
    let storedCursor = 0;
    let logicalCursor = 0;
    for (const block of list) {
      const stored = toSafeNumber(block.stored_length, `${label} stored length`);
      const logical = toSafeNumber(block.logical_length, `${label} logical length`);
      if (toSafeNumber(block.offset, `${label} offset`) !== storedCursor) {
        throw new Error(`${label} blocks are not contiguous`);
      }
      if (stored === 0 || logical === 0) throw new Error(`${label} block is empty`);
      if (!/^[0-9a-f]{64}$/u.test(block.hash)) throw new Error(`${label} block has an invalid hash`);
      starts.push(logicalCursor);
      storedCursor += stored;
      logicalCursor += logical;
    }
    if (storedCursor !== toSafeNumber(entry.storedLength, `${label} section length`)) {
      throw new Error(`${label} blocks do not cover their section exactly`);
    }
    return starts;
  };

  tile(blocks.dictionary, sections.terms, 'lexical_terms');
  const postingsStarts = tile(blocks.postings, sections.postings, 'lexical_postings');

  let previous = null;
  for (const block of blocks.dictionary) {
    if (typeof block.first_term !== 'string') {
      throw new Error('Dictionary block is missing its first term');
    }
    if (previous !== null && block.first_term <= previous) {
      throw new Error('Dictionary block first terms must be strictly increasing');
    }
    previous = block.first_term;
  }
  return postingsStarts;
}

// The one dictionary block that can contain `term`: the last block whose
// first_term is less than or equal to it. Null means the term sorts before
// every block, so it is absent.
// Validate the record block tables against the section directory before any
// block is fetched. Both regions must tile the section exactly and cover every
// declared passage: a short table would make some ordinals silently
// unreachable rather than fail. Mirrors validate_record_blocks in
// rust/src/search.rs.
function validateRecordBlocks(index, entry, passageCount) {
  const stride = toSafeNumber(index.stride, 'record stride');
  const perBlock = toSafeNumber(index.per_block, 'records per block');
  if (stride === 0 || perBlock === 0) throw new Error('Record block index has a zero stride or block size');
  let cursor = 0;
  let recordBytes = 0;
  let idBytes = 0;
  for (const [label, list] of [['record', index.records], ['id', index.ids]]) {
    for (const block of list) {
      const stored = toSafeNumber(block.stored_length, `${label} stored length`);
      const logical = toSafeNumber(block.logical_length, `${label} logical length`);
      if (toSafeNumber(block.offset, `${label} offset`) !== cursor) throw new Error(`${label} blocks are not contiguous`);
      if (stored === 0 || logical === 0) throw new Error(`${label} block is empty`);
      if (!/^[0-9a-f]{64}$/u.test(block.hash)) throw new Error(`${label} block has an invalid hash`);
      cursor += stored;
      if (label === 'record') recordBytes += logical; else idBytes += logical;
    }
  }
  if (cursor !== toSafeNumber(entry.storedLength, 'records section length')) {
    throw new Error('Record blocks do not cover their section exactly');
  }
  if (recordBytes !== passageCount * stride) throw new Error('Record blocks do not cover every passage');
  if (idBytes !== passageCount * ID_ENTRY_STRIDE) throw new Error('Id index does not cover every passage');
  let previous = null;
  for (const block of index.ids) {
    if (typeof block.first_term !== 'string') throw new Error('Id index block is missing its first id');
    if (previous !== null && block.first_term <= previous) {
      throw new Error('Id index block first ids must be strictly increasing');
    }
    previous = block.first_term;
  }
}

function sparseBlockForTerm(blocks, term) {
  let candidate = null;
  for (let index = 0; index < blocks.length; index += 1) {
    const first = blocks[index].first_term;
    if (typeof first !== 'string') return null;
    if (first <= term) candidate = index;
    else break;
  }
  return candidate;
}

function inspectConformance(entries, manifest) {
  // Core and extension verdicts are computed independently: a malformed optional
  // descriptor must never be able to invalidate a structurally sound Core pack.
  // Mirrors rust/src/conformance.rs.
  const coreIssues = [];
  for (const capability of CORE_CAPABILITIES) {
    if (!manifest.capabilities?.includes(capability)) {
      coreIssues.push(`core capability ${capability} is not declared`);
    }
  }
  const coreConformant = coreIssues.length === 0;

  const extensionIssues = [];
  const extensions = [];
  const vectorCount = [SECTION.VECTOR_PROFILE, SECTION.VECTOR_DATA, SECTION.VECTOR_INDEX]
    .filter((type) => entries.some((entry) => entry.type === type)).length;
  if (vectorCount === 3) extensions.push('AN-1');
  else if (vectorCount !== 0) extensionIssues.push('AN-1 vector sections are incomplete');
  if (manifest.policy?.payment || manifest.policy?.encryption) extensions.push('AN-5');
  if (manifest.dependencies?.length) extensions.push('AN-6');
  extensions.sort();
  return {
    wire_format: 'ANNPACK3',
    core_profile: 'annpack-core-v1.0-draft',
    core_conformant: coreConformant,
    extensions_conformant: extensionIssues.length === 0,
    extensions,
    core_issues: coreIssues,
    extension_issues: extensionIssues,
    issues: [...coreIssues, ...extensionIssues],
  };
}

function decodePostings(bytes, expectedCount) {
  const postings = [];
  const cursor = { value: 0 };
  let ordinal = 0;
  for (let index = 0; index < expectedCount; index += 1) {
    const delta = decodeVarint(bytes, cursor);
    if (index !== 0 && delta === 0) throw new Error('Posting ordinals must be strictly increasing');
    ordinal = index === 0 ? delta : ordinal + delta;
    const frequency = decodeVarint(bytes, cursor);
    if (frequency < 1) throw new Error('Zero term frequency');
    postings.push([ordinal, frequency]);
  }
  if (cursor.value !== bytes.length) throw new Error('Posting list contains trailing bytes');
  return postings;
}

function decodeVarint(bytes, cursor) {
  let value = 0;
  let multiplier = 1;
  for (let index = 0; index < 10; index += 1) {
    if (cursor.value >= bytes.length) throw new Error('Truncated varint');
    const byte = bytes[cursor.value++];
    value += (byte & 0x7f) * multiplier;
    if (!Number.isSafeInteger(value)) throw new Error('Varint exceeds safe integer range');
    if ((byte & 0x80) === 0) return value;
    multiplier *= 128;
  }
  throw new Error('Non-terminating varint');
}

function selectIvfOrdinals(index, query, vectorCount, dimensions, requestedProbes) {
  if (index.algorithm !== 'ivf-flat-v1' || index.distance !== 'dot'
    || index.dimensions !== dimensions || !Array.isArray(index.centroids)
    || index.centroids.length < 1 || index.centroids.length !== index.lists?.length
    || !Number.isSafeInteger(index.default_probes) || index.default_probes < 1
    || index.default_probes > index.centroids.length) {
    throw new Error('Invalid or unsupported IVF vector index');
  }
  const seen = new Uint8Array(vectorCount);
  index.centroids.forEach((centroid, cluster) => {
    if (!Array.isArray(centroid) || centroid.length !== dimensions
      || centroid.some((value) => !Number.isFinite(value))) {
      throw new Error(`IVF centroid ${cluster} is invalid`);
    }
    if (!Array.isArray(index.lists[cluster])) throw new Error(`IVF list ${cluster} is invalid`);
    for (const ordinal of index.lists[cluster]) {
      if (!Number.isSafeInteger(ordinal) || ordinal < 0 || ordinal >= vectorCount || seen[ordinal]) {
        throw new Error('IVF lists contain a duplicate or invalid ordinal');
      }
      seen[ordinal] = 1;
    }
  });
  if (seen.some((value) => value === 0)) throw new Error('IVF lists do not cover all vectors');
  const clusters = index.centroids.map((centroid, cluster) => [
    cluster,
    centroid.reduce((score, value, dimension) => score + value * query[dimension], 0),
  ]).sort((left, right) => right[1] - left[1] || left[0] - right[0]);
  const probes = Math.min(requestedProbes, clusters.length);
  return clusters.slice(0, probes).flatMap(([cluster]) => index.lists[cluster]);
}

// Fuse lexical and vector candidates.
//
// Not reciprocal-rank fusion. RRF sums per-list ranks, which makes appearing in
// both lists worth about twice appearing at the top of one -- destructive when
// either retriever has no signal for a query. Measured on evals/corpora/, RRF
// ranked a passage lexical placed 47th above one vectors placed 1st. Each mode
// is instead put on an absolute scale: BM25 over the query's maximum achievable
// score, and cosine as-is. Mirrors fuse_candidates in rust/src/search.rs, which
// carries the full rationale and the measurements.
function fuseCandidates(lexical, lexicalAchievable, vector, effectiveMode) {
  const achievable = lexicalAchievable > 0 ? lexicalAchievable : 1;
  const candidates = new Map();
  const blank = () => ({
    fusedScore: 0,
    lexicalScore: null,
    vectorScore: null,
    lexicalRank: null,
    vectorRank: null,
  });
  lexical.forEach(([ordinal, score], index) => {
    const candidate = candidates.get(ordinal) || blank();
    candidate.lexicalScore = score;
    candidate.lexicalRank = index + 1;
    candidate.fusedScore += effectiveMode === 'lexical'
      ? score
      : Math.min(Math.max(score / achievable, 0), 1);
    candidates.set(ordinal, candidate);
  });
  vector.forEach(([ordinal, score], index) => {
    const candidate = candidates.get(ordinal) || blank();
    candidate.vectorScore = score;
    candidate.vectorRank = index + 1;
    // Cosine below zero points away from the query: no evidence, not evidence
    // against.
    candidate.fusedScore += effectiveMode === 'vector'
      ? score
      : Math.min(Math.max(score, 0), 1);
    candidates.set(ordinal, candidate);
  });
  return [...candidates.entries()].sort((left, right) => (
    right[1].fusedScore - left[1].fusedScore || left[0] - right[0]
  ));
}

async function invokeEmbedding(embed, text, profile) {
  const operation = typeof embed === 'function' ? embed : embed?.embedQuery?.bind(embed);
  if (!operation) throw new TypeError('Embedding adapter must be a function or expose embedQuery()');
  return normalizeEmbedding(await operation(text, profile));
}

function normalizeEmbedding(value) {
  if (typeof value?.tolist === 'function') value = value.tolist();
  if (value?.data !== undefined) value = value.data;
  if (Array.isArray(value) && value.length === 1 && Array.isArray(value[0])) [value] = value;
  if (!Array.isArray(value) && !ArrayBuffer.isView(value)) {
    throw new TypeError('Embedding provider did not return a vector');
  }
  return Array.from(value);
}

export function createEmbeddingAdapter(provider, descriptor = {}) {
  const operation = typeof provider === 'function'
    ? provider
    : provider?.embed?.bind(provider);
  if (!operation) throw new TypeError('Embedding provider must be a function or expose embed()');
  return {
    async embedQuery(text, profile) {
      for (const field of ['id', 'model', 'revision', 'dimensions']) {
        if (descriptor[field] !== undefined && descriptor[field] !== profile[field]) {
          throw new Error(`Embedding provider ${field} does not match pack profile`);
        }
      }
      if (descriptor.runtime !== undefined
        && JSON.stringify(descriptor.runtime) !== JSON.stringify(profile.runtime)) {
        throw new Error('Embedding provider runtime does not match pack profile');
      }
      const input = `${profile.query_prefix || ''}${text}`;
      const vector = normalizeEmbedding(await operation(input, {
        purpose: 'query',
        profile,
      }));
      if (vector.length !== profile.dimensions || vector.some((value) => !Number.isFinite(value))) {
        throw new Error('Embedding provider returned invalid dimensions or values');
      }
      return vector;
    },
  };
}

// Exported so the conformance suite can drive this runtime through the same
// adapter contract as any other implementation (spec/conformance/README.md).
// The browser implements tokenization, BM25 and container parsing separately
// from rust/, and nothing else forces the two to agree.
export function tokenize(text) {
  return text.normalize('NFKC').toLowerCase().split(/\s+/u).map((raw) => raw.replace(
    /^[^\p{L}\p{N}_\-.:/@#]+|[^\p{L}\p{N}_\-.:/@#]+$/gu,
    '',
  )).filter(Boolean);
}

function technicalBoost(term) {
  return /[0-9_\-.:/@#]/u.test(term) ? 3 : 1;
}

function citationUrl(document, passage) {
  if (!document.url) return null;
  if (passage.anchor && !document.url.includes('#')) return `${document.url}#${passage.anchor}`;
  return document.url;
}

function getSafeUint64(view, offset, label) {
  return toSafeNumber(view.getBigUint64(offset, true), label);
}

function parseSafeInteger(value, label) {
  if (value === null || !/^\d+$/u.test(value)) throw new Error(`Invalid ${label}`);
  return toSafeNumber(BigInt(value), label);
}

function toSafeNumber(value, label) {
  const number = typeof value === 'bigint' ? Number(value) : Number(value);
  if (!Number.isSafeInteger(number) || number < 0) throw new Error(`${label} exceeds safe integer range`);
  return number;
}

function concat(parts) {
  const length = parts.reduce((total, part) => total + part.length, 0);
  const output = new Uint8Array(length);
  let offset = 0;
  for (const part of parts) {
    output.set(part, offset);
    offset += part.length;
  }
  return output;
}

function toHex(bytes) {
  return [...bytes].map((value) => value.toString(16).padStart(2, '0')).join('');
}

function fromHex(value) {
  if (!/^[0-9a-f]+$/iu.test(value) || value.length % 2) throw new Error('Invalid hexadecimal value');
  return Uint8Array.from(value.match(/../gu), (pair) => Number.parseInt(pair, 16));
}

// ── EVIDENCE-v1: standalone receipt verification ─────────────────────────────
//
// Verifies with no artifact and no network: the receipt carries every byte
// needed. The chain is passage record -> Merkle path -> logical content root ->
// manifest -> directory -> artifact root, with an optional signature over the
// artifact root. Mirrors verify_receipt in rust/src/evidence.rs.

const PASSAGE_EVIDENCE_CONTEXT = new TextEncoder().encode('ANNPACK3-PASSAGE-EVIDENCE\0');
const NODE_CONTEXT = new TextEncoder().encode('ANNPACK3-EVIDENCE-NODE\0');
const SIGNATURE_CONTEXT = new TextEncoder().encode('ANNPACK3-SIGNATURE\0');
const RECEIPT_SCHEMA = 'annpack-receipt-v2';

function fromBase64(value, label) {
  try {
    return Uint8Array.from(atob(value), (c) => c.charCodeAt(0));
  } catch (error) {
    throw new Error(`${label} is not valid base64`);
  }
}

// `blake3` and `inflate` are supplied by the caller, as elsewhere in this module.
export async function verifyReceipt(receipt, { blake3, inflate }) {
  const hash = async (bytes) => {
    const value = await blake3(bytes);
    return typeof value === 'string' ? value.toLowerCase() : toHex(value);
  };

  if (receipt.schema !== RECEIPT_SCHEMA) {
    throw new Error(`Unsupported receipt schema ${JSON.stringify(receipt.schema)}`);
  }

  // 1. The passage record hashes to the claimed passage hash.
  const record = fromBase64(receipt.passage_record_b64, 'passage record');
  const leafHex = await hash(concat([PASSAGE_EVIDENCE_CONTEXT, record]));
  if (leafHex !== receipt.passage_hash) throw new Error('Passage record does not match passage_hash');

  // 2. The Merkle path folds the leaf to the logical content root. Sibling
  //    order is explicit rather than derived from an index, so a proof cannot
  //    be replayed against a different position.
  let node = fromHex(leafHex);
  for (const step of receipt.inclusion_proof) {
    const sibling = fromHex(step.sibling);
    if (sibling.length !== 32) throw new Error('Inclusion proof sibling is not 32 bytes');
    const pair = step.sibling_is_left ? concat([sibling, node]) : concat([node, sibling]);
    node = fromHex(await hash(concat([NODE_CONTEXT, pair])));
  }
  if (toHex(node) !== receipt.passage_merkle_root) {
    throw new Error('Inclusion proof does not reach passage_merkle_root');
  }

  // 3. The manifest commits to that logical content root.
  const manifestBytes = fromBase64(receipt.manifest_bytes_b64, 'manifest');
  const manifest = JSON.parse(new TextDecoder().decode(manifestBytes));
  if (manifest.passage_merkle_root !== receipt.passage_merkle_root) {
    throw new Error('Manifest does not commit the receipt\'s passage_merkle_root');
  }

  // 4. The manifest bytes match their section-directory entry.
  const directory = fromBase64(receipt.directory_b64, 'directory');
  if (directory.length % DIRECTORY_ENTRY_SIZE !== 0) {
    throw new Error('Directory is not a whole number of entries');
  }
  const entries = [];
  for (let at = 0; at < directory.length; at += DIRECTORY_ENTRY_SIZE) {
    entries.push(directory.subarray(at, at + DIRECTORY_ENTRY_SIZE));
  }
  const entryFor = (sectionId) => {
    const found = entries.find((raw) => new DataView(
      raw.buffer, raw.byteOffset, raw.byteLength,
    ).getUint32(0, true) === sectionId);
    if (!found) throw new Error(`Directory has no entry for section ${sectionId}`);
    return found;
  };
  const typeOf = (raw) => new DataView(raw.buffer, raw.byteOffset, raw.byteLength).getUint16(4, true);

  const manifestEntry = entryFor(receipt.manifest_section_id);
  if (typeOf(manifestEntry) !== SECTION.MANIFEST) {
    throw new Error('manifest_section_id does not reference a Manifest section');
  }
  if (await hash(manifestBytes) !== toHex(manifestEntry.subarray(44, 76))) {
    throw new Error('Manifest bytes do not match their directory entry hash');
  }

  // 5. The directory reproduces the artifact root, excluding signature entries
  //    exactly as the writer does.
  const rooted = [ROOT_CONTEXT];
  for (const raw of entries) if (typeOf(raw) !== SECTION.SIGNATURE) rooted.push(raw);
  if (await hash(concat(rooted)) !== receipt.pack_root) {
    throw new Error('Directory does not reproduce pack_root');
  }

  // 6. The receipt's claims about the passage match the authenticated record.
  const passage = JSON.parse(new TextDecoder().decode(record));
  if (passage.id !== receipt.passage_id) {
    throw new Error('passage_id does not match the authenticated record');
  }
  if (receipt.passage_ordinal !== undefined && passage.ordinal !== receipt.passage_ordinal) {
    throw new Error('passage_ordinal does not match the authenticated record');
  }

  // 7. Pack coordinate and source revision come from the authenticated
  //    manifest, not from unauthenticated receipt fields.
  if (receipt.pack !== `${manifest.name}@${manifest.version}`) {
    throw new Error('Pack coordinate does not match the authenticated manifest');
  }
  if (receipt.source_revision !== manifest.source_revision) {
    throw new Error('source_revision does not match the authenticated manifest');
  }

  // 8. A canonical URL claim must be derivable from the authenticated Documents
  //    section, so a receipt cannot assert an arbitrary URL.
  if (receipt.canonical_url !== null && receipt.canonical_url !== undefined) {
    if (!receipt.documents_bytes_b64 || receipt.documents_section_id === undefined) {
      throw new Error('canonical_url asserted without the Documents section');
    }
    const documentsBytes = fromBase64(receipt.documents_bytes_b64, 'documents section');
    const documentsEntry = entryFor(receipt.documents_section_id);
    if (typeOf(documentsEntry) !== SECTION.DOCUMENTS) {
      throw new Error('documents_section_id does not reference a Documents section');
    }
    if (await hash(documentsBytes) !== toHex(documentsEntry.subarray(44, 76))) {
      throw new Error('Documents bytes do not match their directory entry hash');
    }
    const view = new DataView(documentsEntry.buffer, documentsEntry.byteOffset, documentsEntry.byteLength);
    const codec = view.getUint16(8, true);
    const logicalLength = toSafeNumber(view.getBigUint64(28, true), 'documents logical length');
    const logical = codec === 0 ? documentsBytes : inflate(documentsBytes, logicalLength);
    const documents = JSON.parse(new TextDecoder().decode(
      logical instanceof Uint8Array ? logical : new Uint8Array(logical),
    ));
    const document = documents.find((d) => d.id === passage.document_id);
    if (!document) throw new Error('Documents section has no entry for the passage\'s document');
    let expected = document.url;
    if (document.url && passage.anchor && !document.url.includes('#')) {
      expected = `${document.url}#${passage.anchor}`;
    }
    if (expected !== receipt.canonical_url) {
      throw new Error('canonical_url is not derivable from authenticated bytes');
    }
  }

  // 9. A signature, when present, is over the artifact root under a
  //    domain-separated context. Validity is a separate claim from integrity,
  //    and publisher identity is separate again: a valid signature establishes
  //    neither.
  if (receipt.signature) {
    const envelope = receipt.signature;
    if (String(envelope.algorithm).toLowerCase() !== 'ed25519') {
      throw new Error(`Unsupported signature algorithm ${JSON.stringify(envelope.algorithm)}`);
    }
    if (!globalThis.crypto?.subtle) throw new Error('WebCrypto is unavailable');
    const publicKey = fromHex(envelope.public_key);
    if (await hash(publicKey) !== envelope.key_id) {
      throw new Error('Signature key_id does not match its public key');
    }
    const key = await globalThis.crypto.subtle.importKey('raw', publicKey, 'Ed25519', false, ['verify']);
    const message = concat([SIGNATURE_CONTEXT, fromHex(receipt.pack_root)]);
    const valid = await globalThis.crypto.subtle.verify(
      'Ed25519', key, fromHex(envelope.signature), message,
    );
    if (!valid) throw new Error('Signature is not valid over the artifact root');
  }
}

// ── Run bundles: one agent run's retrieval evidence ──────────────────────────
//
// A bundle is an envelope over receipts and adds no cryptography, so this is a
// loop over verifyReceipt above. Mirrors verify_run_bundle in rust/src/bundle.rs
// and must reach the same verdict for the same file -- web/smoke-bundle.mjs
// asserts that against the native CLI.
//
// The distinction this reports is the one a reader is most likely to get wrong:
// the receipts are attested, and the query, application, model and answer are
// carried alongside them, attested by nothing.

const RUN_BUNDLE_SCHEMA = 'annpack-run-bundle-v1';
const MAX_BUNDLE_RECEIPTS = 256;

export async function verifyRunBundle(bundle, { blake3, inflate, trustedPublicKey = null }) {
  if (bundle.schema !== RUN_BUNDLE_SCHEMA) {
    throw new Error(`Unsupported run bundle schema ${JSON.stringify(bundle.schema)}`);
  }
  const carried = Array.isArray(bundle.receipts) ? bundle.receipts : [];
  if (carried.length > MAX_BUNDLE_RECEIPTS) {
    throw new Error(`Run bundle carries ${carried.length} receipts, above the ${MAX_BUNDLE_RECEIPTS} limit`);
  }

  const issues = [];
  const receipts = [];
  const packRoots = [];
  const sourceRevisions = [];
  let receiptsVerified = 0;
  let allReceiptsSigned = true;
  let allSignersTrusted = true;

  for (const [index, receipt] of carried.entries()) {
    let verified = true;
    let failure = null;
    try {
      await verifyReceipt(receipt, { blake3, inflate });
    } catch (error) {
      verified = false;
      failure = error?.message ?? String(error);
    }
    if (verified) {
      receiptsVerified += 1;
      // Only an authenticated receipt may contribute its root or revision; a
      // failed receipt's self-declared values are strings the sender chose.
      if (!packRoots.includes(receipt.pack_root)) packRoots.push(receipt.pack_root);
      const revision = receipt.source_revision;
      if (revision && !sourceRevisions.includes(revision)) sourceRevisions.push(revision);
    } else {
      issues.push(`receipt ${index} for passage ${receipt.passage_id} did not verify: ${failure}`);
    }
    // verifyReceipt throws when a present signature is invalid, so a receipt
    // that verified and carries a signature carried a valid one.
    const signed = verified && Boolean(receipt.signature);
    if (!signed) allReceiptsSigned = false;
    const trusted = signed && trustedPublicKey !== null
      && receipt.signature.public_key?.toLowerCase() === trustedPublicKey.toLowerCase();
    if (!trusted) allSignersTrusted = false;
    receipts.push({
      index,
      passage_id: receipt.passage_id,
      pack: receipt.pack,
      pack_root: receipt.pack_root,
      verified,
      issues: failure ? [failure] : [],
    });
  }

  let answerHashConsistent = null;
  if (bundle.answer !== undefined && bundle.answer !== null
      && bundle.answer_hash !== undefined && bundle.answer_hash !== null) {
    const digest = await blake3(new TextEncoder().encode(bundle.answer));
    const hex = (typeof digest === 'string' ? digest : toHex(digest)).toLowerCase();
    answerHashConsistent = hex === String(bundle.answer_hash).toLowerCase();
    if (!answerHashConsistent) issues.push('answer_hash does not match the carried answer');
  }

  if (carried.length === 0) {
    issues.push('run bundle carries no receipts, so it attests nothing');
  }

  return {
    run_id: bundle.run_id,
    query: bundle.query,
    receipts_total: carried.length,
    receipts_verified: receiptsVerified,
    pack_roots: packRoots,
    source_revisions: sourceRevisions,
    // An empty bundle would satisfy both `every` conditions vacuously.
    all_receipts_signed: allReceiptsSigned && carried.length > 0,
    all_signers_trusted: allSignersTrusted && carried.length > 0 && trustedPublicKey !== null,
    answer_hash_consistent: answerHashConsistent,
    receipts,
    attested: carried.length > 0 && receiptsVerified === carried.length,
    issues,
  };
}
