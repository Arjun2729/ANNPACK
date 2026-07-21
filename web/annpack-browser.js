const HEADER_SIZE = 128;
const DIRECTORY_ENTRY_SIZE = 80;
const MAX_SECTIONS = 16384;
const MAX_MANIFEST_SIZE = 4 * 1024 * 1024;
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
});

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

    const [manifest, documents, passageIndex, dictionary, postings] = await Promise.all([
      this.readJsonSection(SECTION.MANIFEST),
      this.readJsonSection(SECTION.DOCUMENTS),
      this.readJsonSection(SECTION.PASSAGE_INDEX),
      this.readJsonSection(SECTION.LEXICAL_DICTIONARY),
      this.readSection(this.requireSection(SECTION.LEXICAL_POSTINGS)),
    ]);
    this.manifest = manifest;
    this.documents = documents;
    this.passageIndex = passageIndex;
    this.dictionary = dictionary;
    this.postings = postings;
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
    if (passageIndex.records.length !== dictionary.passage_lengths.length) {
      throw new Error('Passage and lexical index counts disagree');
    }
    if (passageIndex.records.length !== manifest.passage_count) {
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
    const passageIds = new Set();
    this.passageIndex.records.forEach((record) => {
      if (!/^[0-9a-f]{64}$/u.test(record.id) || passageIds.has(record.id)) throw new Error('Invalid or duplicate passage ID');
      passageIds.add(record.id);
      const block = this.passageIndex.blocks[record.block];
      if (!block) throw new Error(`Passage ${record.id} references a missing block`);
      const end = toSafeNumber(record.offset, 'passage offset') + toSafeNumber(record.length, 'passage length');
      if (!Number.isSafeInteger(end) || end > toSafeNumber(block.logical_length, 'passage block logical length')) {
        throw new Error(`Passage ${record.id} exceeds its logical block`);
      }
    });
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
        if (ordinal >= this.passageIndex.records.length) throw new Error(`Posting ordinal for ${JSON.stringify(term)} is invalid`);
      }
      postingCursor = end;
    }
    if (postingCursor !== this.postings.length) throw new Error('Dictionary does not cover postings exactly');
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
    const lexical = mode === 'vector' ? [] : this.lexicalCandidates(terms, depth);
    const vector = mode === 'lexical' || queryVector === null
      ? []
      : await this.vectorCandidates(queryVector, vectorProfile, vectorProbes, depth);
    const effectiveMode = lexical.length && vector.length
      ? 'hybrid'
      : (vector.length ? 'vector' : 'lexical');
    const candidates = fuseCandidates(lexical, vector, effectiveMode).slice(0, limit);
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

  lexicalCandidates(terms, depth) {
    const scores = new Map();
    const passageCount = this.dictionary.passage_lengths.length;
    const averageLength = Math.max(1, this.dictionary.average_passage_length);
    for (const term of terms) {
      const meta = this.dictionary.terms[term];
      if (!meta) continue;
      const start = toSafeNumber(meta.offset, 'posting offset');
      const length = toSafeNumber(meta.length, 'posting length');
      const bytes = this.postings.slice(start, start + length);
      if (bytes.length !== length) throw new Error(`Posting list for ${JSON.stringify(term)} exceeds its section`);
      const postings = decodePostings(bytes, meta.document_frequency);
      const df = meta.document_frequency;
      const idf = Math.log(1 + (passageCount - df + 0.5) / (df + 0.5))
        * technicalBoost(term);
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
    return [...scores.entries()]
      .sort((left, right) => right[1] - left[1] || left[0] - right[0])
      .slice(0, depth);
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
      || profile.passage_ids.length !== this.passageIndex.records.length
      || profile.passage_ids.some((id, ordinal) => id !== this.passageIndex.records[ordinal].id)) {
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
    if (!Number.isSafeInteger(expectedLength) || count !== this.passageIndex.records.length
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

  async getPassage(id) {
    const ordinal = this.passageIndex.records.findIndex((record) => record.id === id);
    if (ordinal < 0) throw new Error(`Unknown passage ID ${id}`);
    return this.getPassageByOrdinal(ordinal);
  }

  async getPassageByOrdinal(ordinal) {
    const record = this.passageIndex.records[ordinal];
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
    if (bytes.length !== length) throw new Error(`Passage ${record.id} exceeds its block`);
    const passage = JSON.parse(new TextDecoder().decode(bytes));
    if (passage.id !== record.id || passage.ordinal !== ordinal) {
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
    if (entry.type >= 1 && entry.type <= 12 && entry.type !== SECTION.SIGNATURE) {
      if (singletonTypes.has(entry.type)) throw new Error(`Duplicate singleton section type ${entry.type}`);
      singletonTypes.add(entry.type);
    }
    if (![0, 1].includes(entry.codec) && (entry.flags & 1)) throw new Error(`Unsupported required codec ${entry.codec}`);
    if ((entry.type < 1 || entry.type > 12) && (entry.flags & 1)) throw new Error(`Unsupported required section type ${entry.type}`);
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
    || manifest.formatVersion !== 1 || manifest.logicalLength > MAX_MANIFEST_SIZE) {
    throw new Error('Manifest section reference is invalid');
  }
  for (const type of [SECTION.MANIFEST, SECTION.DOCUMENTS, SECTION.PASSAGE_INDEX,
    SECTION.PASSAGE_DATA, SECTION.LEXICAL_DICTIONARY, SECTION.LEXICAL_POSTINGS]) {
    const entry = entries.find((value) => value.type === type);
    if (!entry || !(entry.flags & 1) || entry.formatVersion !== 1) {
      throw new Error(`Required v1 profile section ${type} is missing or optional`);
    }
  }
  return entries;
}

function inspectConformance(entries, manifest) {
  const issues = [];
  for (const capability of CORE_CAPABILITIES) {
    if (!manifest.capabilities?.includes(capability)) {
      issues.push(`core capability ${capability} is not declared`);
    }
  }
  const coreConformant = issues.length === 0;
  const extensions = [];
  const vectorCount = [SECTION.VECTOR_PROFILE, SECTION.VECTOR_DATA, SECTION.VECTOR_INDEX]
    .filter((type) => entries.some((entry) => entry.type === type)).length;
  if (vectorCount === 3) extensions.push('ANN-1');
  else if (vectorCount !== 0) issues.push('ANN-1 vector sections are incomplete');
  if (manifest.policy?.payment || manifest.policy?.encryption) extensions.push('ANN-5');
  if (manifest.dependencies?.length) extensions.push('ANN-6');
  extensions.sort();
  return {
    wire_format: 'ANNPACK3',
    core_profile: 'annpack-core-v1.0-draft',
    core_conformant: coreConformant,
    extensions,
    issues,
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

function fuseCandidates(lexical, vector, effectiveMode) {
  const candidates = new Map();
  lexical.forEach(([ordinal, score], index) => {
    const candidate = candidates.get(ordinal) || {
      fusedScore: 0,
      lexicalScore: null,
      vectorScore: null,
      lexicalRank: null,
      vectorRank: null,
    };
    candidate.lexicalScore = score;
    candidate.lexicalRank = index + 1;
    candidate.fusedScore += effectiveMode === 'lexical' ? score : 1 / (60 + index + 1);
    candidates.set(ordinal, candidate);
  });
  vector.forEach(([ordinal, score], index) => {
    const candidate = candidates.get(ordinal) || {
      fusedScore: 0,
      lexicalScore: null,
      vectorScore: null,
      lexicalRank: null,
      vectorRank: null,
    };
    candidate.vectorScore = score;
    candidate.vectorRank = index + 1;
    candidate.fusedScore += effectiveMode === 'vector' ? score : 1 / (60 + index + 1);
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

function tokenize(text) {
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
