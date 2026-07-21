import initWasm, { blake3_hex as blake3, inflate_zlib as inflate } from './pkg/annpack.js';
import { ANNPackBrowser } from './annpack-browser.js';

let wasmReady;

export class ANNPackSearchElement extends HTMLElement {
  #pack = null;

  #embed = null;

  constructor() {
    super();
    const root = this.attachShadow({ mode: 'open' });
    const style = document.createElement('style');
    style.textContent = `
      :host { display: block; font: 14px/1.5 system-ui, sans-serif; color: inherit; }
      form { display: flex; gap: .5rem; }
      input { flex: 1; min-width: 0; padding: .7rem; font: inherit; }
      button { padding: .7rem 1rem; font: inherit; cursor: pointer; }
      [part="status"] { margin: .6rem 0; opacity: .72; }
      article { border-top: 1px solid color-mix(in srgb, currentColor 20%, transparent); padding: .8rem 0; }
      h3 { margin: 0 0 .3rem; font-size: 1rem; }
      p { margin: .3rem 0; }
      code { font-size: .8em; opacity: .72; }
    `;
    const form = document.createElement('form');
    const input = document.createElement('input');
    input.type = 'search';
    input.required = true;
    input.disabled = true;
    input.setAttribute('part', 'input');
    const button = document.createElement('button');
    button.type = 'submit';
    button.disabled = true;
    button.setAttribute('part', 'button');
    button.textContent = 'Search';
    form.append(input, button);
    const status = document.createElement('div');
    status.part = 'status';
    status.textContent = 'Waiting to open knowledge pack…';
    const results = document.createElement('div');
    results.part = 'results';
    root.append(style, form, status, results);
    form.addEventListener('submit', (event) => this.#search(event));
  }

  set embeddingAdapter(value) {
    this.#embed = value;
  }

  get embeddingAdapter() {
    return this.#embed;
  }

  async connectedCallback() {
    const source = this.getAttribute('src');
    const input = this.shadowRoot.querySelector('input');
    const button = this.shadowRoot.querySelector('button');
    const status = this.shadowRoot.querySelector('[part="status"]');
    input.placeholder = this.getAttribute('placeholder') || 'Search this knowledge pack';
    if (!source) {
      status.textContent = 'ANNPack widget requires a src attribute.';
      return;
    }
    try {
      wasmReady ||= initWasm();
      await wasmReady;
      this.#pack = await ANNPackBrowser.open(source, { blake3, inflate });
      status.textContent = `${this.#pack.manifest.name}@${this.#pack.manifest.version} verified`;
      input.disabled = false;
      button.disabled = false;
    } catch (error) {
      status.textContent = `Knowledge pack open failed: ${error.message}`;
      this.dispatchEvent(new CustomEvent('annpack-error', { detail: error }));
    }
  }

  async #search(event) {
    event.preventDefault();
    if (!this.#pack) return;
    const input = this.shadowRoot.querySelector('input');
    const button = this.shadowRoot.querySelector('button');
    const status = this.shadowRoot.querySelector('[part="status"]');
    const results = this.shadowRoot.querySelector('[part="results"]');
    const limit = Number.parseInt(this.getAttribute('limit') || '5', 10);
    const mode = this.getAttribute('mode') || (this.#embed ? 'hybrid' : 'lexical');
    button.disabled = true;
    results.replaceChildren();
    try {
      const response = await this.#pack.search(input.value, {
        limit,
        mode,
        embed: this.#embed,
        debug: true,
      });
      status.textContent = `${response.results.length} results · ${response.pack.name}@${response.pack.version}`;
      for (const hit of response.results) {
        const article = document.createElement('article');
        article.setAttribute('part', 'result');
        const heading = document.createElement('h3');
        if (hit.url) {
          const link = document.createElement('a');
          link.href = hit.url;
          link.rel = 'noreferrer';
          link.textContent = hit.title;
          heading.append(link);
        } else {
          heading.textContent = hit.title;
        }
        const passage = document.createElement('p');
        passage.textContent = hit.text;
        const evidence = document.createElement('code');
        evidence.textContent = `${hit.citation.pack} · ${hit.passage_id.slice(0, 16)}…`;
        article.append(heading, passage, evidence);
        results.append(article);
      }
      this.dispatchEvent(new CustomEvent('annpack-results', { detail: response }));
    } catch (error) {
      status.textContent = `Search failed: ${error.message}`;
      this.dispatchEvent(new CustomEvent('annpack-error', { detail: error }));
    } finally {
      button.disabled = false;
    }
  }
}

if (!customElements.get('annpack-search')) {
  customElements.define('annpack-search', ANNPackSearchElement);
}
