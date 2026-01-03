export type CacheEntry = {
  key: string;
  bytes: Uint8Array;
  etag?: string;
};

export interface Cache {
  get(key: string): Promise<CacheEntry | undefined>;
  set(entry: CacheEntry): Promise<void>;
  delete(key: string): Promise<void>;
  stats(): Record<string, unknown>;
}

export class MemoryCache implements Cache {
  private store = new Map<string, CacheEntry>();

  async get(key: string): Promise<CacheEntry | undefined> {
    return this.store.get(key);
  }

  async set(entry: CacheEntry): Promise<void> {
    this.store.set(entry.key, entry);
  }

  async delete(key: string): Promise<void> {
    this.store.delete(key);
  }

  stats(): Record<string, unknown> {
    return { entries: this.store.size };
  }
}

export class IndexedDBCache implements Cache {
  private dbName: string;
  private storeName: string;

  constructor(dbName = 'annpack', storeName = 'ranges') {
    this.dbName = dbName;
    this.storeName = storeName;
  }

  async get(key: string): Promise<CacheEntry | undefined> {
    const db = await this.open();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(this.storeName, 'readonly');
      const req = tx.objectStore(this.storeName).get(key);
      req.onsuccess = () => resolve(req.result as CacheEntry | undefined);
      req.onerror = () => reject(req.error);
    });
  }

  async set(entry: CacheEntry): Promise<void> {
    const db = await this.open();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(this.storeName, 'readwrite');
      tx.objectStore(this.storeName).put(entry);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  async delete(key: string): Promise<void> {
    const db = await this.open();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(this.storeName, 'readwrite');
      tx.objectStore(this.storeName).delete(key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  stats(): Record<string, unknown> {
    return { backend: 'indexeddb' };
  }

  private open(): Promise<IDBDatabase> {
    if (typeof indexedDB === 'undefined') {
      return Promise.reject(new Error('IndexedDB not available'));
    }
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(this.dbName, 1);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(this.storeName)) {
          db.createObjectStore(this.storeName, { keyPath: 'key' });
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }
}
