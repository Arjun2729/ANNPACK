import numpy as np
from annpack.reader import ANNPackIndex


def main():
    index_path = "wikipedia_1M.annpack"  # or generic_index.annpack
    with ANNPackIndex.open(index_path, probe=8) as idx:
        dim = idx.header.dim
        q = np.random.randn(dim).astype(np.float32)
        q /= np.linalg.norm(q)
        results = idx.search(q, k=5)
        for rank, (id_, score) in enumerate(results, start=1):
            print(f"#{rank}: id={id_} score={score:.4f}")


if __name__ == "__main__":
    main()
