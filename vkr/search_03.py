"""
Unified Retriever — единый интерфейс поиска
====================================================
Поддерживает: tfidf, bm25, sbert, bge

Параметр suffix задаёт суффикс папки индексов и файла чанков:
  suffix=""            - indexes/          chunks.jsonl          (базовый, 512 слов)
  suffix="window"      - indexes_window/   chunks_window.jsonl   (512, overlap=128)
  suffix="c256"        - indexes_c256/     chunks_c256.jsonl     (256 слов)
  suffix="c256_window" - indexes_c256_window/ chunks_c256_window.jsonl

Использование:
  from search_03 import Retriever
  r = Retriever()                        # базовый (512, fixed)
  r = Retriever(suffix="window")         # 512, sliding window
  r = Retriever(suffix="c256")           # 256, fixed
  r = Retriever(suffix="c256_window")    # 256, sliding window
  results = r.search("запрос", method="bge", k=10)
"""

import os, json, pickle, time
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
BGE_MODEL = "BAAI/bge-m3"

def get_device():
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class Retriever:

    def __init__(self, methods=None, window=False, suffix=None):
        """
        suffix — суффикс папки индексов (без ведущего "_").
        """
        # Определяем суффикс
        if suffix is not None:
            self.suffix = suffix
        elif window:
            self.suffix = "window"
        else:
            self.suffix = ""

        sfx = f"_{self.suffix}" if self.suffix else ""
        self.index_dir = os.path.join(BASE_DIR, f"indexes{sfx}" if sfx else "indexes")
        self.chunks_f  = os.path.join(BASE_DIR, "data", f"chunks{sfx}.jsonl")

        self.window = (self.suffix == "window")

        self._tfidf_vec = self._tfidf_mat = self._tfidf_ids = None
        self._bm25 = self._bm25_ids = None
        self._sbert_model = self._sbert_index = self._sbert_ids = None
        self._bge_model = self._bge_index = self._bge_ids   = None
        self._chunks_cache: dict[str, str] = {}

        if methods is None:
            methods = ["tfidf", "bm25", "sbert", "bge"]

        for m in methods:
            self._load(m)

        self._load_cache()

    def _load(self, method):
        d = self.index_dir

        if method == "tfidf":
            from scipy.sparse import load_npz
            self._tfidf_vec = pickle.load(open(f"{d}/tfidf_vectorizer.pkl","rb"))
            self._tfidf_mat = load_npz(f"{d}/tfidf_matrix.npz")
            self._tfidf_ids = json.load(open(f"{d}/tfidf_meta.json", encoding="utf-8"))
            print(f"TF-IDF ({len(self._tfidf_ids)} чанков)")

        elif method == "bm25":
            self._bm25 = pickle.load(open(f"{d}/bm25.pkl","rb"))
            self._bm25_ids = json.load(open(f"{d}/bm25_meta.json", encoding="utf-8"))
            print(f"BM25 ({len(self._bm25_ids)} чанков)")

        elif method == "sbert":
            import faiss
            from sentence_transformers import SentenceTransformer
            self._sbert_index = faiss.read_index(f"{d}/sbert_index.faiss")
            self._sbert_ids = json.load(open(f"{d}/sbert_meta.json", encoding="utf-8"))
            self._sbert_model = SentenceTransformer(SBERT_MODEL, device=get_device())
            print(f"SBERT ({self._sbert_index.ntotal} векторов, dim={self._sbert_index.d})")

        elif method == "bge":
            import faiss
            from sentence_transformers import SentenceTransformer
            self._bge_index = faiss.read_index(f"{d}/bge_index.faiss")
            self._bge_ids = json.load(open(f"{d}/bge_meta.json", encoding="utf-8"))
            self._bge_model = SentenceTransformer(BGE_MODEL, device=get_device())
            print(f"BGE-M3 ({self._bge_index.ntotal} векторов, dim={self._bge_index.d})")

    def _load_cache(self):
        if not os.path.exists(self.chunks_f):
            return
        with open(self.chunks_f, encoding="utf-8") as f:
            for line in f:
                ch = json.loads(line)
                self._chunks_cache[ch["chunk_id"]] = ch["text"]

    # поиск

    def _tfidf_search(self, query, k):
        from sklearn.metrics.pairwise import cosine_similarity
        qv = self._tfidf_vec.transform([query])
        scores = cosine_similarity(qv, self._tfidf_mat).flatten()
        top = np.argsort(scores)[::-1][:k]
        return [{"chunk_id": self._tfidf_ids[i], "score": float(scores[i])}
                for i in top if scores[i] > 0]

    def _bm25_search(self, query, k):
        scores = self._bm25.get_scores(query.lower().split())
        top = np.argsort(scores)[::-1][:k]
        return [{"chunk_id": self._bm25_ids[i], "score": float(scores[i])}
                for i in top if scores[i] > 0]

    def _dense_search(self, query, model, index, ids, k):
        qv = model.encode([query], normalize_embeddings=True,
                          convert_to_numpy=True).astype(np.float32)
        scores, indices = index.search(qv, k)
        return [{"chunk_id": ids[int(idx)], "score": float(sc)}
                for sc, idx in zip(scores[0], indices[0]) if idx != -1]

    def search(self, query: str, method: str, k: int = 10) -> list[dict]:
        """
        Возвращает список {"chunk_id": ..., "score": ...}
        method: tfidf | bm25 | sbert | bge
        """
        if method == "tfidf":
            return self._tfidf_search(query, k)
        elif method == "bm25":
            return self._bm25_search(query, k)
        elif method == "sbert":
            return self._dense_search(query, self._sbert_model,
                                       self._sbert_index, self._sbert_ids, k)
        elif method == "bge":
            return self._dense_search(query, self._bge_model,
                                       self._bge_index, self._bge_ids, k)
        else:
            raise ValueError(f"Неизвестный метод: {method}")

    def search_timed(self, query: str, method: str, k: int = 10):
        """Возвращает (results, elapsed_ms)"""
        t0      = time.perf_counter()
        results = self.search(query, method, k)
        return results, (time.perf_counter() - t0) * 1000

    def get_chunk_text(self, chunk_id: str) -> str:
        return self._chunks_cache.get(chunk_id, "[чанк не найден]")