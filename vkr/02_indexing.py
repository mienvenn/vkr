"""
Индексация всех методов
================================
Запуск:
  python 02_indexing.py                        # базовые чанки (chunks.jsonl → indexes/)
  python 02_indexing.py --suffix window        # chunks_window.jsonl → indexes_window/
  python 02_indexing.py --suffix c256          # chunks_c256.jsonl   → indexes_c256/
  python 02_indexing.py --suffix c256_window   # chunks_c256_window.jsonl → indexes_c256_window/
  python 02_indexing.py --methods tfidf bm25   # только отдельные методы

Соответствие суффиксов (генерируются в 01_chunking.py):
  ""           - chunk_size=512, overlap=0   (базовый)
  "window"     - chunk_size=512, overlap=128
  "c256"       - chunk_size=256, overlap=0
  "c256_window"- chunk_size=256, overlap=64
"""

import os, json, pickle, time, argparse
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
BGE_MODEL = "BAAI/bge-m3"
BATCH_SIZE  = 32


def get_device():
    """MPS (Apple GPU) или CPU fallback"""
    try:
        import torch
        if torch.backends.mps.is_available():
            print("  Устройство: MPS (Apple GPU)")
            return "mps"
    except ImportError:
        pass
    print("Устройство: CPU")
    return "cpu"


def load_chunks(filepath):
    with open(filepath, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def save_meta(ids, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False)


# TF-IDF 

def build_tfidf(chunks, index_dir):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import save_npz

    print("\n[TF-IDF]")
    texts = [ch["text"] for ch in chunks]
    ids = [ch["chunk_id"] for ch in chunks]

    t0 = time.perf_counter()
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95, sublinear_tf=True)
    matrix = vec.fit_transform(texts)
    elapsed = time.perf_counter() - t0

    os.makedirs(index_dir, exist_ok=True)
    pickle.dump(vec, open(f"{index_dir}/tfidf_vectorizer.pkl", "wb"))
    save_npz(f"{index_dir}/tfidf_matrix.npz", matrix)
    save_meta(ids, f"{index_dir}/tfidf_meta.json")

    print(f"Словарь: {len(vec.vocabulary_):,} термов")
    print(f"Матрица: {matrix.shape[0]} × {matrix.shape[1]}")
    print(f"Время индексации: {elapsed:.2f} сек")
    print(f"Сохранено - {index_dir}/tfidf_*")


# BM25

def build_bm25(chunks, index_dir):
    from rank_bm25 import BM25Okapi

    print("\n[BM25]")
    texts = [ch["text"] for ch in chunks]
    ids = [ch["chunk_id"] for ch in chunks]
    tokenized = [t.lower().split() for t in texts]

    t0 = time.perf_counter()
    bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)
    elapsed = time.perf_counter() - t0

    os.makedirs(index_dir, exist_ok=True)
    pickle.dump(bm25, open(f"{index_dir}/bm25.pkl", "wb"))
    save_meta(ids, f"{index_dir}/bm25_meta.json")

    print(f"Документов: {len(tokenized)}")
    print(f"Время индексации: {elapsed:.2f} сек")
    print(f"Сохранено - {index_dir}/bm25*")


# SBERT 

def build_sbert(chunks, index_dir):
    import faiss
    from sentence_transformers import SentenceTransformer

    print(f"\n[SBERT] {SBERT_MODEL}")
    device = get_device()
    model = SentenceTransformer(SBERT_MODEL, device=device)
    texts = [ch["text"] for ch in chunks]
    ids = [ch["chunk_id"] for ch in chunks]

    t0 = time.perf_counter()
    emb = model.encode(
        texts, batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    elapsed = time.perf_counter() - t0

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, f"{index_dir}/sbert_index.faiss")
    save_meta(ids, f"{index_dir}/sbert_meta.json")

    print(f"Векторов: {index.ntotal}, dim={dim}")
    print(f"Время индексации: {elapsed:.2f} сек  ({elapsed/len(texts)*1000:.1f} мс/чанк)")
    print(f"Сохранено - {index_dir}/sbert_*")


# BGE-M3 

def build_bge(chunks, index_dir):
    import faiss
    from sentence_transformers import SentenceTransformer

    print(f"\n[BGE-M3] {BGE_MODEL}")
    device = get_device()
    model = SentenceTransformer(BGE_MODEL, device=device)
    texts = [ch["text"] for ch in chunks]
    ids = [ch["chunk_id"] for ch in chunks]

    t0 = time.perf_counter()
    emb = model.encode(
        texts, batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    elapsed = time.perf_counter() - t0

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, f"{index_dir}/bge_index.faiss")
    save_meta(ids, f"{index_dir}/bge_meta.json")

    print(f"Векторов: {index.ntotal}, dim={dim}")
    print(f"Время индексации: {elapsed:.2f} сек  ({elapsed/len(texts)*1000:.1f} мс/чанк)")
    print(f"Сохранено - {index_dir}/bge_*")


# main 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix",  type=str, default="",
                        help="Суффикс файлов чанков и папки индексов. "
                             "Примеры: window, c256, c256_window. "
                             "Пусто = базовый (chunks.jsonl → indexes/).")
    parser.add_argument("--methods", nargs="+",
                        default=["tfidf","bm25","sbert","bge"],
                        choices=["tfidf","bm25","sbert","bge"])
    args = parser.parse_args()

    sfx = f"_{args.suffix}" if args.suffix else ""
    chunks_f  = os.path.join(BASE_DIR, "data", f"chunks{sfx}.jsonl")
    index_dir = os.path.join(BASE_DIR, f"indexes{sfx}" if sfx else "indexes")

    print("=" * 60)
    print(f"ШАГ 2: ИНДЕКСАЦИЯ  [суффикс: '{args.suffix or 'базовый'}']")
    print(f"  Чанки:   {chunks_f}")
    print(f"  Индексы: {index_dir}")
    print(f"  Методы:  {args.methods}")
    print("=" * 60)

    if not os.path.exists(chunks_f):
        sfx_arg = f" --suffix {args.suffix}" if args.suffix else ""
        print(f"Файл {chunks_f} не найден.")
        print(f"Сначала запусти 01_chunking.py с нужными параметрами.")
        exit(1)

    chunks = load_chunks(chunks_f)
    print(f"Загружено {len(chunks)} чанков\n")

    total_t0 = time.perf_counter()

    if "tfidf" in args.methods:
        build_tfidf(chunks, index_dir)
    if "bm25"  in args.methods:
        build_bm25(chunks, index_dir)
    if "sbert" in args.methods:
        build_sbert(chunks, index_dir)
    if "bge"   in args.methods:
        build_bge(chunks, index_dir)

    total = time.perf_counter() - total_t0
    print(f"\nИндексация завершена за {total:.1f} сек")
    print(f"Индексы сохранены в: {index_dir}/")