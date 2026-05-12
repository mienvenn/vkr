"""
Препроцессинг и чанкинг
================================
Запуск:
  python 01_chunking.py                          # фиксированный чанк 512 слов
  python 01_chunking.py --overlap                # sliding window, 512 слов, overlap=128
  python 01_chunking.py --chunk-size 256         # фиксированный чанк 256 слов
  python 01_chunking.py --chunk-size 256 --overlap  # sliding window, 256 слов, overlap=64

Именование выходных файлов (суффикс формируется автоматически):
  chunk_size=512, overlap=0   →  chunks.jsonl          / qrels.jsonl          (базовый)
  chunk_size=512, overlap=128 →  chunks_window.jsonl   / qrels_window.jsonl
  chunk_size=256, overlap=0   →  chunks_c256.jsonl     / qrels_c256.jsonl
  chunk_size=256, overlap=64  →  chunks_c256_window.jsonl / qrels_c256_window.jsonl
"""

import os, json, re, argparse

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(BASE_DIR, "corpus")

DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP_512 = 128   
DEFAULT_OVERLAP_256 = 64   


def make_suffix(chunk_size: int, overlap: int) -> str:
    """
    Генерирует суффикс имён файлов из параметров чанкинга.
    """
    parts = []
    if chunk_size != DEFAULT_CHUNK_SIZE:
        parts.append(f"c{chunk_size}")
    if overlap > 0:
        parts.append("window")
    return ("_" + "_".join(parts)) if parts else ""


def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[dict]:
    """
    overlap=0 - фиксированный чанк (baseline)
    overlap>0 - sliding window: каждый следующий чанк сдвигается на (chunk_size - overlap) токенов
    """
    tokens = text.split()
    chunks = []
    step   = chunk_size - overlap
    if step <= 0:
        raise ValueError(f"overlap ({overlap}) >= chunk_size ({chunk_size}). "
                         f"Должно быть overlap < chunk_size.")

    for start in range(0, len(tokens), step):
        end   = min(start + chunk_size, len(tokens))
        chunk = tokens[start:end]
        if len(chunk) < 20:
            break
        chunks.append({"text": " ".join(chunk), "start_token": start, "end_token": end})
        if end == len(tokens):
            break
    return chunks


def process_corpus(corpus_dir, chunk_size, overlap):
    all_chunks = []
    files = sorted(f for f in os.listdir(corpus_dir) if f.endswith(".txt"))
    if not files:
        print(f"Папка '{corpus_dir}' пустая или не найдена")
        return []
    print(f"Найдено файлов: {len(files)}")
    for filename in files:
        doc_id = filename.replace(".txt", "")
        with open(os.path.join(corpus_dir, filename), encoding="utf-8") as f:
            text = clean_text(f.read())
        if len(text.split()) < 50:
            print(f"{filename} слишком короткий — пропускаем")
            continue
        raw = chunk_text(text, chunk_size, overlap)
        for i, ch in enumerate(raw):
            all_chunks.append({
                "chunk_id":    f"{doc_id}_chunk_{i:03d}",
                "doc_id":      doc_id,
                "text":        ch["text"],
                "start_token": ch["start_token"],
                "end_token":   ch["end_token"],
            })
        print(f"{filename:<50} → {len(raw)} чанков")
    return all_chunks


def save_chunks(chunks, output_file):
    d = os.path.dirname(output_file)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    print(f"\nСохранено {len(chunks)} чанков → {output_file}")


def load_chunks(filepath):
    with open(filepath, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def attach_chunk_ids(pairs_file, chunks, output_file):
    """Привязывает фрагменты из тестовых пар к chunk_id через поиск подстроки."""
    with open(pairs_file, encoding="utf-8") as f:
        pairs = [json.loads(l) for l in f]

    by_doc = {}
    for ch in chunks:
        by_doc.setdefault(ch["doc_id"], []).append(ch)

    qrels, not_found = [], []
    for pair in pairs:
        doc_id = pair.get("doc_id") or pair.get("file_name")
        fragment = pair.get("fragment", "")
        query_id = pair.get("query_id", "")
        words = fragment.split()
        mid = len(words) // 2
        sigs = [" ".join(words[:20]), " ".join(words[mid:mid+20]), " ".join(words[-20:])]

        found = set()
        for ch in by_doc.get(doc_id, []):
            first_match = len(sigs[0]) > 20 and sigs[0] in ch["text"]
            last_match = len(sigs[2]) > 20 and sigs[2] in ch["text"]
            match_count = sum(1 for sig in sigs if len(sig) > 20 and sig in ch["text"])
            if first_match or last_match or match_count >= 2:
                found.add(ch["chunk_id"])

        if found:
            for cid in found:
                qrels.append({"query_id": query_id, "query": pair.get("query",""),
                               "chunk_id": cid, "doc_id": doc_id, "relevance": 1})
        else:
            not_found.append(query_id)

    d = os.path.dirname(output_file)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for q in qrels:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"  Найдено: {len(qrels)}  Не найдено: {len(not_found)}")
    if not_found:
        print(f"  Не привязаны query_id: {not_found[:5]}{'...' if len(not_found)>5 else ''}")
    print(f"  - {output_file}")
    return qrels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlap",      action="store_true",
                        help="Включить sliding window")
    parser.add_argument("--chunk-size",   type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"Размер чанка в словах (default={DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--overlap-size", type=int, default=None,
                        help="Размер overlap в словах (default: 25%% от chunk-size)")
    args = parser.parse_args()

    # Определяем overlap
    if args.overlap:
        if args.overlap_size is not None:
            overlap = args.overlap_size
        else:
            overlap = args.chunk_size // 4
    else:
        overlap = 0

    suffix = make_suffix(args.chunk_size, overlap)

    mode = (f"sliding window chunk={args.chunk_size}, overlap={overlap}"
            if overlap else f"fixed {args.chunk_size} слов")
    print("=" * 60)
    print(f"ШАГ 1: ЧАНКИНГ [{mode}]")
    print(f"  Суффикс файлов: '{suffix}' (пусто = базовый)")
    print("=" * 60)

    chunks = process_corpus(CORPUS_DIR, args.chunk_size, overlap)
    if not chunks:
        exit(1)

    out = os.path.join(BASE_DIR, "data", f"chunks{suffix}.jsonl")
    save_chunks(chunks, out)

    total = sum(ch["end_token"] - ch["start_token"] for ch in chunks)
    print(f"\n  Документов: {len(set(ch['doc_id'] for ch in chunks))}")
    print(f"  Чанков:     {len(chunks)}")
    print(f"  Ср. размер: {total/len(chunks):.0f} токенов")

    print("\nПривязка тестовых пар:")
    for pf, qf in [
        (f"data/test_pairs.jsonl",     f"data/qrels{suffix}.jsonl"),
        (f"data/test_pairs_sud.jsonl", f"data/qrels_sud{suffix}.jsonl"),
    ]:
        pf = os.path.join(BASE_DIR, pf)
        qf = os.path.join(BASE_DIR, qf)
        if os.path.exists(pf):
            print(f"\n  {pf}")
            attach_chunk_ids(pf, chunks, qf)

    print(f"\nГотово!")