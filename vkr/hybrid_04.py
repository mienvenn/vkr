"""
Гибридный поиск и HyDE
================================
Запуск:
  python 04_hybrid.py                  # базовые чанки
  python 04_hybrid.py --window         # sliding window чанки
  python 04_hybrid.py --methods hybrid_sbert hybrid_bge hybrid_tfidf_bge hyde_sbert hyde_bge

Методы:
  hybrid_sbert      — BM25 + SBERT     - RRF слияние
  hybrid_bge        — BM25 + BGE       - RRF слияние
  hybrid_tfidf_bge  — TF-IDF + BGE     - RRF слияние (TF-IDF стабильно лучше BM25)
  hyde_sbert        — HyDE + SBERT (LLM генерирует гипотетический ответ)
  hyde_bge          — HyDE + BGE

RRF формула: score = Σ 1/(k + rank_i)
  k=60 — классический стандарт 
  k=30 — агрессивнее ранжирует топ, рекомендуется при небольших корпусах
HyDE: запрос - LLM - гипотетический ответ - embed - поиск
"""

import os, json, time, argparse
import numpy as np
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# RRF слияние

def reciprocal_rank_fusion(results_list: list[list[dict]], k: int = 60) -> list[dict]:
    """
    Принимает несколько списков результатов (каждый от своего метода).
    Возвращает объединённый список по формуле RRF.
    """
    scores: dict[str, float] = {}

    for results in results_list:
        for rank, item in enumerate(results, start=1):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)

    merged = [{"chunk_id": cid, "score": sc}
              for cid, sc in sorted(scores.items(), key=lambda x: -x[1])]
    return merged


def weighted_score_fusion(results_list: list[list[dict]],
                          weights: list[float] | None = None) -> list[dict]:
    """
    Альтернатива RRF: взвешенное суммирование нормализованных скоров.
    weights — список весов для каждого метода (должны суммироваться в 1.0)
    """
    if weights is None:
        weights = [1.0 / len(results_list)] * len(results_list)
    if len(weights) != len(results_list):
        raise ValueError("len(weights) != len(results_list)")

    scores: dict[str, float] = {}

    for results, w in zip(results_list, weights):
        if not results:
            continue
        max_sc = max(r["score"] for r in results) or 1.0
        for item in results:
            cid = item["chunk_id"]
            normalized = item["score"] / max_sc
            scores[cid] = scores.get(cid, 0.0) + w * normalized

    merged = [{"chunk_id": cid, "score": sc}
              for cid, sc in sorted(scores.items(), key=lambda x: -x[1])]
    return merged


# HyDE

def generate_hypothesis(query: str, llm_fn) -> str:
    """
    Генерирует гипотетический ответ на запрос через LLM (HyDE).
    llm_fn — функция (query: str) - str
    """
    prompt = (
        "Ты эксперт по трудовому праву Российской Федерации. "
        "Напиши короткий отрывок в стиле судебного решения или нормы ТК РФ, "
        "который напрямую отвечает на следующий вопрос.\n"
        "Требования к тексту:\n"
        "- используй юридическую лексику и канцелярский стиль (как в судебных актах)\n"
        "- ссылайся на конкретные статьи ТК РФ или других НПА\n"
        "- упоминай стороны: работник, работодатель, суд\n"
        "- объём: 3–5 предложений, не более\n\n"
        f"Вопрос: {query}\n\n"
        "Отрывок:"
    )
    return llm_fn(prompt)

def make_openai_llm(model: str = "openai/gpt-4o"):
    """Фабрика LLM-функции для OpenAI."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_KEY, base_url=OPENAI_BASE)
    def llm_fn(prompt):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    return llm_fn

# Hybrid Retriever 

class HybridRetriever:
    """
    Обёртка над базовым Retriever с добавлением гибридных методов и HyDE.
    """

    def __init__(self, window=False, suffix=None, llm_fn=None):
        """
        suffix  — суффикс папки индексов (без "_"). Примеры: "window", "c256", "c256_window".
                  Если задан, имеет приоритет над window.
        window  — для обратной совместимости (эквивалентно suffix="window").
        llm_fn  — функция генерации гипотезы для HyDE. Если None — заглушка.
        """
        import sys
        sys.path.insert(0, BASE_DIR)
        from search_03 import Retriever

        print("Загружаем базовые индексы...")
        self.base = Retriever(methods=["tfidf","bm25","sbert","bge"],
                              window=window, suffix=suffix)
        self.llm_fn = llm_fn 

    def search(self, query: str, method: str, k: int = 10,
               rrf_k: int = 60) -> list[dict]:
        """
        Поддерживает все базовые методы + гибридные:
          tfidf, bm25, sbert, bge
          hybrid_sbert      — BM25  + SBERT - RRF
          hybrid_bge        — BM25  + BGE   - RRF
          hybrid_tfidf_bge  — TF-IDF + BGE  - RRF  
          hybrid_tfidf_bge_k30 — то же, но rrf_k=30 
          hybrid_tfidf_bge_weighted — взвешенное слияние (α=0.3 TF-IDF + 0.7 BGE)
          hyde_sbert        — HyDE + SBERT
          hyde_bge          — HyDE + BGE
        """
        if method in ("tfidf", "bm25", "sbert", "bge"):
            return self.base.search(query, method, k)

        elif method == "hybrid_sbert":
            bm25_res = self.base.search(query, "bm25",  k=k)
            sbert_res = self.base.search(query, "sbert", k=k)
            return reciprocal_rank_fusion([bm25_res, sbert_res], k=rrf_k)[:k]

        elif method == "hybrid_bge":
            bm25_res = self.base.search(query, "bm25", k=k)
            bge_res = self.base.search(query, "bge",  k=k)
            return reciprocal_rank_fusion([bm25_res, bge_res], k=rrf_k)[:k]

        elif method == "hybrid_tfidf_bge":
            tfidf_res = self.base.search(query, "tfidf", k=k)
            bge_res = self.base.search(query, "bge",   k=k)
            return reciprocal_rank_fusion([tfidf_res, bge_res], k=rrf_k)[:k]

        elif method == "hybrid_tfidf_bge_k30":
            tfidf_res = self.base.search(query, "tfidf", k=k)
            bge_res = self.base.search(query, "bge",   k=k)
            return reciprocal_rank_fusion([tfidf_res, bge_res], k=30)[:k]

        elif method == "hybrid_tfidf_bge_weighted":
            tfidf_res = self.base.search(query, "tfidf", k=k)
            bge_res = self.base.search(query, "bge",   k=k)
            return weighted_score_fusion([tfidf_res, bge_res], weights=[0.3, 0.7])[:k]

        elif method == "hyde_sbert":
            hypothesis = generate_hypothesis(query, self.llm_fn)
            return self.base.search(hypothesis, "sbert", k=k)

        elif method == "hyde_bge":
            hypothesis = generate_hypothesis(query, self.llm_fn)
            return self.base.search(hypothesis, "bge", k=k)

        else:
            raise ValueError(f"Неизвестный метод: {method}")

    def search_timed(self, query: str, method: str, k: int = 10):
        t0 = time.perf_counter()
        results = self.search(query, method, k)
        return results, (time.perf_counter() - t0) * 1000

    def get_chunk_text(self, chunk_id: str) -> str:
        return self.base.get_chunk_text(chunk_id)


# main 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", action="store_true")
    parser.add_argument("--methods", nargs="+",
                        default=["hybrid_sbert","hybrid_bge","hyde_sbert","hyde_bge"])
 
    args = parser.parse_args()

    llm = make_openai_llm() 

    print("=" * 60)
    print(f"ШАГ 4: ГИБРИДНЫЙ ПОИСК И HyDE")
    print("=" * 60)

    retriever = HybridRetriever(window=args.window, llm_fn=llm)

    demo_query = "могут ли уволить пока я на больничном?"
    print(f"\nДемо-запрос: «{demo_query}»\n")

    for method in args.methods:
        print(f"─── {method.upper()} ───")
        results, ms = retriever.search_timed(demo_query, method, k=5)
        for i, r in enumerate(results[:3]):
            text = retriever.get_chunk_text(r["chunk_id"])
            print(f"  [{i+1}] {r['chunk_id']}  score={r['score']:.4f}")
            print(f"       {text[:150]}...")
        print(f"Время {ms:.1f} мс\n")