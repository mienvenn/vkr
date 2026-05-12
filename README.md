# Тема дипломной работы: Анализ и сравнение алгоритмов извлечения релевантных юридических документов для RAG-систем
Проект реализует и сравнивает 19 конфигураций методов retrieval на корпусе документов трудового права (ТК РФ + судебные решения). Поддерживаются TF-IDF, BM25, SBERT, BGE-M3, гибридный поиск (RRF) и HyDE.

---

## Структура проекта

```
vkr/
├── corpus/                  # Исходные тексты (.txt)
├── data/                    # Тестовые пары и разметка 
│   ├── test_pairs.jsonl     # запрос - релевантный фрагмент (общий корпус)
│   └── test_pairs_sud.jsonl # то же для судебных решений
│
├── 01_chunking.py           # препроцессинг и нарезка чанков
├── 02_indexing.py           # построение индексов (TF-IDF / BM25 / SBERT / BGE)
├── search_03.py             # базовый Retriever — единый интерфейс поиска
├── hybrid_04.py             # гибридный поиск (RRF) и HyDE
├── 05_evaluate.ipynb        # инференс и оценка всех конфигураций
│
├── requirements.txt
└── .gitignore
```

---

## Быстрый старт

### Клонирование репозитория и создание окружения

```bash
git clone <url>
cd "vkr"

python -m venv venv
source venv/bin/activate        

pip install -r requirements.txt
```


### Запуск пайплайна

```bash
# Нарезка чанков 
python 01_chunking.py                          # 512 слов, fixed
python 01_chunking.py --overlap                # 512 слов, sliding window
python 01_chunking.py --chunk-size 256         # 256 слов, fixed
python 01_chunking.py --chunk-size 256 --overlap  # 256 слов, sliding window

# Построение индексов 
python 02_indexing.py                          # базовый
python 02_indexing.py --suffix window
python 02_indexing.py --suffix c256
python 02_indexing.py --suffix c256_window

# Поиск (используется из notebook или как библиотека)

# Оценка
jupyter notebook 05_evaluate.ipynb
```

---

## Методы поиска

| Label | Метод | Чанкинг |
|---|---|---|
| TF-IDF | `tfidf` | 512, fixed |
| BM25 | `bm25` | 512, fixed |
| SBERT | `sbert` | 512, fixed |
| BGE-M3 | `bge` | 512, fixed |
| TF-IDF+Window | `tfidf` | 512, overlap=128 |
| BM25+Window | `bm25` | 512, overlap=128 |
| SBERT+Window | `sbert` | 512, overlap=128 |
| BGE+Window | `bge` | 512, overlap=128 |
| Hybrid SBERT | `hybrid_sbert` — BM25 + SBERT → RRF (k=60) | 512, fixed |
| Hybrid BGE | `hybrid_bge` — BM25 + BGE → RRF (k=60) | 512, fixed |
| Hybrid TF-IDF+BGE | `hybrid_tfidf_bge` — TF-IDF + BGE → RRF (k=60) | 512, fixed |
| Hybrid TF-IDF+BGE k30 | `hybrid_tfidf_bge_k30` — TF-IDF + BGE → RRF (k=30) | 512, fixed |
| Hybrid TF-IDF+BGE w | `hybrid_tfidf_bge_weighted` — TF-IDF + BGE → взвешенное (0.3/0.7) | 512, fixed |
| HyDE+SBERT | `hyde_sbert` — LLM + SBERT | 512, fixed |
| HyDE+BGE | `hyde_bge` — LLM + BGE | 512, fixed |
| TF-IDF c256 | `tfidf` | 256, fixed |
| BM25 c256 | `bm25` | 256, fixed |
| BGE-M3 c256 | `bge` | 256, fixed |
| Hybrid TF-IDF+BGE c256 | `hybrid_tfidf_bge` — TF-IDF + BGE → RRF (k=60) | 256, fixed |

Итого **19 конфигураций** в ноутбуке.
---

## Метрики оценки

Notebook `05_evaluate.ipynb` вычисляет для каждой конфигурации:

- **Recall@K** — доля запросов, где релевантный чанк нашёлся в топ-K
- **Precision@K** — точность в топ-K
- **MRR** (Mean Reciprocal Rank) — средний обратный ранг первого попадания
- **NDCG@K** — нормализованный дисконтированный кумулятивный выигрыш

Также в файле находится пример работы системы.

---

## Конфигурация окружения

Необходимо создать файл `.env`:

```env
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://openai.api.proxyapi.ru/v1
```

Ключ используется только для методов `hyde_*`.

---

## Требования

- Python 3.10+
- зависимости из requirements.txt

---

## Используемые модели

| Модель | Источник | Размер |
|---|---|---|
| `paraphrase-multilingual-mpnet-base-v2` | sentence-transformers | ~1.1 ГБ |
| `BAAI/bge-m3` | HuggingFace | ~2.3 ГБ |
