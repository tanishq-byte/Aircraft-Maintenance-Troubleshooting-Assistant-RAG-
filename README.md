# Aircraft Maintenance Troubleshooting Assistant (RAG)

A citation-grounded retrieval-augmented generation (RAG) system that helps engineers search aviation maintenance literature (defect reports, airworthiness directives, safety reports) using natural-language symptom descriptions, and returns answers traceable to their source documents.

> **Safety notice:** This is a portfolio/research project built on public data. It is **not** a certified maintenance tool. Never use its output to make real maintenance or airworthiness decisions. Always verify against the official manufacturer manuals and regulator documents.

---

## Table of Contents

1. [Problem](#problem)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Tech Stack](#tech-stack)
5. [Data Sources](#data-sources)
6. [Repository Structure](#repository-structure)
7. [Setup](#setup)
8. [Usage](#usage)
9. [Evaluation](#evaluation)
10. [Guardrails](#guardrails)
11. [Monitoring](#monitoring)
12. [Docker](#docker)
13. [Limitations](#limitations)
14. [Roadmap](#roadmap)
15. [Acknowledgements and Licenses](#acknowledgements-and-licenses)

---

## Problem

When an aircraft has a fault, engineers must search thousands of pages of manuals, service bulletins, airworthiness directives and past defect reports for likely causes and approved fixes. Keyword search misses relevant results because the same issue is described in different words across documents. Slow or wrong answers keep aircraft grounded, and a hallucinated answer in this domain is a safety risk, not just a poor user experience.

This project explores how far a carefully engineered RAG pipeline can go: hybrid retrieval, version-aware filtering, reranking, citation-grounded generation, refusal when evidence is weak, and measured evaluation of every upgrade.

## Key Features

- **Structure-aware ingestion:** parsing and chunking by document section, with metadata (aircraft type, ATA chapter, date, document version, source).
- **Hybrid retrieval:** BM25 (exact match for part numbers and codes) combined with dense embeddings (semantic match for symptoms) using reciprocal rank fusion.
- **Metadata filtering:** restrict results by aircraft type and document version so superseded or wrong-aircraft content is not retrieved.
- **Cross-encoder reranking:** improves precision of the final top results.
- **Citation-grounded answers:** every claim links to its document, section and page.
- **Low-confidence refusal:** the system says "not found" and recommends checking the official manual instead of guessing.
- **Experiment-driven evaluation:** a 100+ question test set and a results table comparing each pipeline upgrade.
- **Production basics:** containerized FastAPI service with tracing, latency and cost monitoring.

## Architecture

```
                 +---------------------+
 Public docs --> |  Ingestion pipeline | --> parse -> clean -> chunk -> tag metadata
 (PDF/HTML/CSV)  +---------------------+
                            |
              +-------------+--------------+
              |                            |
       Dense embeddings              BM25 index
       (vector DB)                  (keyword)
              |                            |
              +-------------+--------------+
                            |
User query -> Metadata filter -> Hybrid retrieval (RRF) -> Reranker
                                                            |
                                                  Confidence check
                                                   |            |
                                              (low) refuse   (ok) LLM generation
                                                                  |
                                                    Answer + citations + sources
                                                                  |
                                              FastAPI  ->  Tracing / metrics
```

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Orchestration | LlamaIndex (LangChain comparison optional) |
| Vector DB | Qdrant (Chroma/FAISS as baselines) |
| Keyword search | BM25 (e.g. `rank_bm25`) |
| Reranker | Cross-encoder (e.g. `bge-reranker`) |
| LLM | Configurable via environment variables |
| Evaluation | Ragas, custom retrieval metrics (hit rate, MRR) |
| Serving | FastAPI, Uvicorn |
| UI (optional) | Streamlit |
| Monitoring | Langfuse tracing, latency/cost logging |
| Packaging | Docker, Docker Compose |

## Data Sources

All data is public. **Check each source's license and terms before downloading or redistributing.** Raw data is not committed to this repository.

| Source | Content | Use |
|---|---|---|
| FAA Service Difficulty Reports (SDR) | Free-text defect reports | Core symptom-based retrieval corpus |
| FAA Airworthiness Directives | Mandatory fix notices | Versioned procedures, filtering tests |
| NASA ASRS | Aviation safety reports | Additional free-text incidents |
| NTSB reports | Accident/incident reports | Long-document ingestion practice |

Download instructions and the exact files used are documented in `data/README.md`.

## Repository Structure

> Adjust this to match your actual code.

```
.
├── README.md
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── data/
│   ├── README.md            # where/how to download each source
│   ├── raw/                 # not committed
│   └── processed/           # chunked documents with metadata
├── src/
│   ├── ingestion/           # parsing, cleaning, chunking, metadata tagging
│   ├── indexing/            # vector index + BM25 index builders
│   ├── retrieval/           # filters, hybrid search, RRF, reranker
│   ├── generation/          # prompts, citation formatting, guardrails
│   ├── api/                 # FastAPI app
│   └── ui/                  # Streamlit app (optional)
├── eval/
│   ├── questions.jsonl      # evaluation set
│   ├── run_eval.py          # runs all pipeline variants
│   └── results/             # saved metrics and tables
├── notebooks/               # EDA and experiments
└── tests/
```

## Setup

**Prerequisites:** Python 3.10+, Docker (for Qdrant and deployment), and an API key or local model for the LLM.

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # then edit values
```

Example `.env`:

```
LLM_PROVIDER=<provider>
LLM_MODEL=<model-name>
LLM_API_KEY=<your-key>
EMBEDDING_MODEL=<embedding-model>
RERANKER_MODEL=<reranker-model>
QDRANT_URL=http://localhost:6333
LANGFUSE_PUBLIC_KEY=<optional>
LANGFUSE_SECRET_KEY=<optional>
CONFIDENCE_THRESHOLD=<tuned-value>
```

Start the vector database:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

## Usage

### 1. Ingest and index

```bash
python -m src.ingestion.run --input data/raw --output data/processed
python -m src.indexing.build --input data/processed
```

### 2. Run the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

### 3. Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
        "question": "Hydraulic pressure fluctuation during landing gear retraction",
        "aircraft_type": "<optional filter>",
        "top_k": 5
      }'
```

Example response shape:

```json
{
  "answer": "...",
  "citations": [
    {"source": "<document>", "section": "<section>", "page": 0, "score": 0.0}
  ],
  "confidence": 0.0,
  "refused": false,
  "latency_ms": 0
}
```

### 4. Optional UI

```bash
streamlit run src/ui/app.py
```

## Evaluation

### Evaluation set

`eval/questions.jsonl` contains 100+ questions, each with the expected source document(s) and a reference answer. Questions cover symptom descriptions, part-number lookups, version-sensitive queries, and **unanswerable** questions to test refusal behavior.

```json
{"id": "q001", "question": "...", "expected_sources": ["..."], "reference_answer": "...", "type": "symptom|part_number|version|unanswerable"}
```

### Metrics

- **Retrieval:** hit rate@k, MRR
- **Generation:** faithfulness and answer correctness (Ragas)
- **Guardrails:** refusal precision/recall on unanswerable questions
- **System:** p50/p95 latency, cost per query

### Reproduce

```bash
python eval/run_eval.py --config all
```

### Results

> Fill this table only with numbers you actually measured by running `eval/run_eval.py`. Do not publish placeholder values.

| Pipeline variant | Hit rate@5 | MRR | Faithfulness | Answer correctness | p95 latency |
|---|---|---|---|---|---|
| Naive dense retrieval | TBD | TBD | TBD | TBD | TBD |
| + Metadata filters | TBD | TBD | TBD | TBD | TBD |
| + Hybrid (BM25 + dense, RRF) | TBD | TBD | TBD | TBD | TBD |
| + Cross-encoder reranker | TBD | TBD | TBD | TBD | TBD |
| + Guardrails (refusal) | TBD | TBD | TBD | TBD | TBD |

Short analysis of what helped, what did not, and the failure cases belongs here.

## Guardrails

- The LLM is instructed to answer **only** from retrieved context.
- If the reranker's top score falls below `CONFIDENCE_THRESHOLD`, the system refuses and recommends consulting the official manual.
- Every response includes citations; answers without supporting sources are rejected.
- Document-version and aircraft-type filters prevent retrieval of superseded or non-applicable procedures.

## Monitoring

- Each request is traced (retrieval results, reranker scores, prompt, response) in Langfuse.
- Latency and token usage are logged per request to compute cost per query.
- The API exposes `/health` and a feedback endpoint so users can mark answers helpful or incorrect.

## Docker

```bash
docker compose up --build
```

This starts the API, the vector database and (optionally) the UI. See `docker-compose.yml` for service configuration.

## Limitations

- Built only on public data; coverage of real manufacturer manuals is limited.
- Table and figure extraction from PDFs is imperfect and can affect part-number lookups.
- Evaluation questions are partly synthetic and may not reflect real engineer queries.
- Answer quality depends on the chosen LLM and the tuned confidence threshold.
- Not certified or validated for operational use.

## Roadmap

- [ ] Embedding fine-tuning on domain question/passage pairs
- [ ] QLoRA fine-tuning and 4-bit quantization benchmark
- [ ] Custom lightweight orchestration layer (tool routing, memory, tracing) compared against LangChain
- [ ] Better table extraction and part-number normalization
- [ ] Larger human-reviewed evaluation set

## Acknowledgements and Licenses

- Data: FAA, NASA ASRS, NTSB (check each source's terms of use).
- Libraries: LlamaIndex, Qdrant, Ragas, Langfuse, FastAPI and others listed in `requirements.txt`.
- Project license: add your chosen license (for example MIT) in `LICENSE`.

---

**Author:** Tanishq Khetwal | [GitHub](https://github.com/tanishq-byte) | [LinkedIn](https://www.linkedin.com/in/tanishq-khetwal-8563a2330)
