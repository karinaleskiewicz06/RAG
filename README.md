# Hybrid RAG over 5,000+ Documents

A Retrieval-Augmented Generation system that answers questions over a corpus of **5,189 internal knowledge-base documents** (Confluence-style runbooks, playbooks, ADRs and specs). Built as part of the **WUST Mentoring Programme** (Wrocław University of Science and Technology), spring 2026.

The project has two sides: the **mathematics of vector search** (k-NN, IVF, Product Quantization, ADC, HNSW), presented in the slides in `docs/`, and a **working pipeline** that puts that theory into practice with FAISS.

## How it works

```
question
   │
   ├─► Vector ANN search (Jina v4 → FAISS IVF-HNSW-PQ) ─┐
   │                                                    ├─► Reciprocal Rank Fusion
   └─► BM25 lexical search ─────────────────────────────┘          │
                                                                   ▼
                                         Cross-encoder rerank (Jina Reranker v2)
                                                                   │
                                         Pick top source doc(s) + expand their chunks
                                                                   │
                                         Second rerank → top 10 chunks
                                                                   │
                                                                   ▼
                                         LLM answer via OpenRouter (with model fallback)
```

**Indexing (offline)**
- Documents split into ~800-character chunks with `chonkie`'s `RecursiveChunker`
- Embedded with **`jina-embeddings-v4`** (3.8B parameters, 512-dim Matryoshka vectors), with resumable checkpointing so the multi-hour embedding job can run on a Colab GPU
- Indexed in **FAISS** as `IVF + HNSW coarse quantizer + PQ`:
  - **IVF**: 256 Voronoi cells, 16 probed per query
  - **HNSW** (M = 32) routes queries to the right cells
  - **PQ** (64 sub-quantizers × 8 bits) compresses each vector to 64 bytes, searched with **ADC** lookup tables

**Retrieval (online)**
1. Vector search and BM25 each return 25 candidates
2. Results merged with **Reciprocal Rank Fusion** (k = 60)
3. **Cross-encoder reranking** keeps the 10 best
4. Source-aware expansion: the strongest document (plus a second one if scores are close) is expanded to all of its chunks, with a filename-match boost
5. Second rerank selects the final 10 chunks for the LLM

**Generation**: answers come from free LLMs on OpenRouter (DeepSeek, GPT-OSS, Llama, Qwen, …), with automatic fallback to the next model when one is unavailable.

## Evaluation

Measured with `evaluate.py` on **64 questions** with gold answers and expected source documents (basic, semantic, constrained, completeness and multi-step questions).

| Metric | Value | Meaning |
|--------|-------|---------|
| Doc Hit@1 | 0.656 | Correct document ranked first |
| Doc Hit@k | 0.719 | Correct document among the selected sources |
| Doc in LLM context | 0.719 | Correct document actually passed to the LLM |
| Chunk MRR | 0.703 | Mean Reciprocal Rank of the first relevant chunk |
| Fact Recall | 0.615 | Share of expected facts present in the generated answer |
| Retrieval latency | 1.84 s | Average per query, including both reranking passes |

The script also supports LLM-as-judge metrics (contextual recall/precision/relevancy, faithfulness, answer correctness).

## Quick start

```bash
pip install -r requirements.txt
echo "OPENROUTER_API_KEY=sk-or-..." > .env

python pipline.py build                    # embed documents and build the index
python pipline.py ask "How long is the runbook author certification valid?"
python evaluate.py                         # run the evaluation
```

Embedding 5,000+ documents with a 3.8B model is slow on a laptop. The split workflow runs the heavy step on a GPU:

```bash
python pipline.py embed-export             # on a GPU (see colab_jina_build.ipynb)
python pipline.py build-index              # locally, ~15 min
```

## Repository structure

| Path | Contents |
|------|----------|
| `pipline.py` | Full pipeline: chunking, embedding, FAISS index, hybrid retrieval, reranking, generation, CLI |
| `evaluate.py` | Retrieval metrics and LLM-judged RAG metrics |
| `build_faiss_only.py` | Index build without PyTorch (works around a FAISS + PyTorch segfault on macOS) |
| `colab_jina_build.ipynb` | Resumable GPU embedding job for Google Colab |
| `eval_dataset.jsonl`, `questions.jsonl` | Evaluation questions with gold answers |
| `confluence/`, `confluence 2/` | Document corpus |
| main brunch | Slides: k-NN in vector databases, and the mathematics of vector search (PQ, ADC, IVF, HNSW) |

## Author

**Karina Leśkiewicz**, Applied Mathematics and AI, Wrocław University of Science and Technology
[LinkedIn](https://www.linkedin.com/in/karina-le%C5%9Bkiewicz-5081502a7/)
