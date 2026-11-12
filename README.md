# Retrieval-Augmented Q&A — experimental prototype

This is an earlier document-question-answering experiment using SentenceTransformers, FAISS, FLAN-T5 and Flask. It is **not** a production-ready retrieval system and no accuracy, latency or throughput claim has been verified for this version.

## Existing architecture

1. Load SQuAD v1.1 training contexts from `documents/train-v1.1.json`.
2. Encode contexts with `multi-qa-MiniLM-L6-cos-v1` and index them with FAISS.
3. Retrieve one context for a question.
4. Generate an answer with `google/flan-t5-large`. The API route also summarizes the retrieved passage using BART, whereas `test.py` uses the original passage.

The repository contains a Flask API in `app.py`, a model/index loader in `model.py`, and an experimental evaluation script in `test.py`.

## Reproducing the diagnostic evaluation

Use an isolated Python environment and install the packages in `requirements.txt` (plus PyTorch and other platform-specific model dependencies as needed). Supply the original SQuAD v1.1 train/dev JSON files under `documents/`. Model weights are downloaded by Hugging Face on initial use and require substantial memory.

First run the deterministic, offline metric tests:

```bash
python -m unittest test_metrics.py
```

Then, once the ML dependencies, models and SQuAD files are available:

```bash
python test.py
```

The diagnostic script processes at most 50 development-set questions and reports normalized exact match and token-overlap F1. The denominator now counts only questions actually evaluated. This **does not** constitute an official SQuAD evaluation: the retrieval index is built from **training** contexts, while the questions come from the **development** set. Scores should not be compared to published SQuAD benchmarks. We have not run the full model experiment or validated a score as part of this repository audit.

## Known limitations

- The index loads all training contexts and embeds them on startup; indexing is not cached.
- Retrieval uses `IndexFlatL2` without documenting a metric/normalization comparison.
- Different routes use different retrieved contexts (summarized vs full).
- The API currently launches Flask in debug mode and lacks authentication and request validation.
- The implementation has no established deployment or load-testing protocol.
- Small-sample metrics are exploratory and may have substantial variance.
- Model compatibility and device selection should be rechecked against current PyTorch/Hugging Face versions.

## Next improvements

- Construct a clearly specified, leakage-free retrieval test corpus and question set.
- Add random/lexical baselines, retrieval recall@k, and answer metrics; publish exact sample counts.
- Separate indexing, retrieval, generation and evaluation modules, and cache embeddings.
- Add documented CPU/MPS configurations, pinned dependency versions and a reproducible run script.
- Re-run all experiments and publish results **only after verifying them**.

This README intentionally describes the current state and its limitations rather than inventing performance results.
