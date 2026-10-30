# OTTER: Verifiable Multihop Reasoning over Educational Ontology for Vietnamese Education

OTTER is the research code behind the paper **"OTTER: Verifiable Multihop Reasoning over Educational Ontology for Vietnamese Education"** (under review). It contains

* **the framework**: a builder that turns scanned textbooks into *source spans* and a span-grounded **Educational Ontology** (concepts, relations, facts, rules, learning objectives, each linked to the spans that support it), and a question-answering pipeline that answers with a **source pointer** (book, lesson, page, span) or abstains;
* **the evaluation code** for **ViEduHop**, a 3,015-row benchmark with gold source spans (the data live on Hugging Face, not here);


**Project page:** https://quin210.github.io/OTTER/ (demo video, ontology explorer, overview).


## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m pytest -q          # unit tests on synthetic fixtures; no GPU, no data needed
```

Models are served locally through OpenAI-compatible vLLM endpoints (`configs/default.yaml`; launch scripts in `scripts/serve/`): Qwen3-32B-FP8 (main), Qwen3-8B (verifier), bge-m3 (embeddings). No external LLM API is used.

## 1. Benchmark Evaluation on ViEduHop

```bash
pip install datasets
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("quin210/vieduhop", "all", split="test")
PY
# write predictions.jsonl with {"id": ..., "prediction": "A"|"B"|"C"|"D"|"E"|text, "source_span_ids": [...]}
python scripts/evaluate.py --data data/all.jsonl --pred predictions.jsonl   # scorer shipped with the dataset
```

The pre-registered headline is on the 1,208 `headline` items: accuracy, located, lesson-located, verified-correct and cost. Everything else is additional (see the dataset card).

## 2. Applying OTTER to a Custom Corpus

```bash
# 1) serve the models (adapt GPUs/ports) 
bash scripts/serve/qwen3_32b.sh; bash scripts/serve/qwen3_8b.sh; bash scripts/serve/bge_m3.sh
# 2) turn page transcripts into spans, then build the ontology (needs your own books)
python scripts/ocr/build_spans.py --help
python -m otter.builder build --help
# 3) answer a benchmark with OTTER and score it (OTTER is registered under the system name `otter_v2`; the name `otter` is an alias of the earlier multi-agent system)
python scripts/eval/run.py --system otter_v2 --backbone main --bench <benchmark.jsonl> --exp my_run --workers 6
python scripts/eval/headline_analysis.py --help
```

OTTER is implemented in `otter/agents/v2.py` with `configs/otter_v2.yaml`: Composer, Router, Retriever (hybrid BM25 + bge-m3 with reciprocal-rank fusion, ontology hints and candidate spans), Judge (per-option *entailed / refuted / unsupported* with verbatim quotes), Verifier (deterministic quote-in-span check plus a smaller model) and a grounding level 0-3; roughly 2.5 LLM calls per question. **Source spans are the evidence of record; the ontology is only a navigation index** and is never cited as proof.

## 3. Ontology Visualization

`viz/ontology_explorer.html` is a self-contained page (open it in a browser, no server) that shows sample lesson graphs of the Educational Ontology: every node links to its source pointer (book, lesson heading, printed page, span id), and no textbook passage is shown. `media/otter_demo.mp4` (68 s, captions only, no audio; preview `media/otter_demo_preview.gif`) walks through the pipeline on one real question and lesson; every fact shown is checked against run files in `media/otter_demo_verification.json`. See `viz/README.md` for how they were generated and what they omit.

## Repository Structure

```
otter/            framework: agents (OTTER and the earlier multi-agent system it delegates calculations to), builder, retrieval, reasoning, tools, storage, evaluation
configs/          endpoints and OTTER configuration (default.yaml, otter_v2.yaml)
scripts/serve/    vLLM launch scripts            scripts/ocr/   page transcription and span building
scripts/eval/     run.py (resumable runner + scorer) and analysis/report scripts
scripts/para/ paraphrase tier   scripts/maintenance/ ontology coverage and maintenance studies
tests/            228 unit tests on synthetic fixtures
```

## Reproducibility

Frozen artefacts: `otter/agents/v2.py` (md5 `8e5e222c...`), `configs/otter_v2.yaml` (the files behind OTTER), the ontology database and the benchmark (hashes in the dataset repository). Running OTTER needs the textbooks and the ontology (not distributed), four GPUs for Qwen3-32B-FP8 and several hours. The comparison systems of the paper (RAG variants, IRCoT, tool agents and the graph-RAG systems GraphRAG, LightRAG, MiniRAG and HippoRAG 2) are not included. This release contains **one version of OTTER, the one used in the paper**.

## Limitations and Responsible Use

* ViEduHop items are LLM-generated or LLM-written and automatically checked; teacher review is in progress at the time of writing and no item should be called teacher-validated yet.
* The ontology is noisy and incomplete; hints from it can mislead, which is why OTTER never counts them as evidence.
* The benchmark measures retrieval, provenance and abstention on one Vietnamese curriculum; it must not be used to grade students or schools.


## License

The code is released under the **Apache License 2.0** (see `LICENSE`). The ViEduHop dataset has its own licence (CC BY-NC-SA 4.0) on Hugging Face. The textbooks are **not** included and remain the property of their publisher.
