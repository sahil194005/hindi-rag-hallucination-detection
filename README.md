# Hindi RAG Hallucination Detection using mmBERT

A research project to **detect and localize hallucinated text in Hindi answers** given a question and retrieved evidence passages. This repository contains the mmBERT token-classification workstream, evaluation scripts, and span-refinement experiments.

> **Scope:** This is an evidence-grounded hallucination detector, **not a complete retrieval-augmented generation (RAG) pipeline**. The model takes a question, context passages, and a candidate answer as input.

## Project overview

- **Base model:** [`jhu-clsp/mmBERT-small`](https://huggingface.co/jhu-clsp/mmBERT-small), fine-tuned for token classification.
- **Dataset:** [`alexandrainst/multi-wiki-qa-synthetic-hallucinations`](https://huggingface.co/datasets/alexandrainst/multi-wiki-qa-synthetic-hallucinations), Hindi (`hi`) configuration.
- **Task:** Label answer tokens as supported (`0`) or hallucinated (`1`), then convert predictions to character spans in the answer.
- **Input:** Hindi question, one or more evidence/context passages, and a Hindi answer.
- **Output:** Hallucination prediction and highlighted answer spans.

## Results

Held-out Hindi test set (800 examples, balanced between supported and hallucinated answers):

| Metric | Result |
|---|---:|
| Answer-level accuracy | **98.75%** |
| Answer-level hallucination F1 | **98.76%** |
| Hallucinated-token F1 | **85.05%** |
| Mean character IoU, hallucinated answers — original | **76.33%** |
| Mean character IoU, hallucinated answers — V2 refinement | **77.42%** |
| Mean character IoU, all evaluable examples — original | **87.35%** |
| Mean character IoU, all evaluable examples — V2 refinement | **87.89%** |

V2 is a **post-processing** step that expands predicted spans to appropriate Hindi word boundaries and trims whitespace. It does not change the underlying classifier's token predictions. Character-IoU metrics exclude four positive test examples without usable gold span annotations (796 evaluable examples; 396 hallucinated).

**Important:** These numbers are measured on the held-out split of the same synthetic dataset family used for training. They are not evidence of generalization to independently collected real-world Hindi RAG outputs.

## Repository layout

```text
.
├── src/
│   ├── find_dataset.py            # Dataset discovery
│   ├── inspect_hindi.py           # Inspect Hindi examples
│   ├── prepare_data.py            # Prepare and split data
│   ├── tokenize_data.py           # Tokenize inputs and align span labels
│   ├── train.py                   # Fine-tune mmBERT
│   ├── evaluate.py                # Token/answer-level evaluation
│   ├── span_evaluate.py           # Character-span evaluation
│   ├── error_analysis.py          # Inspect errors
│   ├── refine_spans.py            # Initial refinement experiment
│   ├── refine_spans_v2.py         # Hindi word-boundary refinement
│   ├── refine_spans_v3.py         # Confidence-aware experiment
│   ├── refine_spans_final.py      # Final held-out V2 evaluation
│   └── demo.py                    # Gradio demonstration
├── results/                       # Experiment CSVs
├── requirements.txt               # Python dependencies
├── requirements-lock.txt          # Local environment snapshot
└── .gitignore
```

Some auxiliary scripts may also be present. Generated datasets, virtual environments, and model checkpoints are excluded from Git by `.gitignore`.

## Setup

Tested during development on Ubuntu with an NVIDIA GPU; other environments may need different PyTorch/CUDA installation steps.

```bash
# Clone the repository (replace with the real GitHub URL)
git clone https://github.com/YOUR_USERNAME/hindi-rag-hallucination-detection.git
cd hindi-rag-hallucination-detection

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The `requirements-lock.txt` file records the original environment, but may contain platform-specific packages. Prefer `requirements.txt` for a fresh installation and verify CUDA separately if using a GPU.

## Get the trained checkpoint

The fine-tuned weights are **not included in this GitHub repository**. The expected local directory for the best checkpoint is:

```text
models/mmbert-hindi-full/best/
```

**Model download link: not published yet.** The project owner should upload the `best/` checkpoint and tokenizer/config files to a Hugging Face model repository, then add its URL here. Until then, request the checkpoint from the project owner. Do **not** assume that downloading the original `jhu-clsp/mmBERT-small` base model reproduces the fine-tuned results.

After obtaining the checkpoint, place its files inside `models/mmbert-hindi-full/best/` and check that the expected model/config/tokenizer files are present.

## Run the demo

With the fine-tuned checkpoint available at the expected path:

```bash
python src/demo.py
```

Open the local URL printed by Gradio (often `http://127.0.0.1:7860`). Enter a Hindi question, evidence/context, and an answer to inspect predicted hallucinated spans.

> The demo's exact highlighting behavior depends on the version of `src/demo.py`. The V2 improvement was separately evaluated in `src/refine_spans_final.py`; do not assume the demo applies V2 unless that code has been integrated.

## Reproduce the research workflow

The scripts are organized in the following order:

1. Inspect the Hindi dataset: `src/find_dataset.py`, `src/inspect_hindi.py`.
2. Prepare data: `src/prepare_data.py`. The project used 6,400 training, 800 validation, and 800 test examples; question+context grouping was used to reduce duplicate leakage.
3. Align answer-token labels with hallucination character spans: `src/tokenize_data.py`.
4. Train the model: `src/train.py`. Three epochs, learning rate `2e-5`, effective batch size 8 (batch 1 × gradient accumulation 8), mixed precision, gradient checkpointing, and best-checkpoint selection using validation token F1.
5. Evaluate: `src/evaluate.py` and `src/span_evaluate.py`.
6. Inspect failures: `src/error_analysis.py`.
7. Compare refinements: `src/refine_spans.py`, `src/refine_spans_v2.py`, `src/refine_spans_v3.py`.
8. Evaluate the selected V2 refinement on the held-out test split: `src/refine_spans_final.py`.

```bash
# Typical script invocation from the repository root
python src/prepare_data.py
python src/tokenize_data.py
python src/train.py
python src/evaluate.py
python src/span_evaluate.py
python src/refine_spans_final.py
```

**Note:** These scripts were developed with a particular local directory layout. Review their path constants and dataset-loading settings before running them on another machine. Reproducing training requires the source dataset, sufficient disk space, and appropriate compute. For inference only, download the fine-tuned checkpoint and run the demo.

## Dataset and annotation caveats

- The dataset includes `context` (a list of evidence passages), `question`, `answer`, `hallucination`, and character-span annotations (`hallucinated_labels`).
- Some hallucination-positive examples lack usable span labels; these were excluded from localization metrics but retained in answer-level evaluation.
- The challenge-dataset template scripts are exploratory and **do not constitute a completed external benchmark**.
- Avoid uploading private datasets, API keys, environment files, or model weights to ordinary Git commits.

## Sharing and collaboration

**GitHub** hosts source code, documentation, dependency lists, and small result files. **Hugging Face Hub** is recommended for sharing the trained checkpoint. After publishing the checkpoint, replace the placeholder in **Get the trained checkpoint** with the actual model URL and add a verified download command.

For changes to the code:

```bash
git add .
git commit -m "Describe the change"
git push
```

## Limitations and future work

- Evaluation so far is primarily on synthetic Hindi hallucination examples from the same dataset source.
- Short dates, numbers, and fragmented word boundaries remain challenging.
- V2 improves average span localization but can worsen some individual examples.
- Independent, reviewed Hindi RAG examples are needed to measure real-world generalization.

## Attribution

Base model: `jhu-clsp/mmBERT-small`. Dataset: `alexandrainst/multi-wiki-qa-synthetic-hallucinations`. Consult their respective model and dataset cards for licensing and attribution requirements before redistribution.
