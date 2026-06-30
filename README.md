# OpenMed Demo

A hands-on demo combining two complementary open-source projects from the [OpenMed](https://openmed.life) team:

- **OpenMed** — clinical text NLP (entity extraction, PII de-identification, 1,000+ models, runs 100% on-device)
- **SynthVision MedVL** — medical image visual question answering (fine-tuned vision-language models)

> ⚠️ **Research tools only.** These are not diagnostic medical devices. Do not use outputs for clinical decision-making. Always verify findings with a qualified medical professional.

---

## What the script does

### Part 1 — Clinical text (OpenMed)

Extracts structured clinical entities from free-form clinical notes — diseases, drugs, medications, anatomy — using OpenMed's specialized biomedical NER models. Everything runs locally; no text is sent to any server.

```
Note: Patient started on imatinib for chronic myeloid leukemia.
   DRUG         imatinib                       conf=0.95
   DISEASE      chronic myeloid leukemia       conf=0.98
```

### Part 2 — Medical images (SynthVision MedVL)

Loads `OpenMed/Qwen2.5-3B-MedVL` — a Qwen2.5-VL-3B model fine-tuned on ~200K medical VQA pairs from the SynthVision pipeline — and answers natural-language questions about any medical image you provide (chest X-ray, pathology slide, CT scan, etc.).

```
Question: What are the key findings in this chest X-ray?
Answer:   The image shows increased opacity in the right lower lobe...
```

---

## Why these two tools are separate

**OpenMed** (the Python library) is a clinical **text** NLP toolkit — it processes notes, reports, and records, not images. For images, the same team built **SynthVision**, which produces vision-language models published on the same [OpenMed HuggingFace org](https://huggingface.co/OpenMed). This script demonstrates both in one place.

---

## Installation

```bash
pip install "openmed[hf]" transformers torch accelerate pillow
```

For Apple Silicon (MLX acceleration):

```bash
pip install "openmed[mlx]" transformers torch accelerate pillow
```

Python 3.10+ required.

---

## Usage

**Text NER only** (no image needed, fast):

```bash
python openmed_demo.py --text-only
```

**Text NER + image Q&A** (downloads ~4B params on first run):

```bash
python openmed_demo.py --image path/to/xray.jpg
```

**Custom question:**

```bash
python openmed_demo.py --image path/to/scan.jpg \
  --question "Are there any signs of pneumonia?"
```

---

## Models used

| Part | Model | Size | Task |
|------|-------|------|------|
| Text | `disease_detection_superclinical` | 434M | Disease & condition NER |
| Text | `pharma_detection_superclinical` | 434M | Drug & medication NER |
| Text | `pii_superclinical_large` | 434M | PII detection & de-identification |
| Image | `OpenMed/Qwen2.5-3B-MedVL` | ~4B | Medical image Q&A (VQA) |

All models are Apache-2.0 licensed. The image model is fine-tuned from [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) using LoRA on VQA-RAD, PathVQA, and SLAKE benchmarks.

---

## OpenMed quick reference

```python
from openmed import analyze_text, extract_pii, deidentify

# Entity extraction
result = analyze_text(
    "Patient started on imatinib for chronic myeloid leukemia.",
    model_name="disease_detection_superclinical",
)
for entity in result.entities:
    print(entity.label, entity.text, entity.confidence)

# PII extraction
result = extract_pii(
    "Patient: John Doe, DOB: 01/15/1970, SSN: 123-45-6789",
    model_name="pii_superclinical_large",
    use_smart_merging=True,
)

# De-identification
deidentify(text, method="mask")     # → [NAME], [DATE], [SSN]
deidentify(text, method="replace")  # → realistic fake values
deidentify(text, method="hash")     # → cryptographic hash
```

---

## Resources

- [OpenMed GitHub](https://github.com/maziyarpanahi/openmed)
- [OpenMed HuggingFace org](https://huggingface.co/OpenMed) — 1,000+ models
- [SynthVision MedVL model card](https://huggingface.co/OpenMed/Qwen2.5-3B-MedVL)
- [SynthVision blog post](https://huggingface.co/blog/OpenMed/synthvision)
- [OpenMed docs](https://openmed.life/docs)
- [arXiv paper](https://arxiv.org/abs/2508.01630)

---

## License

Apache-2.0 — same as OpenMed and its models.
