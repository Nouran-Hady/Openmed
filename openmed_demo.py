"""
OpenMed demo: clinical text NER + medical image Q&A (SynthVision MedVL)
=========================================================================

OpenMed itself (github.com/maziyarpanahi/openmed) is a TEXT-ONLY clinical
NLP toolkit — it extracts diseases, drugs, anatomy, and PII from clinical
notes, running 100% on-device, no cloud, no API key.

OpenMed does NOT process images directly. For medical IMAGES, the same
team built a sister project called SynthVision, which fine-tunes vision-
language models (Qwen2.5-VL, etc.) on medical visual question answering —
published under the same OpenMed HuggingFace org.

This script demonstrates both halves:

  Part 1 — OpenMed: extract structured clinical entities from text
  Part 2 — SynthVision MedVL: ask natural-language questions about a
           medical image (X-ray, scan, pathology slide, etc.)

IMPORTANT — Read before using on real data:
  These are research/open-source tools, NOT diagnostic medical devices.
  Do not use outputs for clinical decision-making. Always verify findings
  with a qualified medical professional.

Install:
    pip install "openmed[hf]" transformers torch accelerate pillow --break-system-packages

Usage:
    python openmed_demo.py --text-only          # just run the NER part
    python openmed_demo.py --image path/to.jpg  # run image Q&A too
    python openmed_demo.py --image path/to.jpg --question "Describe any abnormal findings."
"""

import argparse
import sys


# ---------------------------------------------------------------------------
# Part 1 — OpenMed: clinical text NER (entity extraction)
# ---------------------------------------------------------------------------
def run_clinical_text_demo():
    print("\n=== Part 1: OpenMed — Clinical Text Entity Extraction ===\n")
    try:
        from openmed import analyze_text
    except ImportError:
        print("OpenMed isn't installed. Install it with:")
        print('  pip install "openmed[hf]" --break-system-packages')
        return

    sample_notes = [
        "Patient started on imatinib for chronic myeloid leukemia.",
        "75mg clopidogrel was administered for NSTEMI.",
        "Patient presents with chronic myeloid leukemia and Type 2 diabetes.",
    ]

    for note in sample_notes:
        print(f"Note: {note}")
        try:
            result = analyze_text(note, model_name="disease_detection_superclinical")
            if result.entities:
                for entity in result.entities:
                    print(f"   {entity.label:<12} {entity.text:<30} conf={entity.confidence:.2f}")
            else:
                print("   (no entities found by this model)")
        except Exception as e:
            print(f"   [skipped — model download or runtime issue: {e}]")
        print()


# ---------------------------------------------------------------------------
# Part 2 — SynthVision MedVL: medical image Q&A
# ---------------------------------------------------------------------------
def run_image_demo(image_path: str, question: str):
    print("\n=== Part 2: SynthVision MedVL — Medical Image Q&A ===\n")
    print("Model: OpenMed/Qwen2.5-3B-MedVL  (Qwen2.5-VL-3B fine-tuned on ~200K medical VQA pairs)")
    print(f"Image: {image_path}")
    print(f"Question: {question}\n")

    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, AutoModelForImageTextToText
    except ImportError:
        print("Missing dependencies. Install with:")
        print("  pip install transformers torch accelerate pillow --break-system-packages")
        return

    model_id = "OpenMed/Qwen2.5-3B-MedVL"

    print("Loading model (first run downloads ~4B params, may take a while)...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=512)
    answer = processor.decode(
        output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    )

    print("Model answer:")
    print(f"  {answer}\n")
    print("⚠️  Research model output — not a diagnosis. Always confirm findings with a clinician.")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="OpenMed + SynthVision MedVL demo")
    parser.add_argument("--image", type=str, default=None, help="Path to a medical image (X-ray, scan, etc.)")
    parser.add_argument(
        "--question",
        type=str,
        default="What are the key findings in this medical image?",
        help="Question to ask about the image",
    )
    parser.add_argument("--text-only", action="store_true", help="Only run the clinical text NER demo")
    args = parser.parse_args()

    run_clinical_text_demo()

    if args.text_only:
        return

    if args.image:
        run_image_demo(args.image, args.question)
    else:
        print("\nNo --image provided — skipping the image Q&A demo.")
        print("Try: python openmed_demo.py --image your_xray.jpg")


if __name__ == "__main__":
    main()
