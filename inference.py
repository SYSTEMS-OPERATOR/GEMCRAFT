#!/usr/bin/env python3
"""Simple inference script for Gemma models."""
import argparse
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def load(model_path: str):
    """Load model and tokenizer from a local path or Hugging Face repo."""
    logger.info("Loading model %s", model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def run_inference(model, tokenizer, prompt: str, max_new_tokens: int = 20) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Run a quick inference with a Gemma model")
    parser.add_argument("--model_path", required=True, help="Path or Hugging Face id of the model")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--max_new_tokens", type=int, default=20, help="Number of tokens to generate")
    args = parser.parse_args()

    model, tokenizer = load(args.model_path)
    text = run_inference(model, tokenizer, args.prompt, args.max_new_tokens)
    print(text)


if __name__ == "__main__":
    main()
