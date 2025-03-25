#!/usr/bin/env python3
import argparse
import logging
import math
from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup logging configuration
logging.basicConfig(
    format="[%(levelname)s] %(message)s",
    level=logging.INFO  # Change to DEBUG for more verbosity
)
logger = logging.getLogger(__name__)


def load_model(model_name: str):
    try:
        logger.info(f"Loading model '{model_name}'...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("Model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def inspect_model(model: nn.Module) -> None:
    logger.info("Inspecting top-level model modules:")
    for name, module in model.named_children():
        logger.info(f" - {name}: {module.__class__.__name__}")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params}")


def replace_nonlinear(module: nn.Module, counters: Dict[str, int], report_changes: bool = True) -> None:
    """Recursively replaces selected nonlinear modules with nn.Identity."""
    for name, child in module.named_children():
        child_class = child.__class__.__name__
        if child_class in {"GeGLU", "RMSNorm", "QKNorm"}:
            setattr(module, name, nn.Identity())
            counters["nonlinear_replacements"] += 1
            if report_changes:
                logger.info(f"Replacing {name} ({child_class}) with Identity.")
        else:
            replace_nonlinear(child, counters, report_changes=report_changes)


def wrap_tensor(inputs: torch.Tensor) -> torch.Tensor:
    """Apply toroidal wrapping on the 4D tensor."""
    # Wrap horizontally: last column to front, first column to end
    wrapped = torch.cat([inputs[:, :, -1:], inputs, inputs[:, :, :1]], dim=2)
    # Wrap vertically: last row to top, first row to bottom
    wrapped = torch.cat([wrapped[:, -1:], wrapped, wrapped[:, :1]], dim=1)
    return wrapped


class SeamlessWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            logger.warning(f"Unexpected input dimension ({x.dim()}); seamless wrapping skipped.")
            return self.module(x)

        batch_size, seq_len, hidden_dim = x.shape
        sqrt_val = math.isqrt(seq_len)
        if sqrt_val * sqrt_val != seq_len:
            logger.warning(f"seq_len ({seq_len}) is not a perfect square; seamless wrapping skipped.")
            return self.module(x)

        try:
            x_processed = self.module(x)
            x_reshaped = x_processed.view(batch_size, sqrt_val, sqrt_val, hidden_dim)
            x_wrapped = wrap_tensor(x_reshaped)
            new_seq_len = (sqrt_val + 2) ** 2
            x_final = x_wrapped.view(batch_size, new_seq_len, hidden_dim)
            logger.info("Seamless wrapping applied successfully.")
            return x_final
        except Exception as e:
            logger.error(f"Error during seamless wrapping: {e}. Returning original output.")
            return x_processed


def wrap_feedforward_modules(module: nn.Module, counters: Dict[str, int], report_changes: bool = True) -> None:
    """Recursively wraps feed-forward modules with SeamlessWrapper."""
    for name, child in list(module.named_children()):
        if 'mlp' in name.lower() or 'ffn' in name.lower():
            wrapped_child = SeamlessWrapper(child)
            setattr(module, name, wrapped_child)
            counters["ffn_wrapped"] += 1
            if report_changes:
                logger.info(f"Wrapped '{name}' with SeamlessWrapper.")
        else:
            wrap_feedforward_modules(child, counters, report_changes=report_changes)


def save_model(model: nn.Module, tokenizer, save_path: str) -> None:
    try:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        logger.info(f"Modified model saved successfully to {save_path}")
    except Exception as e:
        logger.error(f"Error saving modified model: {e}")
        raise


def main(args):
    # Load model and tokenizer
    model, tokenizer = load_model(args.model_name)

    # Inspect model structure and count parameters
    inspect_model(model)

    # Setup counters for modifications
    counters = {"nonlinear_replacements": 0, "ffn_wrapped": 0}

    # Replace nonlinear modules (STRIPPER process)
    logger.info("Starting STRIPPER process...")
    replace_nonlinear(model, counters)
    logger.info(f"STRIPPER process complete. Total replacements made: {counters['nonlinear_replacements']}")

    # Wrap feed-forward modules with SeamlessWrapper
    logger.info("Wrapping feed-forward modules with SeamlessWrapper...")
    wrap_feedforward_modules(model, counters)
    logger.info(f"Feed-forward wrapping complete. Total modules wrapped: {counters['ffn_wrapped']}")

    # Save the modified model
    save_model(model, tokenizer, args.save_path)

    # Final summary
    logger.info("Summary:")
    logger.info(f"- Nonlinear modules replaced: {counters['nonlinear_replacements']}")
    logger.info(f"- Feed-forward modules wrapped: {counters['ffn_wrapped']}")
    logger.info("Checkpoint ready for further experiments with deterministic tasks and reflection-based generation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seamless Gemma 3 1B STRIPPER Notebook")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-1b-pt",
                        help="The Hugging Face model identifier (default: google/gemma-3-1b-pt)")
    parser.add_argument("--save_path", type=str, default="./SEAMLESS-GEMMA-1B-RAW",
                        help="The directory path to save the modified model")
    args = parser.parse_args()
    main(args)
