import torch
import torch.nn as nn
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set logging function for clarity.
def log(message):
    print(f"[LOG]: {message}")

# Load and Inspect the Model
model_name = "google/deepmind-gemma-3-1b"
try:
    log(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    log("Model and tokenizer loaded successfully.")
except Exception as e:
    log(f"Error loading model: {e}")
    raise

# Quick inspection: List top-level modules
log("Top-level model modules:")
for name, module in model.named_children():
    log(f" - {name}: {module.__class__.__name__}")

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
log(f"Total parameters: {total_params}")

# STRIPPER – Replace Nonlinear Functions
replacement_counter = 0

def replace_nonlinear(module, report_changes=True):
    global replacement_counter
    for name, child in module.named_children():
        child_class = child.__class__.__name__
        if child_class in ["GeGLU", "RMSNorm", "QKNorm"]:
            setattr(module, name, nn.Identity())
            replacement_counter += 1
            if report_changes:
                log(f"Replacing {name} ({child_class}) with Identity.")
        else:
            replace_nonlinear(child, report_changes=report_changes)

log("Starting STRIPPER process...")
replace_nonlinear(model)
log(f"STRIPPER process complete. Total replacements made: {replacement_counter}")

# Seamless Wrapper Definition

def wrap_tensor(inputs: torch.Tensor) -> torch.Tensor:
    wrapped = torch.cat([inputs[:, :, -1:], inputs, inputs[:, :, :1]], dim=2)
    wrapped = torch.cat([wrapped[:, -1:], wrapped, wrapped[:, :1]], dim=1)
    return wrapped

class SeamlessWrapper(nn.Module):
    def __init__(self, module):
        super(SeamlessWrapper, self).__init__()
        self.module = module

    def forward(self, x):
        if x.dim() != 3:
            log(f"Unexpected input dimension ({x.dim()}), seamless wrapping skipped.")
            return self.module(x)

        batch_size, seq_len, hidden_dim = x.shape
        sqrt_val = math.isqrt(seq_len)
        if sqrt_val * sqrt_val != seq_len:
            log(f"Warning: seq_len ({seq_len}) not perfect square; seamless wrapping skipped.")
            return self.module(x)

        try:
            x_processed = self.module(x)
            x_reshaped = x_processed.view(batch_size, sqrt_val, sqrt_val, hidden_dim)
            x_wrapped = wrap_tensor(x_reshaped)
            new_seq_len = (sqrt_val + 2) ** 2
            x_final = x_wrapped.view(batch_size, new_seq_len, hidden_dim)
            log("Seamless wrapping applied successfully.")
            return x_final
        except Exception as e:
            log(f"Error during seamless wrapping: {e}. Returning original output.")
            return x_processed

# Wrap Feed-Forward Modules
wrap_counter = 0

def wrap_feedforward_modules(module, report_changes=True):
    global wrap_counter
    for name, child in list(module.named_children()):
        if 'mlp' in name.lower() or 'ffn' in name.lower():
            wrapped_child = SeamlessWrapper(child)
            setattr(module, name, wrapped_child)
            wrap_counter += 1
            if report_changes:
                log(f"Wrapped '{name}' with SeamlessWrapper.")
        else:
            wrap_feedforward_modules(child, report_changes=report_changes)

log("Wrapping feed-forward modules with SeamlessWrapper...")
wrap_feedforward_modules(model)
log(f"Feed-forward wrapping complete. Total modules wrapped: {wrap_counter}")

# Save the Modified Model
save_path = "./SEAMLESS-GEMMA-1B-RAW"
try:
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    log(f"Modified model saved successfully to {save_path}")
except Exception as e:
    log(f"Error saving modified model: {e}")
    raise

# Final Summary
log("Summary:")
log(f"- Nonlinear modules replaced: {replacement_counter}")
log(f"- Feed-forward modules wrapped: {wrap_counter}")
log("Checkpoint ready for further experiments with deterministic tasks and reflection-based generation.")
