Below is an example of a complete “STRIPPER Notebook” that you can run either in Google Colab or Kaggle. This notebook loads a Gemma 3 model, recursively replaces selected nonlinear functions with linear alternatives (using Identity modules as stand‐ins), reports which modules were replaced, and saves a new version of the model. You can then run a quick inference test to verify the changes.

---

```python
# %% [markdown]
# # STRIPPER Notebook for Gemma 3
#
# In this notebook we:
# 1. Load Google DeepMind's Gemma 3 model (from Hugging Face).
# 2. Recursively traverse the model to find and replace specific nonlinear modules (e.g. GeGLU, RMSNorm, QKNorm)
#    with linear functions (here using `nn.Identity`).
# 3. Report the changes that were made.
# 4. Save the new, “stripped” model.
# 5. Run a test inference to ensure the modified model runs.
#
# This notebook is designed for both Google Colab and Kaggle.

# %% [code]
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Replace this with the actual model ID if different
model_name = "google/deepmind-gemma-3-27b"  # Hypothetical model ID
print(f"Loading model {model_name}...")

# Load the Gemma 3 model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Model loaded successfully.")

# %% [markdown]
# ## Function to Replace Nonlinear Functions
#
# We define a function `replace_nonlinear` that traverses the model recursively.
# It looks for modules whose class names match our target nonlinear functions:
# - `GeGLU`
# - `RMSNorm`
# - `QKNorm`
#
# Each found module is replaced with a simple `nn.Identity()` layer, effectively removing its effect.

# %% [code]
def replace_nonlinear(module, report_changes=True):
    """
    Recursively replace nonlinear modules (GeGLU, RMSNorm, QKNorm) with Identity.
    Args:
      module (nn.Module): the module to traverse.
      report_changes (bool): if True, print details of replacements.
    """
    for name, child in module.named_children():
        child_class = child.__class__.__name__
        if child_class in ["GeGLU", "RMSNorm", "QKNorm"]:
            if report_changes:
                print(f"Replacing module '{name}' of type '{child_class}' with Identity.")
            setattr(module, name, nn.Identity())
        else:
            replace_nonlinear(child, report_changes=report_changes)

# %% [markdown]
# ## Apply the Replacement to the Model
#
# We run the replacement function on the loaded model.

# %% [code]
print("Starting replacement of nonlinear functions...")
replace_nonlinear(model)
print("Replacement complete.")

# %% [markdown]
# ## Save the Modified Model
#
# We save the new, stripped version of the Gemma 3 model locally.
# This version now uses pure feed-forward (linear) operations in place of the original nonlinearities.

# %% [code]
save_path = "./gemma3_stripped"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Modified model saved to {save_path}")

# %% [markdown]
# ## Quick Inference Test
#
# Let's test the modified model with a simple prompt.
# (Note: Removing nonlinearities may change model behavior dramatically, so this is just to verify execution.)
 
# %% [code]
prompt = "Calculate 8 divided by 2."
inputs = tokenizer(prompt, return_tensors="pt")

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate output with a limited number of tokens
outputs = model.generate(**inputs, max_new_tokens=20)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Test inference output:")
print(generated_text)

# %% [markdown]
# # Conclusion
#
# This notebook:
# - Loaded Gemma 3 from Hugging Face.
# - Replaced its nonlinear functions (GeGLU, RMSNorm, QKNorm) with Identity layers.
# - Reported each replacement.
# - Saved the new version to disk.
#
# You can now further evaluate the impact of removing these nonlinearities on accuracy, efficiency, and inference quality.
#
# Feel free to adjust the list of target modules or the replacement strategy (e.g., replacing with a custom linear layer) as needed.
```

---

This notebook is designed to run seamlessly in both Google Colab and Kaggle. It assumes that the Gemma 3 model is accessible via Hugging Face and that its nonlinear modules are identifiable by their class names. Adjust the model ID and module names if necessary. 
