# GEMCRAFT

Gemma Optimizer ✨
----------------

This project provides tools to modify Google DeepMind's Gemma models.
The optimizer removes nonlinear layers (the **STRIPPER** step) and wraps
feed‑forward blocks with a "seamless" toroidal transformation. The
seamless wrapper now pads non-square sequence lengths so the wrapping is
always applied.

Install the required dependencies with `pip install -r requirements.txt`.

This will provide two console commands: `gemcraft-inference` and
`gemcraft-seamless` for running inference and model modification
respectively.

To process the 1B model run:

```bash
gemcraft-seamless --model_name google/gemma-3-1b-pt \
    --save_path ./SEAMLESS-GEMMA-1B-RAW
```

The script now accepts extra options:

- `--device` selects the torch device (e.g. `cuda`).
- `--skip_stripper` skips nonlinear layer removal.
- `--skip_seamless` skips the seamless wrapping step.

After generating the modified checkpoint you can test inference with:

```bash
gemcraft-inference --model_path ./SEAMLESS-GEMMA-1B-RAW \
    --prompt "Calculate 8 divided by 2."
```

To verify your Python environment you can precompile the scripts:

```bash
python -m compileall NOTEBOOK/seamless-gemma.py inference.py
```


For a walkthrough of how the scripts interact, see [DOC/PROGRAM_FLOW.md](DOC/PROGRAM_FLOW.md).
