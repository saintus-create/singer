"""Standalone SoulX-Singer SVC web app.

Runs the singing voice conversion interface without Hugging Face Spaces or ZeroGPU.
Pretrained models are downloaded from Hugging Face Hub on first start if missing.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from ensure_models import ensure_pretrained_models

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    os.chdir(ROOT)
    ensure_pretrained_models()

    import gradio as gr
    from webui_svc import render_tab_content

    with gr.Blocks(title="SoulX-Singer SVC", theme=gr.themes.Default()) as page:
        gr.Markdown("# SoulX-Singer SVC")
        render_tab_content()

    page.queue()
    page.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        share=False,
    )
