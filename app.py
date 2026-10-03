"""
Hugging Face Space entry point for SoulX-Singer.
Downloads pretrained models from the Hub if needed, then launches the Gradio app.
"""

import os
from pathlib import Path

# Set matplotlib backend before any imports that might use it (required for headless environments)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from ensure_models import ensure_pretrained_models

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    os.chdir(ROOT)
    ensure_pretrained_models()

    import gradio as gr
    from webui import render_tab_content as render_svs_tab
    from webui_svc import render_tab_content as render_svc_tab

    with gr.Blocks(title="SoulX-Singer", theme=gr.themes.Default()) as page:
        gr.HTML(
            '<div style="'
            'text-align: center; '
            'padding: 1.25rem 0 1.5rem; '
            'margin-bottom: 0.5rem;'
            '">'
            '<div style="'
            'display: inline-block; '
            'font-size: 1.75rem; '
            'font-weight: 700; '
            'letter-spacing: 0.02em; '
            'line-height: 1.3;'
            '">SoulX-Singer</div>'
            '<div style="'
            'width: 80px; '
            'height: 3px; '
            'margin: 1rem auto 0; '
            'background: linear-gradient(90deg, transparent, #6366f1, transparent); '
            'border-radius: 2px;'
            '"></div>'
            '</div>'
        )
        with gr.Tabs():
            with gr.Tab("Singing Voice Synthesis"):
                render_svs_tab()
            with gr.Tab("Singing Voice Conversion"):
                render_svc_tab()

    page.queue()
    page.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        share=True,
    )
