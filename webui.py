import os
import re
import random
import shutil
import sys
import traceback
import gc
from pathlib import Path
from typing import Tuple
import spaces

import numpy as np
import torch
import librosa
import soundfile as sf
import gradio as gr

from preprocess.pipeline import PreprocessPipeline
from soulxsinger.utils.file_utils import load_config
from cli.inference import build_model as build_svs_model, process as svs_process


ROOT = Path(__file__).parent


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    try:
        from spaces.config import Config
        if Config.zero_gpu:
            return "cuda:0"
    except (ImportError, AttributeError):
        pass
    return "cpu"


def _session_dir_from_target(target_audio_path: str) -> Path:
    stem = Path(target_audio_path).stem
    safe = re.sub(r"[^\w\-]", "_", stem)
    safe = re.sub(r"_+", "_", safe).strip("_") or "session"
    return ROOT / "outputs" / "gradio" / safe[:64]


class AppState:
    def __init__(self) -> None:
        self.device = _get_device()
        self.preprocess_pipeline = PreprocessPipeline(
            device=self.device,
            language="English",
            save_dir=str(ROOT / "outputs" / "gradio" / "_placeholder" / "transcriptions"),
            vocal_sep=True,
            max_merge_duration=60000,
        )
        config = load_config("soulxsinger/config/soulxsinger.yaml")
        self.svs_config = config
        self.svs_model = build_svs_model(
            model_path="pretrained_models/SoulX-Singer/model.pt",
            config=config,
            device=self.device,
        )
        self.phoneset_path = "soulxsinger/utils/phoneme/phone_set.json"

    def run_preprocess(
        self,
        prompt_path: Path,
        target_path: Path,
        session_base: Path,
        prompt_vocal_sep: bool,
        target_vocal_sep: bool,
        prompt_lyric_lang: str,
        target_lyric_lang: str,
    ) -> Tuple[bool, str]:
        try:
            self.preprocess_pipeline.save_dir = str(session_base / "transcriptions" / "prompt")
            self.preprocess_pipeline.run(
                audio_path=str(prompt_path),
                vocal_sep=prompt_vocal_sep,
                max_merge_duration=20000,
                language=prompt_lyric_lang or "English",
            )
            self.preprocess_pipeline.save_dir = str(session_base / "transcriptions" / "target")
            self.preprocess_pipeline.run(
                audio_path=str(target_path),
                vocal_sep=target_vocal_sep,
                max_merge_duration=60000,
                language=target_lyric_lang or "English",
            )
            return True, "preprocess done"
        except Exception as e:
            return False, f"preprocess failed: {e}"

    def run_svs(
        self,
        control: str,
        session_base: Path,
        auto_shift: bool,
        pitch_shift: int,
    ) -> Tuple[bool, str, Path | None, Path | None, Path | None]:
        if control not in ("melody", "score"):
            control = "score"
        save_dir = session_base / "generated"
        save_dir.mkdir(parents=True, exist_ok=True)

        class Args:
            pass

        args = Args()
        args.device = self.device
        args.model_path = "pretrained_models/SoulX-Singer/model.pt"
        args.config = "soulxsinger/config/soulxsinger.yaml"
        args.prompt_wav_path = str(session_base / "audio" / "prompt.wav")
        prompt_meta_path = session_base / "transcriptions" / "prompt" / "metadata.json"
        target_meta_path = session_base / "transcriptions" / "target" / "metadata.json"
        args.prompt_metadata_path = str(prompt_meta_path)
        args.target_metadata_path = str(target_meta_path)
        args.phoneset_path = self.phoneset_path
        args.save_dir = str(save_dir)
        args.auto_shift = auto_shift
        args.pitch_shift = int(pitch_shift)
        args.control = control
        try:
            svs_process(args, self.svs_config, self.svs_model)
            generated = save_dir / "generated.wav"
            if not generated.exists():
                return False, f"inference finished but {generated} not found", None, prompt_meta_path, target_meta_path
            return True, "svs inference done", generated, prompt_meta_path, target_meta_path
        except Exception as e:
            return False, f"svs inference failed: {e}", None, prompt_meta_path, target_meta_path

    def run_svs_from_paths(
        self,
        prompt_wav_path: str,
        prompt_metadata_path: str,
        target_metadata_path: str,
        control: str,
        auto_shift: bool,
        pitch_shift: int,
        save_dir: Path | None = None,
    ) -> Tuple[bool, str, Path | None]:
        if save_dir is None:
            import uuid
            save_dir = ROOT / "outputs" / "gradio" / "synthesis" / str(uuid.uuid4())[:8]
        save_dir = Path(save_dir)
        audio_dir = save_dir / "audio"
        prompt_meta_dir = save_dir / "transcriptions" / "prompt"
        target_meta_dir = save_dir / "transcriptions" / "target"
        audio_dir.mkdir(parents=True, exist_ok=True)
        prompt_meta_dir.mkdir(parents=True, exist_ok=True)
        target_meta_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(prompt_wav_path, audio_dir / "prompt.wav")
        shutil.copy2(prompt_metadata_path, prompt_meta_dir / "metadata.json")
        shutil.copy2(target_metadata_path, target_meta_dir / "metadata.json")
        ok, msg, merged, _, _ = self.run_svs(
            control=control,
            session_base=save_dir,
            auto_shift=auto_shift,
            pitch_shift=pitch_shift,
        )
        if not ok or merged is None:
            return False, msg or "svs failed", None
        return True, "svs inference done", merged


from ensure_models import ensure_pretrained_models
ensure_pretrained_models()

APP_STATE = AppState()


def _resolve_file_path(x):
    if x is None:
        return None
    if isinstance(x, tuple):
        x = x[0]
    return x if (x and os.path.isfile(x)) else None


def _run_transcription_internal(
    prompt_audio, target_audio,
    prompt_lyric_lang, target_lyric_lang,
    prompt_vocal_sep, target_vocal_sep,
):
    """Run transcription, return (prompt_meta_path, target_meta_path) or (None, None)."""
    if isinstance(prompt_audio, tuple):
        prompt_audio = prompt_audio[0]
    if isinstance(target_audio, tuple):
        target_audio = target_audio[0]

    session_base = _session_dir_from_target(target_audio)
    audio_dir = session_base / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    SR = 44100
    PROMPT_MAX_SEC = 30
    TARGET_MAX_SEC = 60
    prompt_audio_data, _ = librosa.load(prompt_audio, sr=SR, mono=True)
    target_audio_data, _ = librosa.load(target_audio, sr=SR, mono=True)
    prompt_audio_data = prompt_audio_data[: PROMPT_MAX_SEC * SR]
    target_audio_data = target_audio_data[: TARGET_MAX_SEC * SR]
    sf.write(audio_dir / "prompt.wav", prompt_audio_data, SR)
    sf.write(audio_dir / "target.wav", target_audio_data, SR)

    ok, msg = APP_STATE.run_preprocess(
        audio_dir / "prompt.wav",
        audio_dir / "target.wav",
        session_base,
        prompt_vocal_sep=prompt_vocal_sep,
        target_vocal_sep=target_vocal_sep,
        prompt_lyric_lang=prompt_lyric_lang or "English",
        target_lyric_lang=target_lyric_lang or "English",
    )
    if not ok:
        print(msg, file=sys.stderr, flush=True)
        return None, None

    prompt_meta_path = session_base / "transcriptions" / "prompt" / "metadata.json"
    target_meta_path = session_base / "transcriptions" / "target" / "metadata.json"
    p = str(prompt_meta_path) if prompt_meta_path.exists() else None
    t = str(target_meta_path) if target_meta_path.exists() else None
    return p, t


@spaces.GPU
def transcription_function(
    prompt_audio, target_audio,
    prompt_metadata, target_metadata,
    prompt_lyric_lang, target_lyric_lang,
    prompt_vocal_sep, target_vocal_sep,
):
    """Step 1: Run transcription only; output (prompt_meta_path, target_meta_path)."""
    try:
        if isinstance(prompt_audio, tuple):
            prompt_audio = prompt_audio[0]
        if isinstance(target_audio, tuple):
            target_audio = target_audio[0]
        if prompt_audio is None or target_audio is None:
            gr.Warning(message="Please upload both prompt audio and target audio")
            return None, None

        prompt_meta_resolved = _resolve_file_path(prompt_metadata)
        target_meta_resolved = _resolve_file_path(target_metadata)
        use_input_metadata = prompt_meta_resolved is not None and target_meta_resolved is not None

        if use_input_metadata:
            session_base = _session_dir_from_target(target_audio)
            audio_dir = session_base / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            SR = 44100
            prompt_audio_data, _ = librosa.load(prompt_audio, sr=SR, mono=True)
            target_audio_data, _ = librosa.load(target_audio, sr=SR, mono=True)
            prompt_audio_data = prompt_audio_data[: 30 * SR]
            target_audio_data = target_audio_data[: 60 * SR]
            sf.write(audio_dir / "prompt.wav", prompt_audio_data, SR)
            sf.write(audio_dir / "target.wav", target_audio_data, SR)

            prompt_meta_path = session_base / "transcriptions" / "prompt" / "metadata.json"
            target_meta_path = session_base / "transcriptions" / "target" / "metadata.json"
            (session_base / "transcriptions" / "prompt").mkdir(parents=True, exist_ok=True)
            (session_base / "transcriptions" / "target").mkdir(parents=True, exist_ok=True)
            shutil.copy2(prompt_meta_resolved, prompt_meta_path)
            shutil.copy2(target_meta_resolved, target_meta_path)
            return str(prompt_meta_path), str(target_meta_path)
        else:
            return _run_transcription_internal(
                prompt_audio, target_audio,
                prompt_lyric_lang, target_lyric_lang,
                prompt_vocal_sep, target_vocal_sep,
            )
    except Exception:
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return None, None
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@spaces.GPU
def synthesis_function(
    prompt_audio,
    target_audio,
    prompt_metadata=None,
    target_metadata=None,
    control="melody",
    auto_shift=True,
    pitch_shift=0,
    seed=12306,
    prompt_lyric_lang="English",
    target_lyric_lang="English",
    prompt_vocal_sep=True,
    target_vocal_sep=True,
):
    """Single-button: runs transcription first if metadata not provided, then synthesis."""
    try:
        if isinstance(prompt_audio, tuple):
            prompt_audio = prompt_audio[0]
        if isinstance(target_audio, tuple):
            target_audio = target_audio[0]

        if not prompt_audio or not os.path.isfile(prompt_audio):
            gr.Warning(message="Please upload both prompt audio and target audio")
            return None, gr.update(), gr.update()
        if not target_audio or not os.path.isfile(target_audio):
            gr.Warning(message="Please upload both prompt audio and target audio")
            return None, gr.update(), gr.update()

        prompt_meta_path = _resolve_file_path(prompt_metadata)
        target_meta_path = _resolve_file_path(target_metadata)

        # Auto-run transcription if metadata not provided
        if not prompt_meta_path or not target_meta_path:
            p, t = _run_transcription_internal(
                prompt_audio, target_audio,
                prompt_lyric_lang, target_lyric_lang,
                prompt_vocal_sep, target_vocal_sep,
            )
            if not p or not t:
                gr.Warning(message="Transcription failed. Check your audio files.")
                return None, gr.update(), gr.update()
            prompt_meta_path = p
            target_meta_path = t

        # Prepare prompt wav
        session_base = _session_dir_from_target(target_audio)
        prompt_wav = session_base / "audio" / "prompt.wav"
        if not prompt_wav.exists():
            audio_dir = session_base / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            SR = 44100
            data, _ = librosa.load(prompt_audio, sr=SR, mono=True)
            data = data[: 30 * SR]
            sf.write(prompt_wav, data, SR)

        if control not in ("melody", "score"):
            control = "score"
        seed = int(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        ok, msg, merged = APP_STATE.run_svs_from_paths(
            prompt_wav_path=str(prompt_wav),
            prompt_metadata_path=prompt_meta_path,
            target_metadata_path=target_meta_path,
            control=control,
            auto_shift=auto_shift,
            pitch_shift=int(pitch_shift),
        )
        if not ok or merged is None:
            print(msg or "synthesis failed", file=sys.stderr, flush=True)
            return None, gr.update(), gr.update()

        # Return generated audio + update metadata displays
        return str(merged), prompt_meta_path, target_meta_path
    except Exception:
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return None, gr.update(), gr.update()
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()



def render_tab_content() -> None:
    """Render the main content (for embedding in app.py tabs). No Blocks or title."""
    with gr.Row(equal_height=False):
        # ── Left column: inputs & controls ──
        with gr.Column(scale=1):
            prompt_audio = gr.Audio(
                label="Prompt audio (reference voice), max 30s",
                type="filepath",
                interactive=True,
            )
            target_audio = gr.Audio(
                label="Target audio (melody / lyrics source), max 60s",
                type="filepath",
                interactive=True,
            )

            with gr.Row():
                control_radio = gr.Radio(
                    choices=["melody", "score"],
                    value="melody",
                    label="Control type",
                    scale=1,
                )
                auto_shift = gr.Checkbox(
                    label="Auto pitch shift",
                    value=True,
                    interactive=True,
                    scale=1,
                )

            synthesis_btn = gr.Button(
                value="🎤 Generate singing voice",
                variant="primary",
                size="lg",
            )

            # ── Advanced: transcription settings & metadata ──
            with gr.Accordion("Advanced: Transcription & Metadata", open=False):
                with gr.Row():
                    pitch_shift = gr.Number(
                        label="Pitch shift (semitones)",
                        value=0,
                        minimum=-36,
                        maximum=36,
                        step=1,
                        interactive=True,
                        scale=1,
                    )
                    seed_input = gr.Number(
                        label="Seed",
                        value=12306,
                        step=1,
                        interactive=True,
                        scale=1,
                    )
                gr.Markdown(
                    "Upload your own metadata files to skip automatic transcription. "
                    "You can use the [SoulX-Singer-Midi-Editor]"
                    "(https://huggingface.co/spaces/Soul-AILab/SoulX-Singer-Midi-Editor) "
                    "to edit metadata for better alignment."
                )
                with gr.Row():
                    prompt_lyric_lang = gr.Dropdown(
                        label="Prompt lyric language",
                        choices=[
                            ("Mandarin", "Mandarin"),
                            ("Cantonese", "Cantonese"),
                            ("English", "English"),
                        ],
                        value="English",
                        interactive=True,
                        scale=1,
                    )
                    target_lyric_lang = gr.Dropdown(
                        label="Target lyric language",
                        choices=[
                            ("Mandarin", "Mandarin"),
                            ("Cantonese", "Cantonese"),
                            ("English", "English"),
                        ],
                        value="English",
                        interactive=True,
                        scale=1,
                    )
                with gr.Row():
                    prompt_vocal_sep = gr.Checkbox(
                        label="Prompt vocal separation",
                        value=False,
                        interactive=True,
                        scale=1,
                    )
                    target_vocal_sep = gr.Checkbox(
                        label="Target vocal separation",
                        value=True,
                        interactive=True,
                        scale=1,
                    )
                transcription_btn = gr.Button(
                    value="Run singing transcription",
                    variant="secondary",
                    size="lg",
                )
                with gr.Row():
                    prompt_metadata = gr.File(
                        label="Prompt metadata",
                        type="filepath",
                        file_types=[".json"],
                        interactive=True,
                    )
                    target_metadata = gr.File(
                        label="Target metadata",
                        type="filepath",
                        file_types=[".json"],
                        interactive=True,
                    )

        # ── Right column: output ──
        with gr.Column(scale=1):
            output_audio = gr.Audio(
                label="Generated audio",
                type="filepath",
                interactive=False,
            )
            gr.Examples(
                examples=[
                    ["raven.wav", "happy_birthday.mp3"],
                    ["anita.wav", "happy_birthday.mp3"],
                    ["obama.wav", "happy_birthday.mp3"],
                    ["raven.wav", "everybody_loves.wav"],
                    ["anita.wav", "everybody_loves.wav"],
                    ["obama.wav", "everybody_loves.wav"],
                ],
                inputs=[prompt_audio, target_audio],
                outputs=[output_audio, prompt_metadata, target_metadata],
                fn=synthesis_function,
                cache_examples=True,
                cache_mode="lazy"
            )

    # ── Event handlers ──
    prompt_audio.change(
        fn=lambda: None,
        inputs=[],
        outputs=[prompt_metadata],
    )

    target_audio.change(
        fn=lambda: None,
        inputs=[],
        outputs=[target_metadata],
    )

    transcription_btn.click(
        fn=transcription_function,
        inputs=[
            prompt_audio, target_audio,
            prompt_metadata, target_metadata,
            prompt_lyric_lang, target_lyric_lang,
            prompt_vocal_sep, target_vocal_sep,
        ],
        outputs=[prompt_metadata, target_metadata],
    )

    synthesis_btn.click(
        fn=synthesis_function,
        inputs=[
            prompt_audio, target_audio,
            prompt_metadata, target_metadata,
            control_radio, auto_shift, pitch_shift, seed_input,
            prompt_lyric_lang, target_lyric_lang,
            prompt_vocal_sep, target_vocal_sep,
        ],
        outputs=[output_audio, prompt_metadata, target_metadata],
    )


def render_interface() -> gr.Blocks:
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
        render_tab_content()
    return page


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    page = render_interface()
    page.queue()
    page.launch(share=args.share, server_name="0.0.0.0", server_port=args.port)