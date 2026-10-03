import random
import sys
import traceback
import gc
from datetime import datetime
from pathlib import Path
from typing import Literal

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch

import spaces
from preprocess.pipeline import PreprocessPipeline
from soulxsinger.utils.file_utils import load_config
from cli.inference_svc import build_model as build_svc_model, process as svc_process


ROOT = Path(__file__).parent
SAMPLE_RATE = 44100
PROMPT_MAX_SEC_DEFAULT = 30
TARGET_MAX_SEC_DEFAULT = 600

# Example rows: only [prompt_audio, target_audio]; other params use UI defaults when running
EXAMPLE_LIST = [
    [str(ROOT / "example/audio/zh_prompt.mp3"), str(ROOT / "example/audio/zh_target.mp3")],
    [str(ROOT / "example/audio/en_prompt.mp3"), str(ROOT / "example/audio/en_target.mp3")],
    [str(ROOT / "example/audio/svc_webui/Sun Yanzi.mp3"), str(ROOT / "example/audio/svc_webui/I'm Yours.mp3")],
    [str(ROOT / "example/audio/svc_webui/Sun Yanzi.mp3"), str(ROOT / "example/audio/svc_webui/传奇.mp3")],
    [str(ROOT / "example/audio/svc_webui/Sun Yanzi.mp3"), str(ROOT / "example/audio/svc_webui/君が好きだと叫びたい.mp3")],
    [str(ROOT / "example/audio/svc_webui/Sun Yanzi.mp3"), str(ROOT / "example/audio/svc_webui/富士山下.mp3")],
]

_I18N = dict(
	display_lang_label=dict(en="Display Language", zh="显示语言"),
	title=dict(en="## SoulX-Singer SVC", zh="## SoulX-Singer SVC"),
	prompt_audio_label=dict(en=f"Prompt audio", zh=f"Prompt 音频"),
	target_audio_label=dict(en=f"Target audio", zh=f"Target 音频"),
	prompt_vocal_sep_label=dict(en="Prompt vocal separation", zh="Prompt 人声分离"),
	target_vocal_sep_label=dict(en="Target vocal separation", zh="Target 人声分离"),
	auto_shift_label=dict(en="Auto pitch shift", zh="自动变调"),
	auto_mix_acc_label=dict(en="Auto mix accompaniment", zh="自动混合伴奏"),
	pitch_shift_label=dict(en="Pitch shift (semitones)", zh="指定变调（半音）"),
	n_step_label=dict(en="diffusion steps", zh="采样步数"),
	cfg_label=dict(en="cfg scale", zh="cfg系数"),
	seed_label=dict(en="Seed", zh="种子"),
	examples_label=dict(en="Examples", zh="示例"),
	run_btn=dict(en="🎤Singing Voice Conversion", zh="🎤歌声转换"),
	output_audio_label=dict(en="Generated audio", zh="合成结果音频"),
	warn_missing_audio=dict(en="Please provide both prompt audio and target audio.", zh="请同时上传 Prompt 与 Target 音频。"),
	instruction_title=dict(en="Usage", zh="使用说明"),
	instruction_p1=dict(
        en="Upload the Prompt and Target audio, and configure the parameters",
        zh="上传 Prompt 与 Target 音频，并配置相关参数",
    ),
    instruction_p2=dict(
        en="Click「🎤Singing Voice Conversion」to start singing voice conversion.",
        zh="点击「🎤歌声转换」开始最终生成。",
    ),
	tips_title=dict(en="Tips", zh="提示"),
	tip_p1=dict(
        en="Input: The Prompt audio is recommended to be a clean and clear singing voice, while the Target audio can be either a pure vocal or a mixture with accompaniment. If the audio contains accompaniment, please check the vocal separation option.",
        zh="输入：Prompt 音频建议是干净清晰的歌声，Target 音频可以是纯歌声或伴奏，这两者若带伴奏需要勾选分离选项",
    ),
	tip_p2=dict(
        en="Pitch shift: When there is a large pitch range difference between the Prompt and Target audio, you can try enabling auto pitch shift or manually adjusting the pitch shift in semitones. When a non-zero pitch shift is specified, auto pitch shift will not take effect. The accompaniment of auto mix will be pitch-shifted together with the vocal (keeping the same octave).",
        zh="变调：Prompt 音频的音域和 Target 音频的音域差距较大的时候，可以尝试开启自动变调或手动调整变调半音数，指定非0的变调半音数时，自动变调不生效，自动混音的伴奏会配合歌声进行升降调（保持同一个八度）",
    ),
	tip_p3=dict(
        en="Model parameters: Generally, a larger number of sampling steps will yield better generation quality but also longer generation time; a larger cfg scale will increase timbre similarity and melody fidelity, but may cause more distortion, it is recommended to take a value between 1 and 3.",
        zh="模型参数：一般采样步数越大，生成质量越好，但生成时间也越长；一般cfg系数越大，音色相似度和旋律保真度越高，但是会造成更多的失真，建议取1～3之间的值",
    ),
	tip_p4=dict(
        en="If you want to convert a long audio or a whole song with large pitch range, there may be instability in the generated voice. You can try converting in segments.",
        zh="长音频或完整歌曲中，音域变化较大的情况有可能出现音色不稳定，可以尝试分段转换",
    )
)

_GLOBAL_LANG: Literal["zh", "en"] = "zh"


def _i18n(key: str) -> str:
	return _I18N[key][_GLOBAL_LANG]


def _print_exception(context: str) -> None:
	print(f"[{context}]\n{traceback.format_exc()}", file=sys.stderr, flush=True)


def _get_device() -> str:
	return "cuda:0" if torch.cuda.is_available() else "cpu"


def _session_dir() -> Path:
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
	return ROOT / "outputs" / "gradio" / "svc" / timestamp


def _normalize_audio_input(audio):
	return audio[0] if isinstance(audio, tuple) else audio


def _trim_and_save_audio(src_audio_path: str, dst_wav_path: Path, max_sec: int, sr: int = SAMPLE_RATE) -> None:
	audio_data, _ = librosa.load(src_audio_path, sr=sr, mono=True)
	audio_data = audio_data[: max_sec * sr]
	dst_wav_path.parent.mkdir(parents=True, exist_ok=True)
	sf.write(dst_wav_path, audio_data, sr)


def _usage_md() -> str:
	return "\n\n".join([
		f"### {_i18n('instruction_title')}",
		f"**1.** {_i18n('instruction_p1')}",
		f"**2.** {_i18n('instruction_p2')}",
	])


def _tips_md() -> str:
	return "\n\n".join([
		f"### {_i18n('tips_title')}",
		f"- {_i18n('tip_p1')}",
		f"- {_i18n('tip_p2')}",
		f"- {_i18n('tip_p3')}",
		f"- {_i18n('tip_p4')}",
	])


class AppState:
	def __init__(self) -> None:
		self.device = _get_device()
		self.preprocess_pipeline = PreprocessPipeline(
			device=self.device,
			language="Mandarin",
			save_dir=str(ROOT / "outputs" / "gradio" / "_placeholder" / "svc"),
			vocal_sep=True,
			max_merge_duration=60000,
			midi_transcribe=False,
		)

		self.svc_config = load_config("soulxsinger/config/soulxsinger.yaml")
		self.svc_model = build_svc_model(
			model_path="pretrained_models/SoulX-Singer/model-svc.pt",
			config=self.svc_config,
			device=self.device,
		)

	def run_preprocess(self, audio_path: Path, save_path: Path, vocal_sep: bool) -> tuple[bool, str, Path | None, Path | None]:
		try:
			self.preprocess_pipeline.save_dir = str(save_path)
			self.preprocess_pipeline.run(
				audio_path=str(audio_path),
				vocal_sep=vocal_sep,
				max_merge_duration=60000,
				language="Mandarin",
			)
			vocal_wav = save_path / "vocal.wav"
			vocal_f0 = save_path / "vocal_f0.npy"
			if not vocal_wav.exists() or not vocal_f0.exists():
				return False, f"preprocess output missing: {vocal_wav} or {vocal_f0}", None, None
			return True, "ok", vocal_wav, vocal_f0
		except Exception as e:
			return False, f"preprocess failed: {e}", None, None

	def run_svc(
		self,
		prompt_wav_path: Path,
		target_wav_path: Path,
		prompt_f0_path: Path,
		target_f0_path: Path,
		session_base: Path,
		auto_shift: bool,
		auto_mix_acc: bool,
		pitch_shift: int,
		n_step: int,
		cfg: float,
		use_fp16: bool,
		seed: int,
	) -> tuple[bool, str, Path | None]:
		try:
			torch.manual_seed(seed)
			np.random.seed(seed)
			random.seed(seed)

			save_dir = session_base / "generated"
			save_dir.mkdir(parents=True, exist_ok=True)

			class Args:
				pass

			args = Args()
			args.device = self.device
			args.prompt_wav_path = str(prompt_wav_path)
			args.target_wav_path = str(target_wav_path)
			args.prompt_f0_path = str(prompt_f0_path)
			args.target_f0_path = str(target_f0_path)
			args.save_dir = str(save_dir)
			args.auto_shift = auto_shift
			args.pitch_shift = int(pitch_shift)
			args.n_steps = int(n_step)
			args.cfg = float(cfg)
			args.use_fp16 = bool(use_fp16)

			svc_process(args, self.svc_config, self.svc_model)

			generated = save_dir / "generated.wav"
			if not generated.exists():
				return False, f"inference finished but output not found: {generated}", None

			if auto_mix_acc:
				acc_path = session_base / "transcriptions" / "target" / "acc.wav"
				if acc_path.exists():
					vocal_shift = args.pitch_shift
					mul = -1 if vocal_shift < 0 else 1
					acc_shift = abs(vocal_shift) % 12
					acc_shift = mul * acc_shift
					if acc_shift > 6:
						acc_shift -= 12
					if acc_shift < -6:
						acc_shift += 12

					mix_sr = self.svc_config.audio.sample_rate
					vocal, _ = librosa.load(str(generated), sr=mix_sr, mono=True)
					acc, _ = librosa.load(str(acc_path), sr=mix_sr, mono=True)
					if acc_shift != 0:
						acc = librosa.effects.pitch_shift(acc, sr=mix_sr, n_steps=acc_shift)
						print(f"Applied pitch shift of {acc_shift} semitones to accompaniment to match vocal shift of {vocal_shift} semitones.")
						
					mix_len = min(len(vocal), len(acc))
					if mix_len > 0:
						mixed = vocal[:mix_len] + acc[:mix_len]
						peak = float(np.max(np.abs(mixed))) if mixed.size > 0 else 1.0
						if peak > 1.0:
							mixed = mixed / peak
						mixed_path = save_dir / "generated_mixed.wav"
						sf.write(str(mixed_path), mixed, mix_sr)
						generated = mixed_path

			return True, "svc inference done", generated
		except Exception as e:
			return False, f"svc inference failed: {e}", None


APP_STATE = AppState()


@spaces.GPU
def _run_svc_preprocess(
    prompt_audio,
    target_audio,
    prompt_vocal_sep=False,
    target_vocal_sep=True,
):
	try:
		prompt_audio = _normalize_audio_input(prompt_audio)
		target_audio = _normalize_audio_input(target_audio)
		if not prompt_audio or not target_audio:
			gr.Warning(_i18n("warn_missing_audio"))
			return None

		session_base = _session_dir()
		audio_dir = session_base / "audio"
		prompt_raw = audio_dir / "prompt.wav"
		target_raw = audio_dir / "target.wav"
		_trim_and_save_audio(prompt_audio, prompt_raw, PROMPT_MAX_SEC_DEFAULT)
		_trim_and_save_audio(target_audio, target_raw, TARGET_MAX_SEC_DEFAULT)

		prompt_ok, prompt_msg, prompt_wav, prompt_f0 = APP_STATE.run_preprocess(
			audio_path=prompt_raw,
			save_path=session_base / "transcriptions" / "prompt",
			vocal_sep=bool(prompt_vocal_sep),
		)
		if not prompt_ok or prompt_wav is None or prompt_f0 is None:
			print(prompt_msg, file=sys.stderr, flush=True)
			return None

		target_ok, target_msg, target_wav, target_f0 = APP_STATE.run_preprocess(
			audio_path=target_raw,
			save_path=session_base / "transcriptions" / "target",
			vocal_sep=bool(target_vocal_sep),
		)
		if not target_ok or target_wav is None or target_f0 is None:
			print(target_msg, file=sys.stderr, flush=True)
			return None

		return (
			str(session_base),
			str(prompt_wav),
			str(prompt_f0),
			str(target_wav),
			str(target_f0),
		)
	except Exception:
		_print_exception("_run_svc_preprocess")
		return None
	finally:
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()


@spaces.GPU
def _run_svc_convert(
    preprocess_state,
    auto_shift=True,
    auto_mix_acc=True,
    pitch_shift=0,
    n_step=32,
    cfg=1.0,
    use_fp16=True,
    seed=42,
):
	try:
		if preprocess_state is None or not isinstance(preprocess_state, (tuple, list)) or len(preprocess_state) != 5:
			return None
		session_base_str, prompt_wav, prompt_f0, target_wav, target_f0 = preprocess_state
		session_base = Path(session_base_str)

		ok, msg, generated = APP_STATE.run_svc(
			prompt_wav_path=Path(prompt_wav),
			target_wav_path=Path(target_wav),
			prompt_f0_path=Path(prompt_f0),
			target_f0_path=Path(target_f0),
			session_base=session_base,
			auto_shift=bool(auto_shift),
			auto_mix_acc=bool(auto_mix_acc),
			pitch_shift=int(pitch_shift),
			n_step=int(n_step),
			cfg=float(cfg),
			use_fp16=bool(use_fp16),
			seed=int(seed),
		)
		if not ok or generated is None:
			print(msg, file=sys.stderr, flush=True)
			return None
		return str(generated)
	except Exception:
		_print_exception("_run_svc_convert")
		return None
	finally:
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()


@spaces.GPU
def _start_svc(
    prompt_audio,
    target_audio,
    prompt_vocal_sep=False,
    target_vocal_sep=True,
    auto_shift=True,
    auto_mix_acc=True,
    pitch_shift=0,
    n_step=32,
    cfg=1.0,
    use_fp16=True,
    seed=42,
):
	state = _run_svc_preprocess(prompt_audio, target_audio, prompt_vocal_sep, target_vocal_sep)
	if state is None:
		return None
	return _run_svc_convert(state, auto_shift, auto_mix_acc, pitch_shift, n_step, cfg, use_fp16, seed)


def render_tab_content() -> None:
    with gr.Row(equal_height=False):
        # ── Left column: inputs & controls ──
        with gr.Column(scale=1):
            prompt_audio = gr.Audio(
                label="Prompt audio (reference voice)",
                type="filepath",
                interactive=True,
            )
            target_audio = gr.Audio(
                label="Target audio (to convert)",
                type="filepath",
                interactive=True,
            )

            run_btn = gr.Button(
                value="🎤 Singing Voice Conversion",
                variant="primary",
                size="lg",
            )

            with gr.Accordion("Advanced settings", open=False):
                with gr.Row():
                    prompt_vocal_sep = gr.Checkbox(label="Prompt vocal separation", value=False, scale=1)
                    target_vocal_sep = gr.Checkbox(label="Target vocal separation", value=True, scale=1)
                with gr.Row():
                    auto_shift = gr.Checkbox(label="Auto pitch shift", value=True, scale=1)
                    auto_mix_acc = gr.Checkbox(label="Auto mix accompaniment", value=True, scale=1)
                with gr.Row():
                    use_fp16 = gr.Checkbox(label="Use FP16", value=True, scale=1)
                pitch_shift = gr.Slider(label="Pitch shift (semitones)", value=0, minimum=-36, maximum=36, step=1)
                n_step = gr.Slider(label="diffusion steps", value=32, minimum=1, maximum=200, step=1)
                cfg = gr.Slider(label="cfg scale", value=1.0, minimum=0.0, maximum=10.0, step=0.1)
                seed_input = gr.Slider(label="Seed", value=42, minimum=0, maximum=10000, step=1)

        # ── Right column: output ──
        with gr.Column(scale=1):
            output_audio = gr.Audio(label="Generated audio", type="filepath", interactive=False)
            svc_state = gr.State(value=None)
            gr.Examples(
                examples=EXAMPLE_LIST,
                inputs=[prompt_audio, target_audio],
                outputs=[output_audio],
                fn=_start_svc,
                cache_examples=True,
                cache_mode="lazy",
            )

    run_btn.click(
        fn=_run_svc_preprocess,
        inputs=[prompt_audio, target_audio, prompt_vocal_sep, target_vocal_sep],
        outputs=[svc_state],
    ).then(
        fn=_run_svc_convert,
        inputs=[svc_state, auto_shift, auto_mix_acc, pitch_shift, n_step, cfg, use_fp16, seed_input],
        outputs=[output_audio],
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
	parser.add_argument("--port", type=int, default=7861, help="Gradio server port")
	parser.add_argument("--share", action="store_true", help="Create public link")
	args = parser.parse_args()

	page = render_interface()
	page.queue()
	page.launch(share=args.share, server_name="0.0.0.0", server_port=args.port)
