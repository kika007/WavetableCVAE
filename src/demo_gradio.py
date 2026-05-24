from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pyrootutils
import torch
import torchaudio

# ---------------------------------------------------------------------
# setting up root directory
# ---------------------------------------------------------------------

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md", "LICENSE", ".git"],
    pythonpath=True,
    dotenv=True,
)

from src.models.cvae import LitCVAE
from src.models.components.visualize import EvalModelInit

CHECKPOINT_PATH = root / "ckpt/epoch=29999-step=2370000.ckpt"
PREDICT_DIR = root / "data/predict"
PREDICT_LABEL_DIR = PREDICT_DIR / "labels"

SAMPLE_RATE = 44100
TONE_DURATION_SEC = 5.0  # duration of a tone generated from wavetable


# ---------------------------------------------------------------------
# SimpleEvaluator
# ---------------------------------------------------------------------

class SimpleEvaluator(EvalModelInit):
    """Minimal wrapper reusing EvalModelInit without loading datasets."""
    def __init__(self, model: LitCVAE):
        self.model = model


def load_model() -> Tuple[EvalModelInit, torch.device]:
    """Load the trained CVAE checkpoint and return an evaluator helper."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LitCVAE.load_from_checkpoint(str(CHECKPOINT_PATH), map_location=device)
    model.eval()
    evaluator = SimpleEvaluator(model)
    return evaluator, device


# ---------------------------------------------------------------------
# Loading prediction samples and attributes
# ---------------------------------------------------------------------

def _list_predict_waves() -> Tuple[str, ...]:
    return tuple(sorted(f.name for f in PREDICT_DIR.glob("*.wav")))


@lru_cache(maxsize=None)
def _load_wave(sample_name: str) -> Tuple[torch.Tensor, int]:
    wav_path = PREDICT_DIR / sample_name
    waveform, sr = torchaudio.load(wav_path)
    return waveform, sr


@lru_cache(maxsize=None)
def _load_attrs(sample_name: str) -> Dict[str, float]:
    attrs_path = PREDICT_LABEL_DIR / f"{Path(sample_name).stem}_analysis.json"
    with attrs_path.open("r", encoding="utf-8") as fp:
        attrs = json.load(fp)
    return attrs


def _default_attr_values(sample_name: str) -> Tuple[float, float, float, float, float]:
    attrs = _load_attrs(sample_name)
    return (
        float(attrs.get("brightness", 0.0)),
        float(attrs.get("roughness", 0.0)),
        float(attrs.get("fullness", 0.0)),
        float(attrs.get("warmth", 0.0)),
        float(attrs.get("sharpness", 0.0)),
    )


# ---------------------------------------------------------------------
# Anti-Aliasing Filter (FFT Band-limiting)
# ---------------------------------------------------------------------

def bandlimit_wavetable(wavetable: np.ndarray, frequency_hz: float, sample_rate: int) -> np.ndarray:
    if len(wavetable) == 0:
        return wavetable

    nyquist = sample_rate / 2.0
    max_harmonic = int(nyquist / frequency_hz)

    spectrum = np.fft.rfft(wavetable)

    if max_harmonic < len(spectrum):
        spectrum[max_harmonic:] = 0.0

    filtered_wavetable = np.fft.irfft(spectrum, n=len(wavetable))
    return filtered_wavetable.astype(np.float32)


# ---------------------------------------------------------------------
# Wavetable to tone conversion
# ---------------------------------------------------------------------

def wavetable_to_tone(
    wavetable: np.ndarray,
    duration_sec: float = TONE_DURATION_SEC,
    sample_rate: int = SAMPLE_RATE,
    frequency_hz: float = 440.0,
) -> np.ndarray:
    
    wavetable = wavetable.flatten().astype(np.float32)
    num_samples = int(duration_sec * sample_rate)

    if len(wavetable) == 0:
        return np.zeros(num_samples, dtype=np.float32)

    wavetable = bandlimit_wavetable(wavetable, frequency_hz, sample_rate)

    table_len = len(wavetable)
    phase_increment = frequency_hz * table_len / sample_rate
    phase = np.arange(num_samples, dtype=np.float64) * phase_increment

    idx0 = np.floor(phase).astype(np.int64) % table_len
    idx1 = (idx0 + 1) % table_len
    frac = phase - np.floor(phase)

    tone = (1.0 - frac) * wavetable[idx0] + frac * wavetable[idx1]

    fade_len = int(0.01 * sample_rate)
    if fade_len > 0 and fade_len * 2 < len(tone):
        fade_in = np.linspace(0.0, 1.0, fade_len)
        fade_out = np.linspace(1.0, 0.0, fade_len)
        tone[:fade_len] *= fade_in
        tone[-fade_len:] *= fade_out

    return tone.astype(np.float32)


# ---------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------

def infer(
    sample_name: str,
    brightness_norm: float,
    roughness_norm: float,
    fullness_norm: float,
    warmth_norm: float,
    sharpness_norm: float,
    frequency: float,
) -> Tuple[Tuple[int, np.ndarray], plt.Figure]:
    
    waveform, _ = _load_wave(sample_name)
    batched_wav = waveform.unsqueeze(0)

    attrs = {
        "brightness": float(brightness_norm),
        "roughness": float(roughness_norm),
        "fullness": float(fullness_norm),
        "warmth": float(warmth_norm),
        "sharpness": float(sharpness_norm),
    }

    wavetable = evaluator.model_eval(batched_wav, attrs)
    wavetable_np = wavetable.squeeze().detach().cpu().numpy()
    
    # Remove DC offset and normalize
    wavetable_np = wavetable_np - np.mean(wavetable_np)
    max_val = np.max(np.abs(wavetable_np))
    if max_val > 0:
        wavetable_np = wavetable_np / max_val
    
    output_waveform = wavetable_to_tone(
        wavetable_np,
        duration_sec=TONE_DURATION_SEC,
        sample_rate=SAMPLE_RATE,
        frequency_hz=frequency,
    )

    # Remove DC offset and normalize output waveform
    output_waveform = output_waveform - np.mean(output_waveform)
    max_val = np.max(np.abs(output_waveform))
    if max_val > 0:
        output_waveform = output_waveform / max_val

    fig, ax = plt.subplots(figsize=(8, 3))
    sample_axis = np.arange(wavetable_np.size)
    ax.plot(sample_axis, wavetable_np)
    ax.set_title("Generated Wavetable")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()

    return (SAMPLE_RATE, output_waveform), fig


# ---------------------------------------------------------------------
# update attribute sliders
# ---------------------------------------------------------------------

def update_attributes(sample_name: str) -> Tuple[float, float, float, float, float]:
    return _default_attr_values(sample_name)


# ---------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------

css = """
.no-value-slider input[type="number"], .no-value-slider .value {
    display: none !important;
}
"""

def build_interface() -> gr.Blocks:
    sample_names = _list_predict_waves()
    default_sample = sample_names[0] if sample_names else ""
    default_attrs = _default_attr_values(default_sample) if default_sample else (0.0, 0.0, 0.0, 0.0, 0.0)

    with gr.Blocks(title="Wavetable CVAE Demo", css=css) as demo:
        gr.Markdown(
            "# Wavetable CVAE Demo\n"
        )

        with gr.Row():
            with gr.Column():
                sample_dropdown = gr.Dropdown(
                    choices=sample_names,
                    value=default_sample,
                    label="Input wavetable",
                )
                freq_slider = gr.Slider(55.0, 1760.0, value=440.0, step=1.0, label="Frequency (Hz)")
            
            with gr.Column():
                brightness_slider = gr.Slider(0.0, 1.0, value=default_attrs[0], step=0.01, label="Brightness", elem_classes="no-value-slider")
                roughness_slider = gr.Slider(0.0, 1.0, value=default_attrs[1], step=0.01, label="Roughness", elem_classes="no-value-slider")
                fullness_slider = gr.Slider(0.0, 1.0, value=default_attrs[2], step=0.01, label="Fullness", elem_classes="no-value-slider")
                warmth_slider = gr.Slider(0.0, 1.0, value=default_attrs[3], step=0.01, label="Warmth", elem_classes="no-value-slider")
                sharpness_slider = gr.Slider(0.0, 1.0, value=default_attrs[4], step=0.01, label="Sharpness", elem_classes="no-value-slider")

        generate_button = gr.Button("Generate", variant="primary")
        
        with gr.Row():
            audio_output = gr.Audio(label="Generated Audio", interactive=False)
            plot_output = gr.Plot(label="Waveform")

        inputs = [
            sample_dropdown,
            brightness_slider,
            roughness_slider,
            fullness_slider,
            warmth_slider,
            sharpness_slider,
            freq_slider,
        ]
        outputs = [audio_output, plot_output]

        generate_button.click(fn=infer, inputs=inputs, outputs=outputs)

        sample_dropdown.change(
            fn=update_attributes,
            inputs=sample_dropdown,
            outputs=[brightness_slider, roughness_slider, fullness_slider, warmth_slider, sharpness_slider],
        )

    return demo


# ---------------------------------------------------------------------
# load model and launch Gradio app
# ---------------------------------------------------------------------

evaluator, _ = load_model()
demo = build_interface()

if __name__ == "__main__":
    demo.launch(share=True)
