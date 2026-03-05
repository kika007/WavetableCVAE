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

CHECKPOINT_PATH = root / "ckpt/epoch=29999-step=3120000.ckpt"
PREDICT_DIR = root / "data/predict"
PREDICT_LABEL_DIR = PREDICT_DIR / "labels"

SAMPLE_RATE = 44100
TONE_DURATION_SEC = 1.0  # duration of a tone generated from wavetable


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


def _clip_zcr(value: float) -> float:
    """Clamp zero-crossing rate to 0-1 range."""
    return float(np.clip(value, 0.0, 1.0))


def _default_attr_values(sample_name: str) -> Tuple[float, float, float, float]:
    attrs = _load_attrs(sample_name)
    return (
        float(attrs.get("dco_brightness", 0.5)),
        float(attrs.get("dco_richness", 0.5)),
        float(attrs.get("dco_oddenergy", 0.5)),
        _clip_zcr(attrs.get("dco_zcr", 0.5)),
    )


# ---------------------------------------------------------------------
# Wavetable to tone conversion
# ---------------------------------------------------------------------

def wavetable_to_tone(
    wavetable: np.ndarray,
    duration_sec: float = TONE_DURATION_SEC,
    sample_rate: int = SAMPLE_RATE,
    frequency_hz: float = 440.0,   # Target output frequency in Hz
) -> np.ndarray:
    # Flatten wavetable and ensure float32 type
    wavetable = wavetable.flatten().astype(np.float32)
    num_samples = int(duration_sec * sample_rate)

    # If wavetable is empty, return silence
    if len(wavetable) == 0:
        return np.zeros(num_samples, dtype=np.float32)

    table_len = len(wavetable)

    # Phase increment: how many wavetable samples we advance per output sample
    # This determines the resulting pitch
    phase_increment = frequency_hz * table_len / sample_rate

    # Precompute all phase values for speed (one per output sample)
    phase = np.arange(num_samples, dtype=np.float64) * phase_increment

    # Integer part of phase (base index in wavetable)
    idx0 = np.floor(phase).astype(np.int64) % table_len

    # Next index in the wavetable (wrap-around)
    idx1 = (idx0 + 1) % table_len

    # Fractional part for linear interpolation
    frac = phase - np.floor(phase)

    # Linear interpolation between wavetable[idx0] and wavetable[idx1]
    tone = (1.0 - frac) * wavetable[idx0] + frac * wavetable[idx1]

    # Apply fade-in and fade-out (10 ms each) to avoid clicks
    fade_len = int(0.01 * sample_rate)
    if fade_len > 0 and fade_len * 2 < len(tone):
        fade_in = np.linspace(0.0, 1.0, fade_len)
        fade_out = np.linspace(1.0, 0.0, fade_len)
        tone[:fade_len] *= fade_in
        tone[-fade_len:] *= fade_out

    return tone.astype(np.float32)



# ---------------------------------------------------------------------
# Inference function: -> wavetable -> generated wavetable -> tone
# ---------------------------------------------------------------------

def infer(
    sample_name: str,
    dco_brightness: float,
    dco_richness: float,
    dco_oddenergy: float,
    dco_zcr: float,
) -> Tuple[Tuple[int, np.ndarray], plt.Figure]:
    # input wavetable
    waveform, _ = _load_wave(sample_name)
    batched_wav = waveform.unsqueeze(0)  # batch dim

    # Attributs
    attrs = {
        "dco_brightness": float(dco_brightness),
        "dco_richness": float(dco_richness),
        "dco_oddenergy": float(dco_oddenergy),
        "dco_zcr": float(np.clip(dco_zcr, 0.0, 1.0)),
    }

    # New wavetable generation
    wavetable = evaluator.model_eval(batched_wav, attrs)
    wavetable_np = wavetable.squeeze().detach().cpu().numpy()
    
    
    # Wavetable -> tone
    output_waveform = wavetable_to_tone(
        wavetable_np,
        duration_sec=TONE_DURATION_SEC,
        sample_rate=SAMPLE_RATE,
        frequency_hz=440.0,  # Fixed output pitch
    )

    # Plot
    fig, ax = plt.subplots(figsize=(8, 3))
    sample_axis = np.arange(wavetable_np.size)
    ax.plot(sample_axis, wavetable_np)
    ax.set_title("Generated Wavetable")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()

    return (SAMPLE_RATE, output_waveform), fig


# ---------------------------------------------------------------------
# update attribute sliders when sample changes
# ---------------------------------------------------------------------

def update_attributes(sample_name: str) -> Tuple[float, float, float, float]:
    return _default_attr_values(sample_name)


# ---------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------

def build_interface() -> gr.Blocks:
    sample_names = _list_predict_waves()
    default_sample = sample_names[0] if sample_names else ""
    default_attrs = _default_attr_values(default_sample) if default_sample else (0.5, 0.5, 0.5, 0.5)

    with gr.Blocks(title="Wavetable CVAE Demo") as demo:
        gr.Markdown(
            "# Wavetable CVAE Demo\n"
            "Select a wavetable example and tweak its conditioning attributes to generate a new tone."
        )

        with gr.Row():
            sample_dropdown = gr.Dropdown(
                choices=sample_names,
                value=default_sample,
                label="Input wavetable",
            )
            brightness_slider = gr.Slider(0.0, 1.0, value=default_attrs[0], step=0.01, label="Brightness")
            richness_slider = gr.Slider(0.0, 1.0, value=default_attrs[1], step=0.01, label="Richness")
            oddenergy_slider = gr.Slider(0.0, 1.0, value=default_attrs[2], step=0.01, label="Warmth")
            zcr_slider = gr.Slider(0.0, 1.0, value=default_attrs[3], step=0.01, label="Zero Crossing Rate")

        generate_button = gr.Button("Generate")
        audio_output = gr.Audio(label="Generated Audio", interactive=False)
        plot_output = gr.Plot(label="Waveform")

        inputs = [
            sample_dropdown,
            brightness_slider,
            richness_slider,
            oddenergy_slider,
            zcr_slider,
        ]
        outputs = [audio_output, plot_output]

        generate_button.click(fn=infer, inputs=inputs, outputs=outputs)

        sample_dropdown.change(
            fn=update_attributes,
            inputs=sample_dropdown,
            outputs=[brightness_slider, richness_slider, oddenergy_slider, zcr_slider],
        )

    return demo


# ---------------------------------------------------------------------
# load model and launch Gradio app
# ---------------------------------------------------------------------

evaluator, _ = load_model()
demo = build_interface()

if __name__ == "__main__":
    demo.launch(share=True)
