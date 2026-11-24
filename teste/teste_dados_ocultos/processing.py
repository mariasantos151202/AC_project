from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfiltfilt
import tensorflow as tf


@dataclass
class Config:
    data_dir: Path = Path("../CirCor_test_students")
    fs_target: int = 1000
    band: tuple = (25, 400)
    win_sec: float = 5.0
    hop_sec: float = 2.5
    # --- log-mel / CNN 2D ---
    n_mels: int = 40
    frame_len: int = 128
    frame_step: int = 32
    # --- RNN ---
    n_mfcc: int = 13
    rnn_frame_step: int = 64
    seed: int = 42

    @property
    def wav_dir(self):
        return self.data_dir / "test_data"

    @property
    def csv_path(self):
        return self.data_dir / "test_data_participants.csv"


CFG = Config()
print("CSV existe:", CFG.csv_path.exists(), "| pasta wav existe:", CFG.wav_dir.exists())


def load_recording(path, cfg: Config = CFG) -> np.ndarray:
    """Lê .wav, reamostra para fs_target, filtra e normaliza (z-score)."""
    fs, x = wavfile.read(path)
    x = x.astype(np.float32)
    x = x / (np.abs(x).max() + 1e-8)
    g = np.gcd(int(fs), cfg.fs_target)
    x = resample_poly(x, cfg.fs_target // g, int(fs // g)).astype(np.float32)
    sos = butter(4, cfg.band, btype="bandpass", fs=cfg.fs_target, output="sos")
    x = sosfiltfilt(sos, x).astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-8)


def to_windows(x: np.ndarray, cfg: Config = CFG) -> np.ndarray:
    """Janelas de win_sec com passo hop_sec (faz padding se for mais curto)."""
    w, h = int(cfg.win_sec * cfg.fs_target), int(cfg.hop_sec * cfg.fs_target)
    if len(x) < w:
        x = np.pad(x, (0, w - len(x)))
    return np.stack([x[s : s + w] for s in range(0, len(x) - w + 1, h)])


def build_test_waveform_dataset(pids, cfg: Config = CFG):
    """Processa TODAS as gravações de teste (<pid>_<local>[_n].wav) sem precisar de labels."""
    waves, rows = [], []
    for pid in pids:
        wavs = sorted(cfg.wav_dir.glob(f"{pid}_*.wav"))
        assert len(wavs) > 0, f"Sem áudios para o paciente {pid}"
        for wav in wavs:
            loc = wav.stem.split("_")[1]
            win = to_windows(load_recording(wav, cfg), cfg)
            waves.append(win)
            rows += [dict(pid=pid, loc=loc, rec=wav.stem, y=-1)] * len(win)
    return np.concatenate(waves), pd.DataFrame(rows)


def _batched(fn, waves, batch=512):
    return np.concatenate(
        [fn(waves[i : i + batch]) for i in range(0, len(waves), batch)]
    )


def _zscore(a, axes):
    m = a.mean(axis=axes, keepdims=True)
    s = a.std(axis=axes, keepdims=True)
    return (a - m) / (s + 1e-6)


def _log_mel_tf(w, cfg: Config, frame_step: int):
    n_bins = cfg.frame_len // 2 + 1
    stft = tf.signal.stft(w, cfg.frame_len, frame_step, fft_length=cfg.frame_len)
    mel_w = tf.signal.linear_to_mel_weight_matrix(
        cfg.n_mels, n_bins, cfg.fs_target, cfg.band[0], cfg.band[1]
    )
    return tf.math.log(tf.tensordot(tf.abs(stft) ** 2, mel_w, 1) + 1e-6)


def features_cnn2d(waves, cfg: Config = CFG) -> np.ndarray:
    """CNN 2D: log-mel -> (N, frames, n_mels, 1), z-score por janela."""
    def f(w):
        lm = _log_mel_tf(w, cfg, cfg.frame_step).numpy()
        return _zscore(lm, (1, 2))[..., None].astype(np.float32)

    return _batched(f, waves)


def features_rnn(waves, kind="mfcc", delta=True, cfg: Config = CFG) -> np.ndarray:
    """RNN: sequência (N, passos, features)."""
    step = cfg.rnn_frame_step

    def f(w):
        lm = _log_mel_tf(w, cfg, step)
        if kind == "logmel":
            return _zscore(lm.numpy(), (1,)).astype(np.float32)
        mf = tf.signal.mfccs_from_log_mel_spectrograms(lm)[..., : cfg.n_mfcc].numpy()
        feats = [mf]
        if delta:
            d1 = np.gradient(mf, axis=1)
            feats += [d1, np.gradient(d1, axis=1)]
        return _zscore(np.concatenate(feats, axis=-1), (1,)).astype(np.float32)

    return _batched(f, waves)


# --- Execução para os dados ocultos de teste ---
test_meta = pd.read_csv(CFG.csv_path, dtype={"Patient ID": str})
test_pids = test_meta["Patient ID"].tolist()

waves, info = build_test_waveform_dataset(test_pids, CFG)

out = Path("cache")
out.mkdir(exist_ok=True)
info.to_csv(out / "hidden_info.csv", index=False)
np.savez_compressed(
    out / "hidden.npz",
    y=info["y"].values,
    cnn2d=features_cnn2d(waves, CFG),
    rnn=features_rnn(waves, "mfcc", True, CFG),
)
print(f"hidden: {len(test_pids)} pacientes, {len(waves)} janelas guardadas com sucesso!")