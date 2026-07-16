import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("benchmark_results.csv").set_index("case")
snr_cases = ["clean", "snr_20dB", "snr_10dB", "snr_5dB", "snr_0dB", "snr_-5dB", "pure_noise"]
xlabels = ["clean", "20dB", "10dB", "5dB", "0dB", "-5dB", "noise"]

# BNP-ish palette
C = {"harm": "#00915A", "flat": "#C4262E", "snrw": "#1B3A6B", "snrp": "#E8A33D"}

def norm(s):
    s = s.astype(float); return (s - s.min()) / (s.max() - s.min() + 1e-9)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))

sub = df.loc[snr_cases]
ax[0].plot(xlabels, norm(sub["harmonicity"]), "-o", color=C["harm"], lw=2, label="harmonicity (↑cleaner)")
ax[0].plot(xlabels, 1-norm(sub["spectral_flatness"]), "-s", color=C["flat"], lw=2, label="1 - spectral_flatness")
ax[0].plot(xlabels, norm(sub["snr_percentile"]), "-^", color=C["snrp"], lw=2, label="snr_percentile")
ax[0].plot(xlabels, norm(sub["snr_wada"].clip(lower=-20)), "-d", color=C["snrw"], lw=2, label="snr_wada")
ax[0].set_title("Monotonic response to added noise", fontsize=12, fontweight="bold")
ax[0].set_ylabel("normalized score (1 = cleanest)")
ax[0].set_xlabel("degradation level")
ax[0].legend(fontsize=8, frameon=False); ax[0].grid(alpha=.25)

# special cases: which metric flags what
special = ["clean", "clipped", "near_silence", "pure_noise"]
metrics = ["harmonicity", "spectral_flatness", "clipping_ratio", "vad_speech_ratio"]
vals = df.loc[special, metrics].astype(float).T.values
im = ax[1].imshow(vals, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
ax[1].set_xticks(range(len(special))); ax[1].set_xticklabels(special, rotation=20, fontsize=9)
ax[1].set_yticks(range(len(metrics))); ax[1].set_yticklabels(metrics, fontsize=9)
for i in range(len(metrics)):
    for j in range(len(special)):
        ax[1].text(j, i, f"{vals[i,j]:.2f}", ha="center", va="center", fontsize=9)
ax[1].set_title("Who catches what (edge cases)", fontsize=12, fontweight="bold")

plt.tight_layout(); plt.savefig("metric_comparison.png", dpi=130, bbox_inches="tight")
print("saved")
