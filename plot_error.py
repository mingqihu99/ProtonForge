import matplotlib.pyplot as plt
import numpy as np

# Data definitions
data = [
    # layer, error, dtype
    ("rmsn", 0.00000000, "FP32"),
    ("rmsn", 0.00000000, "FP16"),
    ("rmsn", 0.00000000, "BF16"),
    ("mlp", 0.00000256, "FP32"),
    ("mlp", 0.00195312, "FP16"),
    ("mlp", 0.0156, "BF16"),
    ("gqa", 0.00000221, "FP32"),
    ("gqa", 0.00390625, "FP16"),
    ("gqa", 0.0156, "BF16"),
]

# --- styling ---
plt.rcParams["font.sans-serif"] = ["Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.style.use("seaborn-v0_8-whitegrid")  # Using a clean style base

layers = ["rmsn", "mlp", "gqa"]
layer_to_x = {l: i for i, l in enumerate(layers)}

# Professional color palette
colors = {
    "FP32": "#2E86AB",  # Steel Blue
    "FP16": "#F06543",  # Deep Orange
    "BF16": "#FFD700",  # Gold
}

jitter = {
    "FP32": -0.05,
    "FP16": 0.00,
    "BF16": 0.05,
}

fig, ax = plt.subplots(figsize=(9, 6), dpi=100)

for layer, error, dtype in data:
  x = layer_to_x[layer] + jitter[dtype]

  # Scatter point
  scatter = ax.scatter(
      x,
      error,
      color=colors[dtype],
      alpha=0.9,
      s=120,
      edgecolors="white",
      linewidth=1.5,
      label=dtype,
      zorder=3
  )

  # Add labels next to the points
  # Using scientific notation for very small errors
  label_text = f"{error:.2e}" if error < 0.01 and error > 0 else f"{error:.4f}"
  if error == 0:
    label_text = "0.0"

  ax.annotate(
      label_text,
      (x, error),
      textcoords="offset points",
      xytext=(5, 5) if jitter[dtype] > 0 else (-5, 5),
      ha="left" if jitter[dtype] > 0 else "right",
      fontsize=9,
      fontweight="bold",
      color="#444444",
      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7)
  )

# --- refined legend ---
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(
    by_label.values(),
    by_label.keys(),
    frameon=True,
    facecolor="white",
    framealpha=0.9,
    loc="upper left")

# --- Axis and aesthetics ---
ax.set_xticks(range(len(layers)))
ax.set_xticklabels([l.upper() for l in layers], fontsize=12, fontweight="bold")
ax.set_xlabel("Model Layers", fontsize=13, labelpad=10)
ax.set_ylabel("Accuracy Absolute Error", fontsize=13, labelpad=10)
ax.set_title("Layer-wise Numerical Errors",
             fontsize=16, fontweight="bold", pad=20)

# Cleaner grid and spines
ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Log scale might be better for small errors, but user values are very mixed (0 to 0.03)
# To keep it simple we stay linear unless asked, but we ensure 0 is visible.
ax.set_ylim(bottom=-0.0005)


plt.tight_layout()
plt.show()
