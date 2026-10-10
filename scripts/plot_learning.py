"""Loss and learned forcing for the learning_sample.py constant-forcing optimization."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import vizlib as V

SRC, OUT = sys.argv[1], sys.argv[2]
h = json.load(open(os.path.join(SRC, "learning.json")))["CONSTANT"]
loss = np.array(h["loss"]); fx = np.array([f[0] for f in h["forcing"]]); steps = np.arange(len(loss))
V.chart_rc()
for mode in ("light", "dark"):
    c = V.CHART[mode]
    matplotlib.rcParams["axes.titlecolor"] = c["ink"]
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.3), dpi=170)
    fig.patch.set_facecolor(c["surface"])
    # loss
    ax = axs[0]; V.chart_axes(ax, mode)
    ax.semilogy(steps, loss, color=c["series"][0], lw=2)
    ax.plot(steps[[0, -1]], loss[[0, -1]], "o", ms=8, color=c["series"][0], mec=c["surface"], mew=2)
    ax.set_title("Loss (sum of squared velocity errors)")
    ax.set_xlabel("optimization step"); ax.set_xlim(-0.5, steps[-1] + 0.5)
    ax.annotate("%.1e" % loss[-1], (steps[-1], loss[-1]), xytext=(-10, -2), textcoords="offset points",
                ha="right", va="top", color=c["ink2"], fontsize=9)
    ax.annotate("%.1f" % loss[0], (steps[0], loss[0]), xytext=(10, 0), textcoords="offset points",
                ha="left", va="center", color=c["ink2"], fontsize=9)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    # forcing
    ax = axs[1]; V.chart_axes(ax, mode)
    ax.axhline(1.0, color=c["muted"], lw=1)
    ax.text(0.2, 1.0, "target  $f_x = 1$", color=c["ink2"], fontsize=9, va="bottom", ha="left")
    ax.plot(steps, fx, color=c["series"][0], lw=2)
    ax.plot(steps[[0, -1]], fx[[0, -1]], "o", ms=8, color=c["series"][0], mec=c["surface"], mew=2)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_title("Learned forcing $f_x$")
    ax.set_xlabel("optimization step"); ax.set_ylim(-0.05, 1.12); ax.set_xlim(-0.5, steps[-1] + 0.5)
    ax.annotate("%.5f" % fx[-1], (steps[-1], fx[-1]), xytext=(0, -12), textcoords="offset points",
                ha="right", va="top", color=c["ink2"], fontsize=9)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(os.path.join(OUT, "learning_forcing_%s.png" % mode), facecolor=c["surface"])
    plt.close(fig)
print("ok")
