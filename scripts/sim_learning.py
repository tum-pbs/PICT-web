"""Replicates learning_sample.py (learn a forcing through the differentiable solver) and records
loss / forcing histories for plotting."""
import os, sys, json
import numpy as np
import torch
from lib.util.logging import setup_run, get_logger
import learning_sample as ls

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
setup_run(OUT, name="log")
LOG = get_logger("learn")
hist = {}

for kind, opt_steps, lr in [("CONSTANT", 20, 1e-2), ("NETWORK", 100, 4e-4)]:
    domain = ls.make_forcing_domain(16, 16, 1, torch.float32)
    trainer = ls.ForcingTrainer(os.path.join(OUT, kind), domain, forcing_type=kind, target_type="FORCING",
                                lr=lr, sim_steps=3, time_step=0.1, log=LOG)
    trainer.make_base_divergence_free()
    trainer.simulate_target()
    forcing = []
    for step in range(opt_steps):
        if kind == "CONSTANT":
            forcing.append(trainer.var_forcing.detach().cpu().numpy()[0].tolist())
        trainer.optimization_step(step, log_images=False)
    hist[kind] = dict(loss=[float(l) for l in trainer.losses], forcing=forcing)
    print(kind, "loss first %.4e last %.4e" % (hist[kind]["loss"][0], hist[kind]["loss"][-1]), flush=True)

json.dump(hist, open(os.path.join(OUT, "learning.json"), "w"), indent=1)
print("saved")
