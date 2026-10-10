# Training Neural Networks

The main purpose of PICT's differentiability is **simulation-coupled learning**, sometimes called *solver-in-the-loop* training. A neural network \(G_\theta\) runs inside the simulation: in every step it looks at the current flow state and predicts a correcting force \(\mathbf{S}_\theta\), which enters the momentum equation as a velocity source. The simulation is unrolled for \(n\) steps, a loss is computed, and the gradient reaches \(\theta\) through all \(n\) solver steps. The network thus learns how its corrections affect the *future* flow, not just the next frame.

Typical uses are **learned correctors** that make coarse simulations behave like fine ones, and **learned sub-grid-scale (SGS) models** for turbulence.

## The training loop

The pattern from `learning_sample.py` (network mode), condensed:

```python
import torch, PISOtorch, PISOtorch_simulation

net = ForcingNet().to(cuda_device)                 # any torch.nn.Module: velocity -> forcing
optimizer = torch.optim.Adam(net.parameters(), lr=4e-4)

def pfn_set_forcing(domain, **kwargs):              # called at the start of every PISO step
    block = domain.getBlock(0)
    block.setVelocitySource(net(block.velocity))    # per-cell forcing [1, dims, (z,) y, x]
    domain.UpdateDomainData()

for opt_step in range(num_steps):
    opt_domain = domain.Copy()
    opt_domain.PrepareSolve()
    sim = PISOtorch_simulation.Simulation(opt_domain, prep_fn={"PRE": [pfn_set_forcing]},
                                          substeps=1, time_step=0.1, non_orthogonal=False,
                                          differentiable=True, log_images=False)
    sim.run(n_unroll, log_domain=False)

    loss = loss_fn(opt_domain, reference)           # e.g. MSE to a reference velocity
    loss.backward()                                  # through n_unroll solver steps into the network
    optimizer.step()
    optimizer.zero_grad()
    opt_domain.Detach()                              # mandatory: avoids leaking GPU memory
```

## Multi-block convolutions

A standard `torch.nn.Conv2d/3d` applied to each block separately would zero-pad at block interfaces and create artifacts along every connection. **`lib.modules.multiblock_cnn.MultiblockConv`** instead pads each block with cells from its **connected neighbor blocks**, including periodic wrap-around, rotated or flipped connections, and diagonal neighbors. Convolutions are therefore seamless across the multi-block domain:

```python
from lib.modules.multiblock_cnn import MultiblockConv, MultiblockReLU

conv = MultiblockConv(domain, fixed_boundary_padding="ZERO",
                      in_channels=2, out_channels=16, kernel_size=3,     # kernel_size: 3, 5 or 7
                      device=cuda_device, dtype=dtype)
outputs = conv([block.velocity for block in domain.getBlocks()])         # list in, list out (one tensor per block)
outputs = MultiblockReLU()(outputs)
```

- The layer works in 1D, 2D and 3D. Padding is `"same"` (kernel_size // 2), and dilation is not supported.
- Cells beyond fixed boundaries (walls, inflow, outflow) are zero-padded.
- `MultiblockConv_naive` pads every block independently. It is meant for comparisons and for 1×1 convolutions, which need no padding.

The channel-flow SGS model (`channel_flow_learning.py`, abridged) stacks seven `MultiblockConv` layers with ReLU activations and a final 1×1 convolution:

```python
class CorrectionNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, seed=34512468, use_multiblock_conv=True, domain=None):
        torch.manual_seed(seed)
        super().__init__()
        filters = [in_channels, 8, 64, 64, 32, 16, 8, 4]
        self.conv_layers = torch.nn.ModuleList([
            MultiblockConv(domain=domain, fixed_boundary_padding="ZERO", in_channels=filters[i], out_channels=filters[i+1],
                           kernel_size=kernel_size, stride=stride, device=cuda_device, dtype=dtype)
            for i in range(len(filters) - 1)])
        self.conv_final = torch.nn.Conv3d(filters[-1], out_channels, 1, 1, padding="same", device=cuda_device, dtype=dtype)

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = torch.nn.functional.relu(conv_layer([x])[0])
        return self.conv_final(x)
```

## Loss functions

Because gradients flow through the solver, the loss can be anything differentiable that is computed from the simulated states:

- **Supervised trajectory losses.** Mean squared error between the simulated velocity and a reference, e.g. a down-sampled high-resolution simulation, at some or all unrolled steps.
- **Statistics losses.** The flow statistics are matched instead of individual snapshots. The channel-flow trainer accumulates differentiable online statistics over the unrolled steps (`lib.data.online_statistics`) and compares them to DNS reference profiles. It matches the mean velocity components (streamwise \(U^+\) with weight 1, the other two with weight 0.5), the fluctuations \(\overline{u'u'}^+, \overline{v'v'}^+, \overline{w'w'}^+\) and the Reynolds shear stress \(\overline{u'v'}^+\). Statistics losses require **no paired high-resolution data**, only reference statistics.
- **Regularizers.** For example a penalty on the forcing magnitude \(\lambda_S\lVert\mathbf{S}_\theta\rVert^2\). The paper also introduces a gradient modification that steers the network output towards divergence-free forcings (the `add_forcing_divergence_pressure_grad` code path of the channel-flow trainer).

## Training strategies

These practices are used in the paper and the training scripts:

- **Unroll length is the key hyperparameter.** Longer unrolls teach long-term stability, but cost more memory and time. Curricula help: the paper uses 4 → 8 → 16 steps for the vortex street and 10 → 30 → 40 steps for the backward-facing step.
- **Randomized warm-up.** The channel flow trains on 12 differentiable steps that follow a random number of *non-differentiable* pre-steps, with the warm-up range growing over the training phases.
- **Approximate gradients first.** Excluding the backward linear solves ([gradient paths](differentiable.md#gradient-paths-trading-accuracy-for-speed)) speeds up early training. They can be re-enabled for fine-tuning.
- **Batch size 1.** A domain holds exactly one sample. Iterate over initial conditions instead.

## Next steps

- [Learning a Forcing](../examples/learning_forcing.md): the complete minimal example.
- [Turbulent Channel Flow SGS Model](../examples/turbulent_channel_flow.md): the full statistics-based SGS training from the paper.
- [Results from the Paper](../background/results.md): learned correctors for a vortex street, a backward-facing step and 3D channel flow.
