# Mamba in JAX

Code for state space models in JAX/Flax, from a single linear SSM (recurrent and convolutional views), through an S4-style deep SSM and the parallel associative scan, to Mamba's selective SSM, compared on the copying and selective-copying tasks.

---

## Theory

A dedicated lecture on state space models is coming soon on https://parag1604.github.io/notes. Until then, the background is in this [page](https://parag1604.github.io/notes/Probability-and-Linear-Algebra/Markov-Chains#towards-diffusion-models-and-state-space-models) (section "Towards diffusion models and state space models")

## Python Environment Setup

Use python >= 3.10 and install the requirements:

```bash
pip install -r requirements.txt
```

- JAX - arrays, automatic differentiation, `jit`, `vmap`, `lax.scan` and `lax.associative_scan` (install the CUDA build for an NVIDIA GPU)
- Flax (linen) - neural network layers
- Optax - the optimiser
- Numpy, Matplotlib - generating the tasks and plotting

On Apple silicon the experimental `jax-metal` plugin does not support all operations used here (e.g. the matrix inverse in the discretisation); if it is installed, run on the CPU with `JAX_PLATFORMS=cpu python ...`.

## Data

No dataset is needed. The tasks are generated on the fly (`utils.py`):

- copying: `n_tokens` data tokens at the start of the sequence, blanks, then `n_tokens` marker tokens; at the markers the model must output the data tokens in order;
- selective copying (Gu and Dao, 2023): the same, but the data tokens sit at random positions among `n_mem` blanks, so the model must decide from the content of each token whether to remember it.

## Running

The training scripts accept `--selective / --no-selective`, `--n_mem`, `--n_tokens`, `--d_model`, `--n_layers`, `--d_state`, `--steps`, `--batch_size` and `--lr`, and print the loss and the accuracy on the marker positions (train batch and 512 fresh test sequences). Quick check (about 10 seconds on a CPU):

```bash
python 04_mamba.py --n_mem 16 --steps 100 --print_every 50 --d_model 32
```

### 1. SSM basics

```bash
python 01_ssm_basics.py
```

A linear SSM `x' = Ax + Bu, y = Cx`, discretised with the bilinear method, computed two ways: as a recurrence (`jax.lax.scan`, the RNN view) and as a causal convolution with the kernel `K = (C Bb, C Ab Bb, C Ab^2 Bb, ...)` via the FFT (the CNN view). The script checks that both give the same output (max difference about 1e-6), shows that a random `A` gives an unstable discrete system (largest eigenvalue of `Ab` above 1) while the HiPPO matrix does not, and saves the kernels to `outputs/ssm_kernels.png`.

### 2. A deep SSM (S4-style)

```bash
python 02_s4.py --no-selective   # copying
python 02_s4.py                  # selective copying
```

New: the SSM becomes a trainable layer (HiPPO-initialised `A`, learned `B`, `C`, `D` and step size), copied across feature channels with `nn.vmap`, inside residual blocks with LayerNorm, GELU and a gated linear unit. Training uses the convolutional view. The model is linear time-invariant: it applies the same filter to every input. With the defaults (sequence length 36, 3000 steps, about 3 minutes on a CPU) it reached 96.7% test accuracy on copying and 92.4% on selective copying.

### 3. Parallel scan

```bash
python 03_parallel_scan.py
```

New: Mamba's recurrence `h_t = a_t h_{t-1} + b_t` changes at every step, so there is no convolution kernel. Because combining two steps, `(a1, b1) o (a2, b2) = (a1 a2, a2 b1 + b2)`, is associative, the whole recurrence can be computed in O(log L) parallel rounds. The script implements the Hillis-Steele scan by hand, compares it with `jax.lax.scan` (sequential) and `jax.lax.associative_scan` (Blelloch), checks that all three agree, and times them. On a CPU the sequential scan is fastest (L = 4096, 256 channels: 1.2 ms sequential, 2.0 ms Blelloch, 64 ms for the unoptimised Hillis-Steele); the parallel scans pay off on GPUs and TPUs.

### 4. Mamba

```bash
python 04_mamba.py               # selective copying
python 04_mamba.py --no-selective
```

New: the selective SSM. The step size `Delta`, `B` and `C` are computed from the current input, so the model can choose to write a token into its state (large `Delta`) or to ignore it (small `Delta`). `A` is diagonal and negative, the recurrence is computed with `jax.lax.associative_scan`, and the SSM sits in a Mamba block (input projection, short causal depthwise convolution, SiLU, selective SSM, SiLU gate, output projection). With the defaults it reached 100% test accuracy on selective copying within 1000 steps (about 8 minutes for 3000 steps on a CPU), against 92.4% for the S4-style model of step 2. Try longer sequences (`--n_mem 128` or more) to widen the gap between the two models.
