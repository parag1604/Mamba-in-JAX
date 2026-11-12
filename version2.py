"""
Version 2: structured state space models (S4D, a diagonal S4).

What is NEW compared to version1.py
-----------------------------------
* Structure. In version 1, A was a dense N x N matrix: building the kernel
  K_l = C Ab^l Bb costs O(L N^2) (or worse), and that is per channel, per
  training step. If A is DIAGONAL, A = diag(a_1, ..., a_N) with complex a_n,
  every mode evolves on its own:
        Ab_n = exp(Delta a_n),   Bb_n = (Ab_n - 1) / a_n * B_n   (ZOH)
        K_l  = sum_n C_n Bb_n Ab_n^l
  which is a (Vandermonde) matrix-vector product: O(N L) work, trivially
  parallel. The modes come in complex-conjugate pairs, so we store N/2 of
  them and take 2 Re(...) (Gu, Gupta et al., 2022, "S4D").
  S4 proper keeps the HiPPO matrix, which is not diagonalisable in a stable
  way, as "diagonal plus low rank" (DPLR) and computes its kernel with a
  Cauchy kernel and the Woodbury identity; S4D shows that the diagonal part
  alone works about as well (see the README).
* Training in CONVOLUTION mode: compute K once per step, then convolve the
  whole sequence with the FFT, O(L log L), all positions in parallel.
* Inference in RECURRENT mode (`decode=True`): the same parameters run as an
  RNN with a constant-size state (N/2 complex numbers per channel), one
  token at a time. The script checks that both modes give the same outputs
  and times them.
* The deep network around the layer is Parag's original S4 code: H copies
  of the SSM (one per channel, `cloneLayer`), `SequenceBlock` (SSM, GELU,
  gated linear unit, LayerNorm, residual) and `StackedModel`. The only
  addition is the 'cache' collection that holds the recurrent state.
* The model is linear and TIME-INVARIANT: the same kernel K is applied to
  every input. It solves copying (a fixed shift) but cannot do selective
  copying, which needs content-dependent memory. Version 3 fixes this.

Run:
    python version2.py --task copy
    python version2.py --task selective
Initialisation: decay rates and frequencies of the modes are drawn at
random here (`--init random`); version 5 replaces them with the HiPPO-derived
S4D-Lin values (`--init lin`).

Layers adapted from "The Annotated S4" by Sasha Rush and Sidd Karamcheti
(https://srush.github.io/annotated-s4/, MIT licence), as in Parag's original
version of this repository.
"""

import math

import jax
import jax.numpy as np
from flax import linen as nn

from utils import (default_parser, empty_cache, generation_benchmark, plot_generation,
                   plot_style, recurrent_vs_parallel, save_history, savefig,
                   seq_len, timeit, train)
from version1 import K_scan, causal_convolution, discretize_zoh


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    """log Delta uniform in [log dt_min, log dt_max] (Parag's original)."""
    def init(key, shape):
        return jax.random.uniform(key, shape) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)

    return init


def s4d_A_init(kind):
    """Initial diagonal A = -exp(log_A_real) + i A_imag, N2 = N/2 modes.
    'random': decay rates U(0.1, 1), frequencies U(0, pi N2), no structure;
    'lin'   : S4D-Lin, a_n = -1/2 + i pi n  (from HiPPO, see version 5)."""
    def log_A_real(key, shape):
        if kind == 'lin':
            return np.log(0.5) * np.ones(shape)
        return np.log(jax.random.uniform(key, shape, minval=0.1, maxval=1.0))

    def A_imag(key, shape):
        if kind == 'lin':
            return math.pi * np.arange(shape[0], dtype=np.float32)
        return jax.random.uniform(key, shape) * math.pi * shape[0]

    return log_A_real, A_imag


def s4d_kernel(dtA, CB, L):
    """K_l = 2 Re sum_n CB_n exp(l * dtA_n): a Vandermonde product, O(N L)."""
    V = np.exp(dtA[:, None] * np.arange(L))          # (N2, L)
    return 2 * (CB @ V).real


class S4DLayer(nn.Module):
    """One diagonal SSM, mapping a 1-D sequence u (L,) to y (L,)."""
    N: int
    l_max: int = 0          # unused: the kernel takes the input's length
    decode: bool = False
    A_init: str = 'random'

    def setup(self):
        N2 = self.N // 2
        init_re, init_im = s4d_A_init(self.A_init)
        self.log_A_real = self.param('log_A_real', init_re, (N2,))
        self.A_imag = self.param('A_imag', init_im, (N2,))
        # B is fixed to 1 (as in S4D); C is complex, stored as (real, imag)
        self.C = self.param('C', nn.initializers.normal(0.5 ** 0.5), (N2, 2))
        self.D = self.param('D', nn.initializers.ones, (1,))
        self.log_step = self.param('log_step', log_step_initializer(), (1,))
        if self.decode:
            # the recurrent state: N/2 complex numbers, whatever the length
            self.h = self.variable('cache', 'h', np.zeros, (N2,), np.complex64)

    def discrete(self):
        A = -np.exp(self.log_A_real) + 1j * self.A_imag
        dtA = np.exp(self.log_step) * A
        dA = np.exp(dtA)                       # ZOH
        dB = (dA - 1.0) / A                    # ZOH, with B = 1
        C = self.C[:, 0] + 1j * self.C[:, 1]
        return dtA, dA, dB, C

    def kernel(self, L):
        dtA, _, dB, C = self.discrete()
        return s4d_kernel(dtA, C * dB, L)

    def __call__(self, u):
        dtA, dA, dB, C = self.discrete()
        if not self.decode:   # CNN mode, for training
            K = s4d_kernel(dtA, C * dB, u.shape[0])
            return causal_convolution(u, K) + self.D * u

        # RNN mode, for inference: h_k = dA h_{k-1} + dB u_k, y_k = 2Re(C h_k)
        def step(h, u_k):
            h = dA * h + dB * u_k
            return h, 2 * (C @ h).real

        h, y = jax.lax.scan(step, self.h.value, u)
        if self.is_mutable_collection('cache'):
            self.h.value = h
        return y + self.D * u


# ------------------------- Parag's original S4 network (with a cache)
def cloneLayer(layer):
    # H copies of the layer, one per feature channel (axis 1 of (L, H))
    return nn.vmap(
        layer,
        in_axes=1,
        out_axes=1,
        variable_axes={"params": 1, "cache": 1},
        split_rngs={"params": True},
    )


class SequenceBlock(nn.Module):
    layer_cls: nn.Module
    layer: dict  # Hyperparameters of inner layer
    dropout: float
    d_model: int
    prenorm: bool = True
    glu: bool = True
    training: bool = True

    def setup(self):
        self.seq = self.layer_cls(**self.layer)
        self.norm = nn.LayerNorm()
        self.out = nn.Dense(self.d_model)
        if self.glu:
            self.out2 = nn.Dense(self.d_model)
        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

    def __call__(self, x):
        skip = x
        if self.prenorm:
            x = self.norm(x)
        x = self.seq(x)
        x = self.drop(nn.gelu(x))
        if self.glu:
            x = self.out(x) * jax.nn.sigmoid(self.out2(x))
        else:
            x = self.out(x)
        x = skip + self.drop(x)
        if not self.prenorm:
            x = self.norm(x)
        return x


class StackedModel(nn.Module):
    layer_cls: nn.Module
    layer: dict  # Extra arguments to pass into layer constructor
    vocab_size: int
    d_model: int
    n_layers: int
    prenorm: bool = True
    dropout: float = 0.0
    training: bool = False

    def setup(self):
        self.encoder = nn.Embed(self.vocab_size, self.d_model)
        self.decoder = nn.Dense(self.vocab_size)
        self.layers = [
            SequenceBlock(
                layer_cls=self.layer_cls,
                layer=self.layer,
                prenorm=self.prenorm,
                d_model=self.d_model,
                dropout=self.dropout,
                training=self.training,
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, x):
        # x: (L,) token ids of one sequence -> (L, vocab) logits
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x)
        return self.decoder(x)


# add the batch dimension: (B, L) -> (B, L, vocab), parameters shared,
# one recurrent state ('cache') per sequence
BatchStackedModel = nn.vmap(
    StackedModel,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "dropout": None, "cache": 0,
                   "intermediates": 0},
    split_rngs={"params": False, "dropout": True},
)

S4DBank = cloneLayer(S4DLayer)


def make_s4d_model(args, decode=False, init=None):
    return BatchStackedModel(
        layer_cls=S4DBank,
        layer={'N': args.d_state, 'decode': decode,
               'A_init': init or args.init},
        vocab_size=args.n_data + 2,
        d_model=args.d_model,
        n_layers=args.n_layers,
    )


# -------------------------------------------------------------- demos
def demo_kernel_cost():
    """Dense A (version 1) vs diagonal A (this version): kernel cost."""
    print('\n== Kernel cost: dense A vs diagonal A (H = 64 channels) ==')
    L, H = 1024, 64
    Ns = [8, 16, 32, 64, 128, 256]
    t_dense, t_diag = [], []
    for N in Ns:
        k = jax.random.split(jax.random.PRNGKey(0), 3)
        A = jax.random.normal(k[0], (H, N, N)) / N ** 0.5 - np.eye(N)
        B = jax.random.normal(k[1], (H, N, 1))
        C = jax.random.normal(k[2], (H, 1, N))

        @jax.jit
        def dense(A, B, C):
            def one(A, B, C):
                return K_scan(*discretize_zoh(A, B, C, 0.01), L)
            return jax.vmap(one)(A, B, C)

        dtA = (-0.5 + 1j * np.arange(N // 2)) * 0.01 * np.ones((H, 1))
        CB = np.ones((H, N // 2), np.complex64)
        diag = jax.jit(jax.vmap(lambda a, cb: s4d_kernel(a, cb, L)))
        t_dense.append(1e3 * timeit(dense, A, B, C, repeats=3))
        t_diag.append(1e3 * timeit(diag, dtA, CB, repeats=10))
        print(f'N = {N:4d}: dense {t_dense[-1]:9.2f} ms | diagonal '
              f'{t_diag[-1]:7.2f} ms')
    plt = plot_style()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.loglog(Ns, t_dense, 'o-', label='dense A: expm + L mat-vec products')
    ax.loglog(Ns, t_diag, 's-', label='diagonal A: Vandermonde product')
    ax.set_xlabel('state size N')
    ax.set_ylabel('time to build the kernels (ms)')
    ax.set_title(f'Kernel for L = {L}, H = {H} channels (CPU)')
    ax.legend(fontsize=8)
    savefig(fig, 'v2_kernel_cost.png')
    plt.close(fig)


def demo_modes(model, decode_model, params, args, prefix):
    """Recurrent == parallel, training vs inference timings."""
    rng = jax.random.PRNGKey(1)
    x = jax.random.randint(rng, (8, seq_len(args)), 1, args.n_data + 2)
    diff, scale = recurrent_vs_parallel(model, decode_model, params, x)
    print(f'\nmax |parallel (training) logits - recurrent (decode) logits| = '
          f'{diff:.1e}  (logits up to {scale:.1f})')

    # processing a whole batch of sequences: convolution vs recurrence
    print('Whole sequences (batch 16): parallel mode vs recurrent mode')
    lengths = [64, 256, 1024, 4096]
    par_fn = jax.jit(lambda x: model.apply({'params': params}, x))
    t_full = []
    for L in lengths:
        x = jax.random.randint(rng, (16, L), 1, args.n_data + 2)
        cache = empty_cache(decode_model, x)
        rec_fn = jax.jit(lambda x, c: decode_model.apply(
            {'params': params, 'cache': c}, x, mutable=['cache'])[0])
        tp = 1e3 * timeit(par_fn, x, repeats=3)
        tr = 1e3 * timeit(rec_fn, x, cache, repeats=3)
        t_full.append((tp, tr))
        print(f'  L = {L:5d}: parallel {tp:8.1f} ms | recurrent scan '
              f'{tr:8.1f} ms')

    res = generation_benchmark(model, decode_model, params,
                               [64, 256, 1024, 4096, 16384],
                               args.n_data + 2)
    print('Generating one token after a context of length t:')
    for L, tp, tr in zip(res['lengths'], res['parallel_ms'],
                         res['recurrent_ms']):
        print(f'  t = {L:6d}: re-run in parallel mode {tp:8.2f} ms | one '
              f'recurrent step {tr:6.3f} ms')
    print(f'  recurrent state: {res["state_bytes"]} bytes, the same for '
          f'every t')
    plot_generation(res, f'{prefix}: cost of the next token',
                    f'{prefix.lower()}_generation_cost.png')
    return dict(full=dict(lengths=lengths, times=t_full), gen=res,
                equality=diff)


def plot_kernels(model, params, args, name):
    """The learned kernels of a few channels of the first layer."""
    bank = params['layers_0']['seq']
    L = seq_len(args)
    layer = S4DLayer(N=args.d_state, A_init=args.init)
    plt = plot_style()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for c in range(4):
        # the vmapped bank stores channel c at axis 1 of every parameter
        p = jax.tree_util.tree_map(lambda a: np.take(a, c, axis=1), bank)
        K = layer.apply({'params': p}, L, method=S4DLayer.kernel)
        ax.plot(K / np.abs(K).max() + 2.2 * c, lw=1)
    ax.axvline(args.n_mem, color='k', ls=':', lw=1)
    ax.set_xlabel('lag l')
    ax.set_yticks([])
    ax.set_title(f'Learned S4D kernels (4 channels of layer 1, '
                 f'{args.task} task)')
    savefig(fig, name)
    plt.close(fig)


def main(args):
    model = make_s4d_model(args)
    decode_model = make_s4d_model(args, decode=True)
    if not args.skip_demo:
        demo_kernel_cost()
    if args.skip_train:
        return
    params, hist = train(model, args, f'S4D ({args.init} init)')
    save_history(hist, args.save_history, model='S4D', task=args.task,
                 init=args.init)
    if not args.skip_demo:
        plot_kernels(model, params, args, f'v2_kernels_{args.task}.png')
        demo_modes(model, decode_model, params, args, 'V2')


if __name__ == '__main__':
    parser = default_parser('S4D (diagonal structured SSM) on copying tasks')
    parser.add_argument('--init', default='random', choices=['random', 'lin'])
    main(parser.parse_args())
