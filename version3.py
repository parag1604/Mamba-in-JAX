"""
Version 3: selective state space models (S6, the SSM inside Mamba).

What is NEW compared to version2.py
-----------------------------------
* Selection (Gu and Dao, 2023, https://arxiv.org/abs/2312.00752). The step
  size Delta and the matrices B and C stop being fixed parameters and become
  FUNCTIONS OF THE CURRENT INPUT u_t (small linear layers):
        Delta_t = softplus(W_Delta (W_r u_t) + b_Delta),
        B_t = W_B u_t,   C_t = W_C u_t,
        h_t = exp(Delta_t A) h_{t-1} + Delta_t B_t u_t,
        y_t = C_t h_t + D u_t
  (A stays a fixed, diagonal, negative parameter; the A term is discretised
  with ZOH and the B term with the simpler Euler rule, as in Mamba).
  A large Delta_t lets u_t overwrite the state ("remember this token"); a
  small one leaves the state almost unchanged ("ignore this token"). That is
  the content-based selection that selective copying needs, and that a
  time-invariant model (versions 1-2) cannot express.
* The price: the system is no longer time invariant, so there is NO single
  convolution kernel and no FFT. Training instead uses the PARALLEL
  ASSOCIATIVE SCAN: one step of the recurrence is the affine map
  h -> a_t h + b_t, and composing two steps,
        (a1, b1) o (a2, b2) = (a1 a2,  a2 b1 + b2),
  is associative, so all h_t can be computed in O(log L) parallel rounds
  (Blelloch: `jax.lax.associative_scan`). Inference is still RECURRENT, one
  token at a time from a constant-size state.
* Everything else is unchanged: the selective SSM simply replaces the S4D
  bank inside Parag's SequenceBlock / StackedModel from version 2, so the
  comparison with version 2 isolates the effect of selection.

A note on names: S4 is the STRUCTURED state space model (structured A,
time-invariant); its selective, input-dependent version is called S6 in the
Mamba paper ("S4 + selection + scan").

Run:
    python version3.py --task selective
    python version3.py --task copy

The Hillis-Steele scan follows the idea of Franz A. Heinsen's
torch_parallel_scan (https://github.com/glassroom/torch_parallel_scan, MIT),
as in Parag's original 03_parallel_scan.py.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from utils import (default_parser, plot_style, save_history, savefig,
                   seq_len, timeit, train, make_batch)
from version2 import BatchStackedModel, demo_modes


# ----------------------------------------------------- the parallel scan
def sequential_scan(a, b):
    """h_t = a_t h_{t-1} + b_t, h_{-1} = 0; a, b: (L, ...)."""
    def step(h, ab):
        a_t, b_t = ab
        h = a_t * h + b_t
        return h, h
    _, hs = jax.lax.scan(step, jnp.zeros_like(b[0]), (a, b))
    return hs


def combine(left, right):
    """The associative operator: first apply `left`, then `right`."""
    a1, b1 = left
    a2, b2 = right
    return a1 * a2, a2 * b1 + b2


def hillis_steele_scan(a, b):
    L = a.shape[0]
    offset = 1
    while offset < L:  # log2(L) rounds; the loop is unrolled when jitted
        # element t combines with element t - offset; the first `offset`
        # elements combine with the identity (a = 1, b = 0)
        a_left = jnp.concatenate([jnp.ones_like(a[:offset]), a[:-offset]])
        b_left = jnp.concatenate([jnp.zeros_like(b[:offset]), b[:-offset]])
        a, b = combine((a_left, b_left), (a, b))
        offset *= 2
    return b


def blelloch_scan(a, b):
    _, hs = jax.lax.associative_scan(combine, (a, b), axis=0)
    return hs


# ------------------------------------------------ the selective SSM (S6)
def A_initializer(kind):
    """log of -Re(A) (and Im(A) for the complex 'lin' variant), shape (H, M).
    'random': -A ~ U(0.1, 1), no structure (versions 3 and 4);
    'real'  : S4D-Real, A_n = -(n + 1)          (version 5);
    'lin'   : S4D-Lin,  A_n = -1/2 + i pi n     (version 5, complex)."""
    def log_neg_re(key, shape):
        H, M = shape
        if kind == 'real':
            return jnp.log(jnp.tile(jnp.arange(1, M + 1, dtype=jnp.float32),
                                    (H, 1)))
        if kind == 'lin':
            return jnp.log(0.5) * jnp.ones(shape)
        return jnp.log(jax.random.uniform(key, shape, minval=0.1, maxval=1.0))

    def imag(key, shape):
        return math.pi * jnp.tile(jnp.arange(shape[1], dtype=jnp.float32),
                                  (shape[0], 1))
    return log_neg_re, imag


def dt_bias_init(kind, dt_min=1e-3, dt_max=0.1):
    """Bias of the Delta projection.
    'default'   : zeros, so Delta = softplus(0 + ...) ~ 0.7 (a naive choice);
    'loguniform': softplus(bias) = Delta log-uniform in [dt_min, dt_max]
                  (Mamba, version 5)."""
    def init(key, shape, dtype=jnp.float32):
        if kind == 'default':
            return jnp.zeros(shape, dtype)
        dt = jnp.exp(jax.random.uniform(key, shape, dtype) *
                     (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        return dt + jnp.log(-jnp.expm1(-dt))  # inverse of softplus
    return init


class SelectiveSSM(nn.Module):
    """S6: maps one sequence u (L, H) to y (L, H), H channels, state N each."""
    N: int = 16
    dt_rank: int = 4
    decode: bool = False
    A_init: str = 'random'
    dt_init: str = 'default'

    @nn.compact
    def __call__(self, u):
        L, H = u.shape
        cplx = self.A_init == 'lin'          # complex modes, N/2 of them
        M = self.N // 2 if cplx else self.N
        init_re, init_im = A_initializer(self.A_init)
        A = -jnp.exp(self.param('A_log', init_re, (H, M)))
        if cplx:
            A = A + 1j * self.param('A_imag', init_im, (H, M))
        D = self.param('D', nn.initializers.ones, (H,))

        # selection: Delta, B, C are computed from the input at each step
        x_dbl = nn.Dense(self.dt_rank + 2 * self.N, use_bias=False,
                         name='x_proj')(u)
        dt, B, C = jnp.split(x_dbl, [self.dt_rank, self.dt_rank + self.N],
                             axis=-1)
        if cplx:
            B = B[:, :M] + 1j * B[:, M:]
            C = C[:, :M] + 1j * C[:, M:]
        delta = jax.nn.softplus(nn.Dense(
            H, bias_init=dt_bias_init(self.dt_init), name='dt_proj')(dt))
        self.sow('intermediates', 'delta', delta)            # (L, H)

        # discretise: ZOH for A, Euler for B (as in Mamba)
        dA = jnp.exp(delta[..., None] * A)                    # (L, H, M)
        dBu = delta[..., None] * B[:, None, :] * u[..., None]  # (L, H, M)

        if not self.decode:   # training: parallel associative scan
            h = blelloch_scan(dA, dBu)
        else:                 # inference: recurrence from the cached state
            cache = self.variable('cache', 'h', jnp.zeros, (H, M), dA.dtype)

            def step(h, ab):
                h = ab[0] * h + ab[1]
                return h, h
            h_last, h = jax.lax.scan(step, cache.value, (dA, dBu))
            if self.is_mutable_collection('cache'):
                cache.value = h_last
        y = jnp.einsum('lhm,lm->lh', h, C)                    # y_t = C_t h_t
        if cplx:
            y = 2 * y.real
        return y + u * D


def make_s6_model(args, decode=False, A_init='random', dt_init='default'):
    return BatchStackedModel(
        layer_cls=SelectiveSSM,
        layer={'N': args.d_state, 'dt_rank': math.ceil(args.d_model / 16),
               'decode': decode, 'A_init': A_init, 'dt_init': dt_init},
        vocab_size=args.n_data + 2,
        d_model=args.d_model,
        n_layers=args.n_layers,
    )


# -------------------------------------------------------------- demos
def demo_scans():
    print('\n== Parallel scans for h_t = a_t h_{t-1} + b_t (D = 256) ==')
    lengths = [256, 1024, 4096, 16384]
    times = {'sequential (lax.scan)': [], 'Hillis-Steele': [],
             'Blelloch (lax.associative_scan)': []}
    fns = dict(zip(times, [sequential_scan, hillis_steele_scan,
                           blelloch_scan]))
    for L in lengths:
        ka, kb = jax.random.split(jax.random.PRNGKey(L))
        a = jax.random.uniform(ka, (L, 256), minval=0.5, maxval=1.0)
        b = jax.random.normal(kb, (L, 256))
        ref = jax.jit(sequential_scan)(a, b)
        msg = f'L = {L:6d}:'
        for name, fn in fns.items():
            f = jax.jit(fn)
            err = float(jnp.abs(f(a, b) - ref).max())
            t = 1e3 * timeit(f, a, b, repeats=5)
            times[name].append(t)
            msg += f' {name.split()[0]} {t:7.2f} ms (err {err:.0e}) |'
        print(msg)
    plt = plot_style()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for name, t in times.items():
        ax.loglog(lengths, t, 'o-', label=name)
    ax.set_xlabel('sequence length L')
    ax.set_ylabel('time (ms)')
    ax.set_title(f'Linear recurrence, 256 channels '
                 f'({jax.devices()[0].platform.upper()})')
    ax.legend(fontsize=8)
    savefig(fig, 'v3_scan_timing.png')
    plt.close(fig)


def plot_selection(model, params, args, name):
    """Delta_t (averaged over channels) along one selective-copying input:
    the model opens the gate on data tokens and keeps it shut on blanks."""
    x, _ = make_batch(np.random.default_rng(5), 1, 'selective', args.n_mem,
                      args.n_tokens, args.n_data)
    _, inter = model.apply({'params': params}, jnp.asarray(x),
                           mutable=['intermediates'])
    plt = plot_style()
    fig, axes = plt.subplots(args.n_layers, 1, figsize=(8, 1.9 * args.n_layers),
                             sharex=True, squeeze=False)
    is_data = (x[0] >= 1) & (x[0] <= args.n_data)
    for i in range(args.n_layers):
        d = inter['intermediates'][f'layers_{i}']['seq']['delta'][0][0]
        ax = axes[i, 0]
        ax.bar(np.arange(x.shape[1]), np.asarray(d.mean(-1)),
               color=np.where(is_data, 'C3', 'C0'))
        ax.set_ylabel(f'layer {i + 1}\nmean Δ_t')
    axes[0, 0].set_title('Selection at work: Δ_t on one selective-copying '
                         'input (red = data token, blue = blank / marker)')
    axes[-1, 0].set_xlabel('position t')
    fig.tight_layout()
    savefig(fig, name)
    plt.close(fig)


def main(args):
    model = make_s6_model(args)
    decode_model = make_s6_model(args, decode=True)
    if not args.skip_demo:
        demo_scans()
    if args.skip_train:
        return
    params, hist = train(model, args, 'Selective SSM (S6)')
    save_history(hist, args.save_history, model='S6', task=args.task)
    if not args.skip_demo:
        if args.task == 'selective':
            plot_selection(model, params, args, 'v3_selection_delta.png')
        demo_modes(model, decode_model, params, args, 'V3')


if __name__ == '__main__':
    main(default_parser('Selective SSM (S6) on copying tasks').parse_args())
