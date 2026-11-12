"""
Version 5: Mamba with a proper HiPPO-derived initialisation.

What is NEW compared to version4.py
-----------------------------------
Only the INITIAL VALUES of A and Delta. The architecture is unchanged.

* HiPPO (Gu et al., 2020) asks: which A makes the state x(t) an optimal
  online summary of the input's whole history? For the scaled Legendre
  measure (LegS) the answer is the matrix of version 1 (`make_HiPPO`):
        A_nk = -sqrt(2n+1) sqrt(2k+1)  (n > k),   -(n+1)  (n = k),   0 (n < k).
  It is not stably diagonalisable, but it is NORMAL PLUS LOW RANK:
        A = S - P P^T,   P_n = sqrt(n + 1/2),
  where S + I/2 is skew-symmetric, so S = V diag(lambda) V* with every
  lambda_n = -1/2 + i omega_n. S4 keeps the low-rank term (DPLR);
  S4D (Gu, Gupta et al., 2022) simply drops it and keeps diagonal values:
    S4D-LegS : the exact lambda_n of S;
    S4D-Inv  : -1/2 + i N/pi (N/(2n+1) - 1), a closed-form fit of S4D-LegS;
    S4D-Real : -(n + 1), the diagonal of A itself (real; Mamba's default);
    S4D-Lin  : -1/2 + i pi n, which comes from the Fourier variant of HiPPO
               (FouT) rather than LegS: a damped Fourier basis.
  All of them give each mode a DIFFERENT decay rate or frequency, i.e. a
  spread of memory lengths and a well-conditioned set of basis functions.
* Delta: Mamba initialises the bias of the Delta projection so that
  Delta = softplus(bias) is LOG-UNIFORM in [1e-3, 1e-1]. Together with
  |A_n| = 1..N this spreads the memory lengths 1 / (Delta |A_n|) of the
  modes over several orders of magnitude (about 1 to 1000 steps).
* The naive initialisation of versions 3-4 (-A ~ U(0.1, 1), Delta ~
  softplus(0) = 0.69) gives almost every mode a memory of only a few steps,
  so at initialisation gradients from the marker positions barely reach
  the data tokens. (In practice the selective Delta learns to shrink on
  blanks quickly; on the short default task the naive init trains just as
  well, see the README.)

The script plots the HiPPO spectra and the memory lengths at
initialisation, and trains Mamba with a chosen initialisation:
    python version5.py --A_init real --dt_init loguniform   # S4D-Real (default)
    python version5.py --A_init lin  --dt_init loguniform   # S4D-Lin (complex)
    python version5.py --A_init random --dt_init default    # = version 4
"""

import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np

from utils import (default_parser, make_batch, plot_style, save_history,
                   savefig, train)
from version1 import make_HiPPO
from version2 import demo_modes
from version4 import make_mamba


def hippo_spectra(N):
    """The diagonal approximations of HiPPO-LegS (state size N)."""
    with jax.experimental.enable_x64():           # exact enough to check
        A = np.asarray(make_HiPPO(N), dtype=np.float64)
    P = np.sqrt(np.arange(N) + 0.5)
    S = A + np.outer(P, P)                 # normal part: -I/2 + skew
    # S + I/2 is skew-symmetric, so -i (S + I/2) is Hermitian: eigh is stable
    omega = np.linalg.eigvalsh(-1j * (S + 0.5 * np.eye(N)))
    omega = np.sort(omega[omega >= 0])     # one of each conjugate pair
    n = np.arange(N // 2)
    return dict(A=A, S=S, legs=omega, inv=np.sort(N / np.pi * (N / (2 * n + 1) - 1)),
                lin=np.pi * n, real=-(np.arange(N) + 1.0))


def demo_hippo(N=16):
    print(f'\n== HiPPO-LegS (N = {N}) and its diagonal approximations ==')
    sp = hippo_spectra(N)
    S, I = sp['S'], np.eye(N)
    print(f'S = A + P P^T: diagonal entries {np.unique(np.round(np.diag(S), 9))}, '
          f'S + I/2 skew-symmetric: {np.allclose(S + I / 2, -(S + I / 2).T)}')
    print(f'diagonal of A (S4D-Real): {np.diag(sp["A"])[:6]} ...')
    print('Im(lambda_n): S4D-LegS (exact) | S4D-Inv (fit) | S4D-Lin (pi n)')
    for k in range(N // 2):
        print(f'  {sp["legs"][k]:8.2f} | {sp["inv"][k]:8.2f} | '
              f'{sp["lin"][k]:6.2f}')

    plt = plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    n = np.arange(N // 2)
    ax = axes[0]
    ax.semilogy(n, sp['legs'], 'o', ms=7, mfc='none',
                label='S4D-LegS: eigenvalues of S (exact)')
    ax.semilogy(n, sp['inv'], 'x', ms=7, label='S4D-Inv (closed-form fit)')
    ax.semilogy(n[1:], sp['lin'][1:], 's', ms=4,
                label=r'S4D-Lin: $\pi n$ (Fourier, HiPPO-FouT)')
    ax.set_xlabel('mode n (sorted)')
    ax.set_ylabel(r'frequency Im($a_n$)')
    ax.set_title('Complex diagonal inits (all with Re = -1/2)')
    ax.legend(fontsize=8)
    ax = axes[1]
    im = ax.imshow(sp['A'], cmap='RdBu', vmin=-np.abs(sp['A']).max(),
                   vmax=np.abs(sp['A']).max())
    ax.set_title(f'HiPPO-LegS A (N = {N}); diagonal = S4D-Real -(n+1)')
    ax.set_xlabel('k')
    ax.set_ylabel('n')
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    savefig(fig, 'v5_hippo_spectra.png')
    plt.close(fig)


def memory_lengths(args, A_init, dt_init):
    """1 / (Delta_t |Re a_n|) over channels, modes and positions, at init."""
    model = make_mamba(args, A_init=A_init, dt_init=dt_init)
    x, _ = make_batch(np.random.default_rng(0), 8, args.task, args.n_mem,
                      args.n_tokens, args.n_data)
    variables = model.init(jax.random.PRNGKey(args.seed), jnp.asarray(x))
    _, inter = model.apply({'params': variables['params']}, jnp.asarray(x),
                           mutable=['intermediates'])
    out = []
    for i in range(args.n_layers):
        delta = inter['intermediates'][f'layers_{i}']['seq']['delta'][0]
        neg_re_A = jnp.exp(variables['params'][f'layers_{i}']['seq']['A_log'])
        out.append((1.0 / (delta[..., None] * neg_re_A)).reshape(-1))
    return np.asarray(jnp.concatenate(out))


def demo_memory(args):
    print('\n== Memory length 1 / (Delta |Re a_n|) of the modes at init ==')
    plt = plot_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bins = np.logspace(-1, 4, 51)
    for (A_init, dt_init), label in [
            (('random', 'default'), 'naive (versions 3-4)'),
            (('real', 'loguniform'), 'S4D-Real + log-uniform Δ')]:
        m = memory_lengths(args, A_init, dt_init)
        print(f'{label:28s}: median {np.median(m):7.1f} steps, '
              f'{100 * np.mean(m > args.n_mem):5.1f}% of modes remember '
              f'longer than {args.n_mem} steps')
        ax.hist(m, bins=bins, alpha=0.6, weights=np.full(m.size, 1.0 / m.size),
                label=label)
    ax.axvline(args.n_mem, color='k', ls=':', lw=1,
               label=f'distance to recall ({args.n_mem})')
    ax.set_xscale('log')
    ax.set_xlabel('memory length of a mode (steps)')
    ax.set_ylabel('fraction of (channel, mode, t)')
    ax.set_title('Memory lengths at initialisation')
    ax.legend(fontsize=8)
    savefig(fig, 'v5_memory_lengths.png')
    plt.close(fig)


def main(args):
    if not args.skip_demo:
        demo_hippo(args.d_state)
        demo_memory(args)
    if args.skip_train:
        return
    model = make_mamba(args, A_init=args.A_init, dt_init=args.dt_init)
    params, hist = train(model, args,
                         f'Mamba (A {args.A_init}, Delta {args.dt_init})')
    save_history(hist, args.save_history, model='Mamba', task=args.task,
                 A_init=args.A_init, dt_init=args.dt_init, seed=args.seed)
    if not args.skip_demo:
        decode_model = make_mamba(args, decode=True, A_init=args.A_init,
                                  dt_init=args.dt_init)
        demo_modes(model, decode_model, params, args, 'V5')


if __name__ == '__main__':
    parser = default_parser('Mamba with HiPPO initialisation')
    parser.add_argument('--A_init', default='real',
                        choices=['random', 'real', 'lin'])
    parser.add_argument('--dt_init', default='loguniform',
                        choices=['default', 'loguniform'])
    main(parser.parse_args())
