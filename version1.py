"""
Version 1: a state space model (SSM), and its two faces: an RNN and a CNN.

What this file teaches
----------------------
A continuous-time linear SSM maps an input signal u(t) to an output y(t)
through an N-dimensional hidden state x(t):

    x'(t) = A x(t) + B u(t),        y(t) = C x(t) + D u(t)

1. Discretisation with a step size Delta, giving x_k = Ab x_{k-1} + Bb u_k,
   y_k = C x_k + D u_k:
     * zero-order hold (ZOH), exact if u is constant over each step:
           Ab = exp(Delta A),  Bb = A^-1 (exp(Delta A) - I) B
       (computed here without inverting A, from the exponential of the
       block matrix [[A, B], [0, 0]] * Delta, Van Loan's trick);
     * bilinear (Tustin), used by the original S4:
           Ab = (I - Delta/2 A)^-1 (I + Delta/2 A),
           Bb = (I - Delta/2 A)^-1 Delta B.
2. Recurrent view: run the recurrence step by step (`jax.lax.scan`).
   O(L) sequential steps, O(N) memory: ideal for generating one token at a
   time.
3. Convolutional view: unrolling the recurrence from x_{-1} = 0 gives
       y_k = sum_j C Ab^j Bb u_{k-j} + D u_k = (K * u)_k + D u_k,
       K = (C Bb, C Ab Bb, C Ab^2 Bb, ...),
   a causal convolution with a length-L kernel, computed for all positions
   at once with the FFT: ideal for training on whole sequences.
4. A trainable SSM layer (dense A, learned B, C, D, log Delta), trained in
   convolution mode on a toy task (delay the input by `delay` steps), then
   run in recurrent mode: both give the same outputs.

The script checks that the views agree, compares the two discretisations
against the exact solution, and saves plots to static/.

The functions make_HiPPO, discretize (bilinear), K_conv and
causal_convolution are Parag's original code, adapted from "The Annotated
S4" by Sasha Rush and Sidd Karamcheti (https://srush.github.io/annotated-s4/,
MIT licence).
"""

import argparse
import time

import jax
import jax.experimental
import jax.numpy as np
import numpy as onp
import optax
from jax.numpy.linalg import inv, matrix_power
from jax.scipy.linalg import expm

from utils import plot_style, savefig


# ------------------------------------------------------------ the SSM
def make_HiPPO(N):
    """HiPPO-LegS matrix (negated, so that the dynamics are stable).
    Used again in version 5, where it explains the initialisation."""
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A


def random_stable_SSM(rng, N, scale=1.0):
    """A random SSM whose A has eigenvalues in the left half-plane."""
    a_r, b_r, c_r = jax.random.split(rng, 3)
    M = jax.random.normal(a_r, (N, N)) / np.sqrt(N)
    A = scale * (M - M.T) - 0.5 * scale * np.eye(N)  # skew part + damping
    B = jax.random.normal(b_r, (N, 1))
    C = jax.random.normal(c_r, (1, N)) / np.sqrt(N)
    return A, B, C


def discretize(A, B, C, step):
    """Bilinear (Tustin) discretisation (Parag's original, from S4)."""
    I = np.eye(A.shape[0])
    BL = inv(I - (step / 2.0) * A)
    Ab = BL @ (I + (step / 2.0) * A)
    Bb = (BL * step) @ B
    return Ab, Bb, C


def discretize_zoh(A, B, C, step):
    """Zero-order hold: Ab = exp(step A), Bb = A^-1 (Ab - I) B.
    exp([[A, B], [0, 0]] step) = [[Ab, Bb], [0, I]], so no inverse needed."""
    N, M = A.shape[0], B.shape[1]
    big = np.zeros((N + M, N + M)).at[:N, :N].set(A).at[:N, N:].set(B)
    E = expm(big * step)
    return E[:N, :N], E[:N, N:], C


def scan_SSM(Ab, Bb, Cb, u, x0):
    """Recurrent view: one step per input, carrying the state x."""
    def step(x_k_1, u_k):
        x_k = Ab @ x_k_1 + Bb @ u_k
        y_k = Cb @ x_k
        return x_k, y_k

    return jax.lax.scan(step, x0, u)


def run_recurrent(Ab, Bb, Cb, D, u):
    """y = recurrence(u) + D u, from the zero state; u: (L,)."""
    x_last, y = scan_SSM(Ab, Bb, Cb, u[:, None], np.zeros(Ab.shape[0]))
    return y.reshape(-1) + D * u, x_last


def K_conv(Ab, Bb, Cb, L):
    """Convolutional view: the kernel K_l = C Ab^l Bb, l = 0..L-1
    (Parag's original: one matrix power per lag, O(L N^3))."""
    return np.array(
        [(Cb @ matrix_power(Ab, l) @ Bb).reshape() for l in range(L)]
    )


def K_scan(Ab, Bb, Cb, L):
    """The same kernel by unrolling the impulse response: O(L N^2), jittable."""
    def step(v, _):
        return Ab @ v, (Cb @ v).reshape()
    _, K = jax.lax.scan(step, Bb[:, 0], None, length=L)
    return K


def causal_convolution(u, K, nofft=False):
    if nofft:
        return np.convolve(u, K, mode="full")[: u.shape[0]]
    else:
        # zero-pad to 2L so that the FFT's circular convolution does not wrap
        # the end of the sequence around to the start (keeps it causal)
        assert K.shape[0] == u.shape[0]
        ud = np.fft.rfft(np.pad(u, (0, K.shape[0])))
        Kd = np.fft.rfft(np.pad(K, (0, u.shape[0])))
        out = ud * Kd
        return np.fft.irfft(out)[: u.shape[0]]


# ------------------------------------------ 1) the two views agree
def demo_views(args):
    print('\n== 1. Recurrent view vs convolutional view ==')
    L, N = args.L, args.N
    A, B, C = random_stable_SSM(jax.random.PRNGKey(args.seed), N)
    D = 0.5
    u = jax.random.normal(jax.random.PRNGKey(1), (L,))
    kernels = {}
    for name, disc in [('ZOH', discretize_zoh), ('bilinear', discretize)]:
        Ab, Bb, Cb = disc(A, B, C, step=args.dt)
        y_rnn, _ = run_recurrent(Ab, Bb, Cb, D, u)
        K = K_scan(Ab, Bb, Cb, L)
        K_slow = K_conv(Ab, Bb, Cb, L)
        y_fft = causal_convolution(u, K) + D * u
        y_dir = causal_convolution(u, K, nofft=True) + D * u
        kernels[name] = K
        print(f'{name:9s}: max |K_scan - K_matrix_power| = '
              f'{float(np.abs(K - K_slow).max()):.1e} | '
              f'max |RNN - CNN(FFT)| = {float(np.abs(y_rnn - y_fft).max()):.1e}'
              f' | max |RNN - CNN(direct)| = '
              f'{float(np.abs(y_rnn - y_dir).max()):.1e} '
              f'(|y| up to {float(np.abs(y_rnn).max()):.1f})')

    # a random UNstructured A is often unstable; HiPPO (version 5) is not
    Ar = jax.random.uniform(jax.random.PRNGKey(3), (N, N))
    for name, Am in [('random uniform A', Ar), ('HiPPO-LegS A', make_HiPPO(N))]:
        Ab, _, _ = discretize(Am, B, C, step=1.0 / L)
        rho = onp.abs(onp.linalg.eigvals(onp.asarray(Ab))).max()
        print(f'{name:17s}: largest |eigenvalue| of Ab = {rho:.4f} '
              f'({"unstable, the state blows up" if rho > 1 else "stable"})')

    plt = plot_style()
    Ab, Bb, Cb = discretize_zoh(A, B, C, step=args.dt)
    y_rnn, _ = run_recurrent(Ab, Bb, Cb, D, u)
    y_fft = causal_convolution(u, kernels['ZOH']) + D * u
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].plot(kernels['ZOH'], label='ZOH')
    axes[0].plot(kernels['bilinear'], '--', label='bilinear')
    axes[0].set_xlabel('lag l')
    axes[0].set_ylabel(r'$K_l = C\bar A^l \bar B$')
    axes[0].set_title(f'Kernel of a random stable SSM (N={N}, Δ={args.dt})')
    axes[0].legend()
    axes[1].plot(y_rnn, lw=2.5, alpha=0.6, label='recurrent (lax.scan)')
    axes[1].plot(y_fft, 'k:', label='convolution (FFT)')
    axes[1].set_xlabel('time step k')
    axes[1].set_ylabel('$y_k$')
    axes[1].set_title(f'Same output: max diff '
                      f'{float(np.abs(y_rnn - y_fft).max()):.0e}')
    axes[1].legend()
    fig.tight_layout()
    savefig(fig, 'v1_kernel_and_views.png')
    plt.close(fig)


# ----------------------------------- 2) how good are the discretisations
def demo_discretisation(args):
    print('\n== 2. ZOH vs bilinear against the exact continuous solution ==')
    # float64 here, so that the reference is accurate below the float32 noise
    with jax.experimental.enable_x64():
        _demo_discretisation(args)


def _demo_discretisation(args):
    N = args.N
    A, B, C = random_stable_SSM(jax.random.PRNGKey(args.seed), N)
    A, B, C = (np.asarray(M, np.float64) for M in (A, B, C))
    T = 10.0
    f = lambda t: np.sin(1.3 * t) + 0.5 * np.cos(3.1 * t)   # smooth input

    # "exact": ZOH with a tiny step, sampling the input at mid-steps
    Ls = [25, 50, 100, 200, 400, 800]            # Δ = T / L
    fine = 64 * Ls[-1]                           # every sample time is on the fine grid
    tf = (np.arange(fine) + 0.5) * T / fine
    Abf, Bbf, _ = discretize_zoh(A, B, C, T / fine)
    _, y_exact = scan_SSM(Abf, Bbf, C, f(tf)[:, None], np.zeros(N))
    y_exact = y_exact.reshape(-1)

    steps = [T / L for L in Ls]
    errs = {'ZOH': [], 'bilinear': []}
    for L, dt in zip(Ls, steps):
        t = np.arange(1, L + 1) * dt               # sample times
        idx = np.arange(1, L + 1) * (fine // L) - 1
        # ZOH holds u constant over a step (use its midpoint value);
        # bilinear (trapezoid) averages the two ends
        u_zoh = f(t - dt / 2)
        u_bil = 0.5 * (f(t) + f(t - dt))
        for name, disc, u in [('ZOH', discretize_zoh, u_zoh),
                              ('bilinear', discretize, u_bil)]:
            Ab, Bb, Cb = disc(A, B, C, dt)
            y, _ = run_recurrent(Ab, Bb, Cb, 0.0, u)
            errs[name].append(float(np.abs(y - y_exact[idx]).max()))
    for i, dt in enumerate(steps):
        print(f'Δ = {dt:7.4f}: max error ZOH {errs["ZOH"][i]:.2e} | '
              f'bilinear {errs["bilinear"][i]:.2e}')

    plt = plot_style()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.loglog(steps, errs['ZOH'], 'o-', label='zero-order hold')
    ax.loglog(steps, errs['bilinear'], 's-', label='bilinear')
    ref = onp.array(steps)
    ax.loglog(ref, errs['ZOH'][0] * (ref / ref[0]) ** 2, 'k:', lw=1,
              label=r'slope 2 ($O(\Delta^2)$)')
    ax.set_xlabel('step size Δ')
    ax.set_ylabel('max |y_discrete - y_exact|')
    ax.set_title('Discretisation error (smooth input)')
    ax.legend(fontsize=8)
    savefig(fig, 'v1_discretisation_error.png')
    plt.close(fig)


# ------------------------------- 3) a trainable SSM layer on a toy task
def init_ssm_params(rng, N):
    A, B, C = random_stable_SSM(rng, N, scale=1.0)
    return dict(A=A, B=B, C=C, D=np.zeros(()),
                log_dt=np.log(0.1) * np.ones(()))


def ssm_conv(params, u):
    """Training mode: discretise (ZOH), build the kernel, FFT-convolve."""
    Ab, Bb, Cb = discretize_zoh(params['A'], params['B'], params['C'],
                                np.exp(params['log_dt']))
    K = K_scan(Ab, Bb, Cb, u.shape[0])
    return causal_convolution(u, K) + params['D'] * u


def ssm_recurrent(params, u):
    """Inference mode: the same parameters, run as an RNN."""
    Ab, Bb, Cb = discretize_zoh(params['A'], params['B'], params['C'],
                                np.exp(params['log_dt']))
    return run_recurrent(Ab, Bb, Cb, params['D'], u)[0]


def demo_training(args):
    print(f'\n== 3. Train one SSM layer (N={args.N}) to delay its input by '
          f'{args.delay} steps ==')
    L, delay = args.L, args.delay
    params = init_ssm_params(jax.random.PRNGKey(args.seed), args.N)

    def batch(key):
        # smooth random inputs: white noise through a short moving average
        w = jax.random.normal(key, (args.batch_size, L + 4))
        u = (w[:, :-4] + w[:, 1:-3] + w[:, 2:-2] + w[:, 3:-1]) / 2.0
        y = np.pad(u, ((0, 0), (delay, 0)))[:, :L]       # y_t = u_{t-delay}
        return u, y

    def loss_fn(p, u, y):
        pred = jax.vmap(ssm_conv, in_axes=(None, 0))(p, u)
        return ((pred - y) ** 2).mean()

    tx = optax.adam(args.lr_v1)
    opt = tx.init(params)

    @jax.jit
    def step(p, opt, key):
        u, y = batch(key)
        loss, g = jax.value_and_grad(loss_fn)(p, u, y)
        upd, opt = tx.update(g, opt, p)
        return optax.apply_updates(p, upd), opt, loss

    key = jax.random.PRNGKey(123)
    losses, t0 = [], time.time()
    for i in range(args.steps_v1 + 1):
        key, sub = jax.random.split(key)
        params, opt, loss = step(params, opt, sub)
        losses.append(float(loss))
        if i % 500 == 0:
            print(f'step {i:5d} | MSE {float(loss):.4f} '
                  f'(predicting 0 gives ~{1.0:.1f}) | {time.time() - t0:.0f}s')

    u, y = batch(jax.random.PRNGKey(999))
    y_conv = ssm_conv(params, u[0])
    y_rec = ssm_recurrent(params, u[0])
    print(f'trained: test MSE {float(((y_conv - y[0]) ** 2).mean()):.4f}, '
          f'Δ = {float(np.exp(params["log_dt"])):.3f} | max |conv mode - '
          f'recurrent mode| = {float(np.abs(y_conv - y_rec).max()):.1e}')

    Ab, Bb, Cb = discretize_zoh(params['A'], params['B'], params['C'],
                                np.exp(params['log_dt']))
    K = K_scan(Ab, Bb, Cb, L)
    plt = plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
    axes[0].semilogy(losses)
    axes[0].set_xlabel('training step')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('Learning curve (convolution mode)')
    axes[1].stem(onp.arange(40), onp.asarray(K[:40]), basefmt=' ')
    axes[1].axvline(delay, color='r', ls=':', label=f'lag {delay}')
    axes[1].set_xlabel('lag l')
    axes[1].set_ylabel('$K_l$')
    axes[1].set_title('Learned kernel (ideal: a spike at the delay)')
    axes[1].legend()
    axes[2].plot(y[0], lw=2.5, alpha=0.5, label='target $u_{t-%d}$' % delay)
    axes[2].plot(y_conv, label='convolution mode')
    axes[2].plot(y_rec, 'k:', label='recurrent mode')
    axes[2].set_xlim(0, 100)
    axes[2].set_xlabel('time step t')
    axes[2].set_title('Prediction on a new input')
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, 'v1_trained_delay.png')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SSM basics')
    parser.add_argument('--L', type=int, default=256, help='sequence length')
    parser.add_argument('--N', type=int, default=16, help='state size')
    parser.add_argument('--dt', type=float, default=0.1, help='step size Δ')
    parser.add_argument('--delay', type=int, default=10)
    parser.add_argument('--steps_v1', type=int, default=4000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr_v1', type=float, default=1e-2)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    demo_views(args)
    demo_discretisation(args)
    demo_training(args)
