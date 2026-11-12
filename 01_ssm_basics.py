"""
Step 1: a state space model (SSM), and its two faces: an RNN and a CNN.

What this file teaches
----------------------
A continuous-time linear SSM maps an input signal u(t) to an output y(t)
through an N-dimensional hidden state x(t):

    x'(t) = A x(t) + B u(t)          y(t) = C x(t) (+ D u(t))

1. Discretisation. With a step size Delta, the bilinear (Tustin) method gives
       x_k = Ab x_{k-1} + Bb u_k,     y_k = C x_k
   with Ab = (I - Delta/2 A)^-1 (I + Delta/2 A), Bb = (I - Delta/2 A)^-1 Delta B.
2. Recurrent view: run that recurrence step by step (`jax.lax.scan`).
   O(L) sequential steps, O(1) memory per step: great for generation.
3. Convolutional view: unrolling the recurrence (x_{-1} = 0) gives
       y_k = sum_j  C Ab^j Bb  u_{k-j}  =  (K * u)_k,   K = (C Bb, C Ab Bb, ...)
   a causal convolution with a length-L kernel K, computed for all positions
   at once with the FFT: great for training on whole sequences.
4. HiPPO initialisation of A (Gu et al., 2020): a specific matrix whose state
   keeps an optimal polynomial summary of the input's history; a random A
   usually forgets quickly or blows up.
The script checks that both views give the same output, and plots the kernel K.

Code adapted from "The Annotated S4" by Sasha Rush and Sidd Karamcheti
(https://srush.github.io/annotated-s4/, MIT licence).
"""

import argparse
import os

import jax
import jax.numpy as np
import numpy as onp
from jax.numpy.linalg import inv, matrix_power

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_HiPPO(N):
    """HiPPO-LegS matrix (negated, so that the dynamics are stable)."""
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A


def random_SSM(rng, N):
    a_r, b_r, c_r = jax.random.split(rng, 3)
    A = jax.random.uniform(a_r, (N, N))
    B = jax.random.uniform(b_r, (N, 1))
    C = jax.random.uniform(c_r, (1, N))
    return A, B, C


def discretize(A, B, C, step):
    I = np.eye(A.shape[0])
    BL = inv(I - (step / 2.0) * A)
    Ab = BL @ (I + (step / 2.0) * A)
    Bb = (BL * step) @ B
    return Ab, Bb, C


def scan_SSM(Ab, Bb, Cb, u, x0):
    """Recurrent view: one step per input, carrying the state x."""
    def step(x_k_1, u_k):
        x_k = Ab @ x_k_1 + Bb @ u_k
        y_k = Cb @ x_k
        return x_k, y_k

    return jax.lax.scan(step, x0, u)


def run_SSM(A, B, C, u):
    L = u.shape[0]
    N = A.shape[0]
    Ab, Bb, Cb = discretize(A, B, C, step=1.0 / L)
    # run the recurrence from the zero state
    return scan_SSM(Ab, Bb, Cb, u[:, np.newaxis], np.zeros((N,)))[1]


def K_conv(Ab, Bb, Cb, L):
    """Convolutional view: the kernel K_l = C Ab^l Bb, l = 0..L-1."""
    return np.array(
        [(Cb @ matrix_power(Ab, l) @ Bb).reshape() for l in range(L)]
    )


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


def main(args):
    rng = jax.random.PRNGKey(args.seed)
    L, N = args.L, args.N
    u = jax.random.uniform(jax.random.PRNGKey(1), (L,))

    # 1) random SSM: RNN view == CNN view (with and without the FFT)
    A, B, C = random_SSM(rng, N)
    y_rnn = run_SSM(A, B, C, u).reshape(-1)
    ssm = discretize(A, B, C, step=1.0 / L)
    K = K_conv(*ssm, L)
    y_cnn = causal_convolution(u, K)
    y_direct = causal_convolution(u, K, nofft=True)
    print(f'random A : max |RNN - CNN(FFT)| = '
          f'{float(np.abs(y_rnn - y_cnn).max()):.2e}, '
          f'max |RNN - CNN(direct)| = '
          f'{float(np.abs(y_rnn - y_direct).max()):.2e}')

    # 2) the same with a HiPPO matrix
    A_h = make_HiPPO(N)
    ssm_h = discretize(A_h, B, C, step=1.0 / L)
    K_h = K_conv(*ssm_h, L)
    y_rnn_h = run_SSM(A_h, B, C, u).reshape(-1)
    print(f'HiPPO A  : max |RNN - CNN(FFT)| = '
          f'{float(np.abs(y_rnn_h - causal_convolution(u, K_h)).max()):.2e}')
    # (eigenvalues of a non-symmetric matrix: plain numpy, JAX does them on CPU only)
    eig_r = onp.abs(onp.linalg.eigvals(onp.asarray(ssm[0]))).max()
    eig_h = onp.abs(onp.linalg.eigvals(onp.asarray(ssm_h[0]))).max()
    print(f'largest |eigenvalue| of Ab: random A {float(eig_r):.4f} '
          f'(> 1 means the state blows up), HiPPO {float(eig_h):.4f}')

    os.makedirs('outputs', exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(K, label='random A')
    axes[0].plot(K_h, label='HiPPO A')
    axes[0].set_xlabel('lag l')
    axes[0].set_ylabel('K[l] = C Ab^l Bb')
    axes[0].set_title('SSM convolution kernels')
    axes[0].legend()
    axes[1].plot(u, label='input u', alpha=0.5)
    axes[1].plot(y_rnn_h, label='output y (HiPPO SSM)')
    axes[1].set_xlabel('time step k')
    axes[1].set_title('Input and output')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('outputs/ssm_kernels.png', dpi=100)
    print('Saved outputs/ssm_kernels.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SSM basics')
    parser.add_argument('--L', type=int, default=64, help='sequence length')
    parser.add_argument('--N', type=int, default=8, help='state size')
    parser.add_argument('--seed', type=int, default=0)
    main(parser.parse_args())
