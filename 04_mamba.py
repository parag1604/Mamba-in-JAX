"""
Step 4: Mamba, a selective state space model.

What is NEW compared to 02_s4.py and 03_parallel_scan.py
--------------------------------------------------------
Selective SSM (Gu and Dao, 2023, https://arxiv.org/abs/2312.00752):
the step size Delta and the matrices B and C are no longer fixed parameters
but FUNCTIONS OF THE CURRENT INPUT x_t (small linear layers):

    Delta_t = softplus(W_Delta x_t),  B_t = W_B x_t,  C_t = W_C x_t
    h_t = exp(Delta_t A) h_{t-1} + Delta_t B_t x_t,     y_t = C_t h_t + D x_t

A large Delta_t lets the state be overwritten by x_t ("remember this token"),
a small one keeps the state almost unchanged ("ignore this token"). That is
the content-based selection the selective-copying task needs.
Other differences from step 2:
  * A is diagonal (A = -exp(A_log), initialised to -1, -2, ..., -N), so the
    recurrence is element-wise and exp(Delta A) is cheap;
  * the discretisation is zero-order hold for A (exp) and Euler for B;
  * no convolution kernel exists any more, so the recurrence is computed with
    the parallel associative scan of step 3;
  * the Mamba block wraps the SSM: an input projection to 2 * d_model
    channels, a short causal depthwise convolution, SiLU, the selective SSM,
    a SiLU gate from the second branch, and an output projection.

Train it on the same task as step 2 and compare:
    python 04_mamba.py                   # selective copying
    python 04_mamba.py --no-selective    # copying
"""

import argparse
import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from utils import add_task_args, train


def selective_scan(dA, dBu):
    """h_t = dA_t * h_{t-1} + dBu_t along axis 1, with the parallel scan."""
    def combine(left, right):
        a1, b1 = left
        a2, b2 = right
        return a1 * a2, a2 * b1 + b2
    _, h = jax.lax.associative_scan(combine, (dA, dBu), axis=1)
    return h


def dt_bias_init(dt_min=1e-3, dt_max=0.1):
    """Bias such that softplus(bias) = Delta is log-uniform in [dt_min, dt_max]."""
    def init(key, shape, dtype=jnp.float32):
        dt = jnp.exp(jax.random.uniform(key, shape, dtype) *
                     (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        return dt + jnp.log(-jnp.expm1(-dt))  # inverse of softplus
    return init


class SelectiveSSM(nn.Module):
    d_state: int = 16
    dt_rank: int = 4

    @nn.compact
    def __call__(self, u):
        # u: (B, L, E)
        E, N = u.shape[-1], self.d_state
        A_log = self.param(
            'A_log', lambda k, s: jnp.log(jnp.tile(jnp.arange(1, N + 1,
                                                   dtype=jnp.float32), (E, 1))),
            (E, N))
        A = -jnp.exp(A_log)  # (E, N), negative: a stable, decaying memory
        D = self.param('D', nn.initializers.ones, (E,))

        # the input-dependent parameters ("selection")
        x_dbl = nn.Dense(self.dt_rank + 2 * N, use_bias=False)(u)
        dt, B, C = jnp.split(x_dbl, [self.dt_rank, self.dt_rank + N], axis=-1)
        delta = jax.nn.softplus(
            nn.Dense(E, bias_init=dt_bias_init())(dt))       # (B, L, E)

        # discretise: dA = exp(Delta A), dB u = Delta B u
        dA = jnp.exp(delta[..., None] * A)                    # (B, L, E, N)
        dBu = delta[..., None] * B[:, :, None, :] * u[..., None]  # (B, L, E, N)
        h = selective_scan(dA, dBu)                           # (B, L, E, N)
        y = jnp.einsum('blen,bln->ble', h, C)                 # y_t = C_t h_t
        return y + u * D


class MambaBlock(nn.Module):
    d_model: int
    d_state: int = 16
    expand: int = 2
    d_conv: int = 4

    @nn.compact
    def __call__(self, x):
        # x: (B, L, d_model)
        E = self.expand * self.d_model
        x, z = jnp.split(nn.Dense(2 * E)(x), 2, axis=-1)
        # short causal depthwise convolution: pad only on the left
        x = nn.Conv(E, kernel_size=(self.d_conv,), feature_group_count=E,
                    padding=[(self.d_conv - 1, 0)])(x)
        x = nn.silu(x)
        y = SelectiveSSM(self.d_state,
                         dt_rank=math.ceil(self.d_model / 16))(x)
        y = y * nn.silu(z)  # gating branch
        return nn.Dense(self.d_model)(y)


class Mamba(nn.Module):
    vocab_size: int
    d_model: int
    n_layers: int
    d_state: int = 16

    @nn.compact
    def __call__(self, tokens):
        # tokens: (B, L) -> logits (B, L, vocab)
        x = nn.Embed(self.vocab_size, self.d_model)(tokens)
        for _ in range(self.n_layers):
            x = x + MambaBlock(self.d_model, self.d_state)(nn.LayerNorm()(x))
        return nn.Dense(self.vocab_size)(nn.LayerNorm()(x))


def main(args):
    model = Mamba(vocab_size=args.n_data + 2, d_model=args.d_model,
                  n_layers=args.n_layers, d_state=args.d_state)
    train(model, args, 'Mamba')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mamba on copying tasks')
    add_task_args(parser)
    main(parser.parse_args())
