"""
Version 4: Mamba, the selective SSM inside a gated block.

What is NEW compared to version3.py
-----------------------------------
The selective SSM (S6) of version 3 is unchanged. What changes is the block
around it (Gu and Dao, 2023, Fig. 3), which replaces Parag's
SequenceBlock (SSM -> GELU -> gated linear unit):

    x (L, d_model)
    |-- in_proj --> [ u | z ]            (two branches of E = expand*d_model)
    u --> short CAUSAL depthwise conv1d (kernel d_conv = 4) --> SiLU
      --> selective SSM (Delta, B, C from u) --> y
    y * SiLU(z)                          (multiplicative gate)
      --> out_proj --> (L, d_model)

and the model stacks these blocks with pre-norm residual connections,
x <- x + MambaBlock(RMSNorm(x)), like a Transformer without attention and
without MLPs. The short convolution lets each position see its immediate
neighbours before the SSM decides what to store.

Training is still parallel (associative scan + an ordinary convolution);
inference is still recurrent. The recurrent state of a block is now TWO
things, both of fixed size: the SSM state h (E x N) and the last d_conv - 1
inputs of the convolution. The script checks that token-by-token decoding
reproduces the parallel outputs and times both modes.

Run:
    python version4.py --task selective
    python version4.py --task copy
    python version4.py --task induction --n_mem 64
"""

import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from utils import default_parser, save_history, train
from version2 import demo_modes
from version3 import SelectiveSSM, plot_selection


class MambaBlock(nn.Module):
    """One Mamba block on one sequence: (L, d_model) -> (L, d_model)."""
    d_model: int
    d_state: int = 16
    expand: int = 2
    d_conv: int = 4
    decode: bool = False
    A_init: str = 'random'
    dt_init: str = 'default'

    @nn.compact
    def __call__(self, x):
        L = x.shape[0]
        E, k = self.expand * self.d_model, self.d_conv
        u, z = jnp.split(nn.Dense(2 * E, name='in_proj')(x), 2, axis=-1)

        # short causal depthwise convolution: y_t = sum_j w_j u_{t-k+1+j} + b
        bound = 1.0 / math.sqrt(k)
        w = self.param('conv_w', nn.initializers.uniform(2 * bound), (k, E))
        w = w - bound                                  # U(-bound, bound)
        b = self.param('conv_b', nn.initializers.zeros, (E,))
        if self.decode:   # keep the last k - 1 inputs as part of the state
            buf = self.variable('cache', 'conv', jnp.zeros, (k - 1, E))
            up = jnp.concatenate([buf.value, u], axis=0)
            if self.is_mutable_collection('cache'):
                buf.value = up[-(k - 1):]
        else:             # training: pad with zeros on the left (causal)
            up = jnp.pad(u, ((k - 1, 0), (0, 0)))
        u = sum(w[j] * up[j:j + L] for j in range(k)) + b
        u = nn.silu(u)

        y = SelectiveSSM(N=self.d_state,
                         dt_rank=math.ceil(self.d_model / 16),
                         decode=self.decode, A_init=self.A_init,
                         dt_init=self.dt_init, name='seq')(u)
        y = y * nn.silu(z)                               # gating branch
        return nn.Dense(self.d_model, name='out_proj')(y)


class Mamba(nn.Module):
    vocab_size: int
    d_model: int
    n_layers: int
    d_state: int = 16
    decode: bool = False
    A_init: str = 'random'
    dt_init: str = 'default'

    @nn.compact
    def __call__(self, tokens):
        # tokens: (L,) -> logits (L, vocab)
        x = nn.Embed(self.vocab_size, self.d_model)(tokens)
        for i in range(self.n_layers):
            x = x + MambaBlock(self.d_model, self.d_state,
                               decode=self.decode, A_init=self.A_init,
                               dt_init=self.dt_init,
                               name=f'layers_{i}')(nn.RMSNorm()(x))
        return nn.Dense(self.vocab_size)(nn.RMSNorm()(x))


BatchMamba = nn.vmap(
    Mamba,
    in_axes=0,
    out_axes=0,
    variable_axes={'params': None, 'cache': 0, 'intermediates': 0},
    split_rngs={'params': False},
)


def make_mamba(args, decode=False, A_init='random', dt_init='default'):
    return BatchMamba(vocab_size=args.n_data + 2, d_model=args.d_model,
                      n_layers=args.n_layers, d_state=args.d_state,
                      decode=decode, A_init=A_init, dt_init=dt_init)


def main(args):
    model = make_mamba(args)
    decode_model = make_mamba(args, decode=True)
    params, hist = train(model, args, 'Mamba')
    save_history(hist, args.save_history, model='Mamba', task=args.task)
    if not args.skip_demo:
        if args.task == 'selective':
            plot_selection(model, params, args, 'v4_selection_delta.png')
        demo_modes(model, decode_model, params, args, 'V4')


if __name__ == '__main__':
    main(default_parser('Mamba on copying tasks').parse_args())
