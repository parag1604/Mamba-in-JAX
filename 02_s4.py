"""
Step 2: a deep SSM network (S4-style) trained on two memory tasks.

What is NEW compared to 01_ssm_basics.py
----------------------------------------
* The SSM becomes a trainable layer (`SSMLayer`): A (HiPPO-initialised), B, C,
  D and the step size Delta (learned in log-space) are parameters. During
  training the layer uses the convolutional view (FFT), so a whole sequence
  is processed in parallel.
* One SSM maps a 1-D signal to a 1-D signal; `nn.vmap` makes H independent
  copies, one per feature channel (`cloneLayer`).
* `SequenceBlock`: SSM (mixes information along time) + GELU + a gated
  linear unit (mixes the channels), with LayerNorm and a residual connection.
  `StackedModel`: embedding -> blocks -> per-position classifier;
  `nn.vmap` again adds the batch dimension.
* Two synthetic tasks (utils.py): copying (data at fixed positions) and
  selective copying (data at random positions). Run both:
      python 02_s4.py --no-selective      # copying
      python 02_s4.py                     # selective copying
  The SSM is linear and TIME-INVARIANT: the same kernel K is applied
  whatever the input. A fixed time-shift solves copying, but selective
  copying requires deciding from the content which tokens to keep, which
  is exactly what an LTI system cannot do. Step 4 fixes this.

Code adapted from "The Annotated S4" by Sasha Rush and Sidd Karamcheti
(https://srush.github.io/annotated-s4/, MIT licence).
"""

import argparse
import importlib

import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn.initializers import lecun_normal

from utils import add_task_args, train

ssm_basics = importlib.import_module('01_ssm_basics')
make_HiPPO = ssm_basics.make_HiPPO
discretize = ssm_basics.discretize
K_conv = ssm_basics.K_conv
causal_convolution = ssm_basics.causal_convolution


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    def init(key, shape):
        return jax.random.uniform(key, shape) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)

    return init


class SSMLayer(nn.Module):
    N: int
    l_max: int

    def setup(self):
        # SSM parameters
        self.A = self.param("A", lambda rng, shape: make_HiPPO(self.N),
                            (self.N, self.N))
        self.B = self.param("B", lecun_normal(), (self.N, 1))
        self.C = self.param("C", lecun_normal(), (1, self.N))
        self.D = self.param("D", nn.initializers.ones, (1,))

        # Step parameter (log Delta, so that Delta stays positive)
        self.log_step = self.param("log_step", log_step_initializer(), (1,))

    def __call__(self, u):
        # u: (L,) one channel of one sequence
        step = np.exp(self.log_step)
        ssm = discretize(self.A, self.B, self.C, step=step)
        K = K_conv(*ssm, self.l_max)
        # CNN mode; D u is a skip connection
        return causal_convolution(u, K) + self.D * u


def cloneLayer(layer):
    # H copies of the layer, one per feature channel (axis 1 of (L, H))
    return nn.vmap(
        layer,
        in_axes=1,
        out_axes=1,
        variable_axes={"params": 1},
        split_rngs={"params": True},
    )


SSMLayer = cloneLayer(SSMLayer)


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


# add the batch dimension: (B, L) -> (B, L, vocab), parameters shared
BatchStackedModel = nn.vmap(
    StackedModel,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "dropout": None},
    split_rngs={"params": False, "dropout": True},
)


def main(args):
    L = args.n_mem + args.n_tokens
    model = BatchStackedModel(
        layer_cls=SSMLayer,
        layer={'N': args.d_state, 'l_max': L},
        vocab_size=args.n_data + 2,
        d_model=args.d_model,
        n_layers=args.n_layers,
    )
    train(model, args, 'S4-style SSM')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='S4-style SSM on copying tasks')
    add_task_args(parser)
    main(parser.parse_args())
