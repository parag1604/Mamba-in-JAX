"""
The synthetic tasks and the training loop shared by 02_s4.py and 04_mamba.py.

No dataset is needed: both tasks are generated on the fly.

Copying (the classic memory test for sequence models):
    input : 3 7 1 4 0 0 0 0 ... 0 9 9 9 9
    target:                         3 7 1 4
  n_tokens data tokens at the START, blanks (0), then n_tokens marker tokens
  (9 here); at the marker positions the model must output the data tokens in
  order. The data always sit at the same distance from the markers, so a
  model that applies a FIXED time-shift (a convolution) can solve it.

Selective copying (Gu and Dao, 2023, the Mamba paper):
    input : 0 3 0 0 7 0 1 0 0 0 4 0 ... 9 9 9 9
    target:                         3 7 1 4
  same, but the data tokens are scattered at RANDOM positions among the
  blanks. The model has to decide, based on the CONTENT of each input,
  whether to remember it or to ignore it. A linear time-invariant (LTI)
  system such as S4 applies the same filter to every input, so it cannot
  do this well; Mamba's input-dependent (selective) SSM can.
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax


def make_batch(rng, batch_size, n_mem, n_tokens, n_data, selective):
    """
    Token ids: 0 = blank, 1..n_data = data, n_data + 1 = marker.
    Returns x of shape (B, n_mem + n_tokens) and y of shape (B, n_tokens).
    """
    marker = n_data + 1
    y = rng.integers(1, n_data + 1, size=(batch_size, n_tokens))
    x = np.zeros((batch_size, n_mem + n_tokens), dtype=np.int32)
    for b in range(batch_size):
        if selective:
            pos = np.sort(rng.choice(n_mem, n_tokens, replace=False))
        else:
            pos = np.arange(n_tokens)
        x[b, pos] = y[b]
    x[:, n_mem:] = marker
    return x, y.astype(np.int32)


def train(model, args, name):
    """Train with Adam on freshly generated batches; prints accuracy."""
    rng = np.random.default_rng(args.seed)
    vocab = args.n_data + 2
    x0, _ = make_batch(rng, 2, args.n_mem, args.n_tokens, args.n_data,
                       args.selective)
    params = model.init(jax.random.PRNGKey(args.seed), jnp.asarray(x0))
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f'{name}: {n_params / 1e3:.1f}k parameters | task: '
          f'{"selective copying" if args.selective else "copying"}')

    schedule = optax.warmup_cosine_decay_schedule(
        0.0, args.lr, min(100, args.steps // 10), args.steps)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule))
    opt_state = tx.init(params)

    def loss_fn(params, x, y):
        logits = model.apply(params, x)[:, -args.n_tokens:]  # marker positions
        logp = jax.nn.log_softmax(logits)
        nll = -jnp.take_along_axis(logp, y[..., None], axis=-1).mean()
        acc = (logits.argmax(-1) == y).mean()
        return nll, acc

    @jax.jit
    def step(params, opt_state, x, y):
        (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, x, y)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, acc

    eval_fn = jax.jit(loss_fn)
    t0 = time.time()
    for i in range(args.steps + 1):
        x, y = make_batch(rng, args.batch_size, args.n_mem, args.n_tokens,
                          args.n_data, args.selective)
        params, opt_state, loss, acc = step(params, opt_state, x, y)
        if i % args.print_every == 0 or i == args.steps:
            xv, yv = make_batch(rng, 512, args.n_mem, args.n_tokens,
                                args.n_data, args.selective)
            _, val_acc = eval_fn(params, xv, yv)
            print(f'step {i:5d} | loss {float(loss):.4f} | train acc '
                  f'{float(acc):.3f} | test acc {float(val_acc):.3f} '
                  f'| {time.time() - t0:.0f}s')
    return params, float(val_acc)


def add_task_args(parser):
    parser.add_argument('--selective', action=argparse.BooleanOptionalAction, default=True,
                        help='selective copying (default) or plain copying')
    parser.add_argument('--n_mem', type=int, default=32,
                        help='length of the part that holds the data tokens')
    parser.add_argument('--n_tokens', type=int, default=4,
                        help='number of tokens to copy')
    parser.add_argument('--n_data', type=int, default=8,
                        help='size of the data vocabulary')
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--print_every', type=int, default=250)
    parser.add_argument('--seed', type=int, default=0)
    return parser

