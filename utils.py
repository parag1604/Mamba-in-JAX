"""
Shared helpers: the synthetic tasks, the training loop, timing and plotting.

No dataset is needed: every task is generated on the fly.

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
  blanks. The model has to decide, from the CONTENT of each input, whether
  to remember it or to ignore it. A linear time-invariant (LTI) system such
  as S4 applies the same filter to every input, so it cannot do this; an
  input-dependent (selective) SSM can.

Induction heads (Olsson et al., 2022), optional:
    input : 5 2 8 T 6 1 3 ... 7 4 T
    target:                       6
  random tokens, a trigger token T somewhere followed by a value; at the end
  T appears again and the model must recall the token that followed it the
  first time.
"""

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')


# ---------------------------------------------------------------- the tasks
def make_batch(rng, batch_size, task, n_mem, n_tokens, n_data):
    """
    Token ids: 0 = blank, 1..n_data = data, n_data + 1 = marker / trigger.
    Returns x of shape (B, L) and y of shape (B, n_out); the targets belong
    to the LAST n_out positions of the sequence.
    """
    marker = n_data + 1
    if task == 'induction':
        L = n_mem + 1
        x = rng.integers(1, n_data + 1, size=(batch_size, L)).astype(np.int32)
        pos = rng.integers(0, L - 2, size=batch_size)
        y = rng.integers(1, n_data + 1, size=(batch_size, 1)).astype(np.int32)
        x[np.arange(batch_size), pos] = marker
        x[np.arange(batch_size), pos + 1] = y[:, 0]
        x[:, -1] = marker
        return x, y
    y = rng.integers(1, n_data + 1, size=(batch_size, n_tokens))
    x = np.zeros((batch_size, n_mem + n_tokens), dtype=np.int32)
    for b in range(batch_size):
        if task == 'selective':
            pos = np.sort(rng.choice(n_mem, n_tokens, replace=False))
        else:  # 'copy'
            pos = np.arange(n_tokens)
        x[b, pos] = y[b]
    x[:, n_mem:] = marker
    return x, y.astype(np.int32)


def seq_len(args):
    return args.n_mem + (1 if args.task == 'induction' else args.n_tokens)


# ------------------------------------------------------------ the training
def train(model, args, name, verbose=True):
    """
    Train `model` (tokens (B, L) -> logits (B, L, vocab)) with AdamW on fresh
    batches. Returns the parameters and the history (a list of dicts with the
    step, the training loss and the accuracy on 512 fixed test sequences).
    """
    rng = np.random.default_rng(args.seed)
    test_rng = np.random.default_rng(10_000 + args.seed)
    mb = lambda r, n: make_batch(r, n, args.task, args.n_mem, args.n_tokens,
                                 args.n_data)
    x_test, y_test = mb(test_rng, 512)
    params = model.init(jax.random.PRNGKey(args.seed),
                        jnp.asarray(x_test[:2]))['params']
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    if verbose:
        print(f'{name}: {n_params / 1e3:.1f}k parameters | task {args.task} '
              f'| sequence length {x_test.shape[1]}')

    schedule = optax.warmup_cosine_decay_schedule(
        0.0, args.lr, min(100, args.steps // 10), args.steps)
    tx = optax.chain(optax.clip_by_global_norm(1.0),
                     optax.adamw(schedule, weight_decay=args.weight_decay))
    opt_state = tx.init(params)

    def loss_fn(params, x, y):
        logits = model.apply({'params': params}, x)[:, -y.shape[1]:]
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
    history = []
    t0 = time.time()
    for i in range(args.steps + 1):
        x, y = mb(rng, args.batch_size)
        if i > 0:  # step 0 = the untrained model
            params, opt_state, loss, acc = step(params, opt_state, x, y)
        if i % args.eval_every == 0 or i == args.steps:
            loss, acc = eval_fn(params, x, y)
            _, test_acc = eval_fn(params, x_test, y_test)
            rec = dict(step=i, loss=float(loss), train_acc=float(acc),
                       test_acc=float(test_acc), time=time.time() - t0)
            history.append(rec)
            if verbose and (i % args.print_every == 0 or i == args.steps):
                print(f'step {i:5d} | loss {rec["loss"]:.4f} | train acc '
                      f'{rec["train_acc"]:.3f} | test acc '
                      f'{rec["test_acc"]:.3f} | {rec["time"]:.0f}s')
    return params, history


def add_task_args(parser, steps=2000):
    parser.add_argument('--task', default='selective',
                        choices=['copy', 'selective', 'induction'])
    parser.add_argument('--n_mem', type=int, default=64,
                        help='length of the part that holds the data tokens')
    parser.add_argument('--n_tokens', type=int, default=8,
                        help='number of tokens to copy')
    parser.add_argument('--n_data', type=int, default=8,
                        help='size of the data vocabulary')
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--steps', type=int, default=steps)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--eval_every', type=int, default=50)
    parser.add_argument('--print_every', type=int, default=250)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_history', default=None,
                        help='write the learning curve to this JSON file')
    parser.add_argument('--skip_demo', action='store_true',
                        help='only train (skip the timing / equality demos)')
    parser.add_argument('--skip_train', action='store_true',
                        help='only run the demos')
    return parser


def save_history(history, path, **meta):
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(dict(meta, history=history), f)
        print(f'Saved {path}')


# ------------------------------------------------- generation and timing
def timeit(fn, *a, repeats=10):
    """Mean wall time of a (jitted) call, after one warm-up/compile call."""
    jax.block_until_ready(fn(*a))
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = fn(*a)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / repeats


def empty_cache(decode_model, x):
    """The zero recurrent state for a batch like x. (Calling init also runs
    the model once, which writes into the cache, so we zero it.)"""
    cache = decode_model.init(jax.random.PRNGKey(0), x[:, :1])['cache']
    return jax.tree_util.tree_map(jnp.zeros_like, cache)


def recurrent_logits(decode_model, params, x):
    """
    Run a model in RECURRENT (decode) mode, one token at a time, with a jitted
    single-step function. x: (B, L) tokens -> (B, L, vocab) logits.
    The model must keep its state in the 'cache' collection.
    """
    cache = empty_cache(decode_model, x)

    @jax.jit
    def step(cache, tok):
        logits, mut = decode_model.apply({'params': params, 'cache': cache},
                                         tok, mutable=['cache'])
        return mut['cache'], logits

    outs = []
    for t in range(x.shape[1]):
        cache, logits = step(cache, x[:, t:t + 1])
        outs.append(logits)
    return jnp.concatenate(outs, axis=1), cache


def state_bytes(cache):
    return sum(v.size * v.dtype.itemsize
               for v in jax.tree_util.tree_leaves(cache))


def generation_benchmark(model, decode_model, params, lengths, vocab,
                         repeats=20):
    """
    Cost of producing ONE new token after a context of length t:
      * recurrent: one decode step from the cached state (flat in t);
      * parallel: re-run the training-mode forward on the whole prefix
        (grows with t), which is what you would do without a recurrence.
    Returns a dict of times in milliseconds, and the state size in bytes.
    """
    x1 = jnp.ones((1, 1), jnp.int32)
    cache = empty_cache(decode_model, x1)

    @jax.jit
    def step(cache, tok):
        logits, mut = decode_model.apply({'params': params, 'cache': cache},
                                         tok, mutable=['cache'])
        return mut['cache'], logits

    fwd = jax.jit(lambda x: model.apply({'params': params}, x)[:, -1])
    t_rec, t_par = [], []
    rng = np.random.default_rng(0)
    for L in lengths:
        x = jnp.asarray(rng.integers(1, vocab, size=(1, L)), jnp.int32)
        t_par.append(1e3 * timeit(fwd, x, repeats=repeats))
        # the recurrent step does not depend on L (the state has a fixed size)
        t_rec.append(1e3 * timeit(step, cache, x1, repeats=repeats * 5))
    return dict(lengths=list(lengths), recurrent_ms=t_rec, parallel_ms=t_par,
                state_bytes=state_bytes(cache))


# --------------------------------------------------------------- plotting
def plot_style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 110,
                         'axes.grid': True, 'grid.alpha': 0.3,
                         'axes.spines.top': False,
                         'axes.spines.right': False,
                         'font.size': 10})
    return plt


def savefig(fig, name):
    os.makedirs(STATIC, exist_ok=True)
    path = os.path.join(STATIC, name)
    fig.savefig(path, bbox_inches='tight')
    print(f'Saved static/{name}')


def plot_generation(res, title, name):
    plt = plot_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.loglog(res['lengths'], res['parallel_ms'], 'o-',
              label='parallel mode: re-run the whole prefix')
    ax.loglog(res['lengths'], res['recurrent_ms'], 's-',
              label=f'recurrent mode: one step from the state '
                    f'({res["state_bytes"] / 1024:.0f} KiB)')
    ax.set_xlabel('context length t (tokens)')
    ax.set_ylabel('time to produce one token (ms)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    savefig(fig, name)
    plt.close(fig)


def recurrent_vs_parallel(model, decode_model, params, x):
    """Max |difference| between training-mode and decode-mode logits."""
    par = model.apply({'params': params}, x)
    rec, _ = recurrent_logits(decode_model, params, x)
    return float(jnp.abs(par - rec).max()), float(jnp.abs(par).max())


def default_parser(description, steps=2000):
    return add_task_args(argparse.ArgumentParser(description=description),
                         steps=steps)
