"""
Step 3: computing a linear recurrence in parallel (the associative scan).

What is NEW compared to 02_s4.py
--------------------------------
S4 can train with a convolution only because its recurrence is time
INVARIANT: the same Ab, Bb at every step, hence a single kernel K. Mamba makes
the recurrence depend on the input,

    h_t = a_t * h_{t-1} + b_t        (a_t, b_t different at every step),

so there is no kernel any more. Running the recurrence step by step
(`jax.lax.scan`) needs L sequential steps, which is slow on a GPU.

The trick: steps of a linear recurrence compose. Applying (a1, b1) and then
(a2, b2) to a state h gives a2 (a1 h + b1) + b2 = (a1 a2) h + (a2 b1 + b2), so

    (a1, b1) o (a2, b2) = (a1 * a2,  a2 * b1 + b2)

is an ASSOCIATIVE operator, and all prefix "sums" h_0, h_1, ..., h_{L-1} of
an associative operator can be computed in O(log L) parallel rounds:

* Hillis-Steele scan (written below): in round r every element combines with
  the element 2^r positions to its left. log2(L) rounds, O(L log L) work.
* Blelloch scan (`jax.lax.associative_scan`): an up-sweep and a down-sweep
  over a tree; O(L) work, 2 log2(L) rounds.

The script checks that the three methods agree and times them. On a CPU the
sequential scan is often fastest (there is little parallel hardware to use);
the parallel scans pay off on a GPU/TPU and for long sequences.

The Hillis-Steele implementation follows the idea of Franz A. Heinsen's
torch_parallel_scan (https://github.com/glassroom/torch_parallel_scan, MIT).
"""

import argparse
import time

import jax
import jax.numpy as jnp


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


def benchmark(fn, a, b, repeats=10):
    fn = jax.jit(fn)
    fn(a, b).block_until_ready()  # compile once
    t0 = time.time()
    for _ in range(repeats):
        out = fn(a, b)
    out.block_until_ready()
    return out, (time.time() - t0) / repeats


def main(args):
    key_a, key_b = jax.random.split(jax.random.PRNGKey(args.seed))
    # a_t in (0, 1): a decaying memory, like exp(Delta * A) with A < 0
    a = jax.random.uniform(key_a, (args.L, args.D), minval=0.5, maxval=1.0)
    b = jax.random.normal(key_b, (args.L, args.D))
    print(f'Recurrence of length L = {args.L} over D = {args.D} channels '
          f'on {jax.devices()[0].platform}')
    ref, t_seq = benchmark(sequential_scan, a, b)
    for name, fn, t in [('sequential (lax.scan)', None, t_seq),
                        ('Hillis-Steele', hillis_steele_scan, None),
                        ('Blelloch (lax.associative_scan)', blelloch_scan, None)]:
        if fn is not None:
            out, t = benchmark(fn, a, b)
            err = float(jnp.abs(out - ref).max())
        else:
            err = 0.0
        print(f'{name:32s} {1e3 * t:8.2f} ms | max |error| vs sequential '
              f'{err:.1e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parallel scans')
    parser.add_argument('--L', type=int, default=4096)
    parser.add_argument('--D', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    main(parser.parse_args())
