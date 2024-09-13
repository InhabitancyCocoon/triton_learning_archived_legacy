"""
Low-Memory Dropout
==================

In this tutorial, you will write a memory-efficient implementation of dropout whose state
will be composed of a single int32 seed. This differs from more traditional implementations of dropout,
whose state is generally composed of a bit mask tensor of the same shape as the input.

In doing so, you will learn about:

* The limitations of naive implementations of Dropout with PyTorch.

* Parallel pseudo-random number generation in Triton.

"""

# %%
# Baseline
# --------
#
# The *dropout* operator was first introduced in [SRIVASTAVA2014]_ as a way to improve the performance
# of deep neural networks in low-data regime (i.e. regularization).
#
# It takes a vector as input and produces a vector of the same shape as output. Each scalar in the
# output has a probability :math:`p` of being changed to zero and otherwise it is copied from the input.
# This forces the network to perform well even when only :math:`1 - p` scalars from the input are available.
#
# At evaluation time we want to use the full power of the network so we set :math:`p=0`. Naively this would
# increase the norm of the output (which can be a bad thing, e.g. it can lead to artificial decrease
# in the output softmax temperature). To prevent this we multiply the output by :math:`\frac{1}{1 - p}`, which
# keeps the norm consistent regardless of the dropout probability.
#
# Let's first take a look at the baseline implementation.

from codecs import ascii_encode
from random import random
from tarfile import BLOCKSIZE
import tabulate
import torch

import triton
import triton.language as tl


# %%
# Seeded dropout
# --------------
#
# The above implementation of dropout works fine, but it can be a bit awkward to deal with. Firstly
# we need to store the dropout mask for backpropagation. Secondly, dropout state management can get
# very tricky when using recompute/checkpointing (e.g. see all the notes about `preserve_rng_state` in
# https://pytorch.org/docs/stable/checkpoint.html). In this tutorial we'll describe an alternative implementation
# that (1) has a smaller memory footprint; (2) requires less data movement; and (3) simplifies the management
# of persisting randomness across multiple invocations of the kernel.
#
# Pseudo-random number generation in Triton is simple! In this tutorial we will use the
# :code:`triton.language.rand` function which generates a block of uniformly distributed :code:`float32`
# values in [0, 1), given a seed and a block of :code:`int32` offsets. But if you need it, Triton also provides
# other :ref:`random number generation strategies<Random Number Generation>`.
#
# .. note::
#    Triton's implementation of PRNG is based on the Philox algorithm (described on [SALMON2011]_).
#
# Let's put it all together.


@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    n_cols,
    p,
    seeds,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = tl.program_id(0) * BLOCK_SIZE
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_seed = seeds + pid
    mask = col_offsets < n_cols
    x = tl.load(x_ptr + row_start + col_offsets, mask=mask)
    random = tl.rand(row_seed, row_start + col_offsets)
    x_keep = random > p
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + row_start + col_offsets, output, mask=mask)


def seeded_dropout(x, p, seeds):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (x.shape[0], 1)
    BLOCKSIZE = triton.next_power_of_2(x.shape[1])
    _seeded_dropout[grid](x, output, x.shape[1], p, seeds, BLOCK_SIZE=BLOCKSIZE)
    return output


x = torch.randn(size=(3, 5)).cuda()
# Compare this to the baseline - dropout mask is never instantiated!
seeds_1 = torch.rand(size=(x.shape[0], )).cuda()
seeds_2 = torch.rand(size=(x.shape[0], )).cuda()
output = seeded_dropout(x, p=0.5, seeds=seeds_1)
output2 = seeded_dropout(x, p=0.5, seeds=seeds_1)
output3 = seeded_dropout(x, p=0.5, seeds=seeds_2)


assert torch.allclose(output, output2)

print(x)

print(output)

print(output2)

print(output3)

# %%
# Et Voilà! We have a triton kernel that applies the same dropout mask provided the seed is the same!
# If you'd like explore further applications of pseudorandomness in GPU programming, we encourage you
# to explore the `python/triton/language/random.py`!

# %%
# Exercises
# ---------
#
# 1. Extend the kernel to operate over a matrix and use a vector of seeds - one per row.
# 2. Add support for striding.
# 3. (challenge) Implement a kernel for sparse Johnson-Lindenstrauss transform which generates the projection matrix on the fly each time using a seed.

# %%
# References
# ----------
#
# .. [SALMON2011] John K. Salmon, Mark A. Moraes, Ron O. Dror, and David E. Shaw, "Parallel Random Numbers: As Easy as 1, 2, 3", 2011
# .. [SRIVASTAVA2014] Nitish Srivastava and Geoffrey Hinton and Alex Krizhevsky and Ilya Sutskever and Ruslan Salakhutdinov, "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", JMLR 2014
