#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

# This is an implementation from scratch of a "GPT", that is a model
# composed of several causal self-attention blocks. It is equipped
# with a caching mechanism for keys and values to avoid a O(N^3) cost
# for auto-regression.

# This implementation is equipped with RNN layers to replace the MHA

import math, warnings

import torch, einops

from torch import nn
from torch.nn import functional as F

import ffutils

# import memload

######################################################################

# A BracketedSequence is a BxTx... tensor with a first and a nb time
# steps to compute.

# Modules able to process it expect that they will have to process a
# first bracket starting at t=0, followed by a succession of brackets
# that move forward in time, do not overlap, and cover the axis T with
# no holes.
#
# Although it is more general, for a classical prompt-conditioned
# auto-regressive process it will be a first bracket starting at 0 and
# of arbitrary length for the "prompt", followed by brackets of length
# 1 for the successive tokens.
#
# Modules able to process brackets may implement a cache that is
# resetted when init_cache is True


class BracketedSequence:
    def __init__(self, x, first=None, nb=None, init_cache=None):
        self.x = x
        assert (first is None and nb is None and init_cache is None) or (
            first is not None and nb is not None and init_cache is not None
        )

        self.first = 0 if first is None else first
        self.nb = x.size(1) if nb is None else nb
        self.init_cache = True if init_cache is None else init_cache

    def slice(self):
        return self.x[:, self.first : self.first + self.nb]

    def complete(self):
        return self.first == 0 and self.nb == self.x.size(1)


######################################################################


class CacheWrapper(nn.Module):
    def __init__(self, *f):
        super().__init__()
        self.f = f[0] if len(f) == 1 else nn.Sequential(*f)

    def forward(self, bs):
        if bs.init_cache:
            y = self.f(bs.slice())
            self.cache_y = y.new(*((y.size(0), bs.x.size(1)) + y.size()[2:]))
            self.cache_y[:, bs.first : bs.first + bs.nb] = y
        else:
            assert tuple(bs.x.size()[:2]) == tuple(self.cache_y.size()[:2])
            assert bs.first + bs.nb <= self.cache_y.size(1)
            self.cache_y[:, bs.first : bs.first + bs.nb] = self.f(bs.slice())

        return BracketedSequence(self.cache_y, bs.first, bs.nb, bs.init_cache)


##############################


class WithResidual(nn.Module):
    def __init__(self, *f):
        super().__init__()
        self.f = f[0] if len(f) == 1 else nn.Sequential(*f)

    def forward(self, bs):
        return BracketedSequence(bs.x + self.f(bs).x, bs.first, bs.nb, bs.init_cache)


##############################


class AddPositionalEncoding(nn.Module):
    def __init__(self, len_max):
        super().__init__()
        self.len_max = len_max

    # [Vaswani et al 2018] PE_{t,2i} = sin(t/(L^{2i/D})), PE_{t,2i+1} = cos(t/(L^{2i/D}))

    def forward(self, bs):
        if bs.init_cache:
            t = torch.arange(bs.x.size(1), dtype=bs.x.dtype, device=bs.x.device)[
                :, None
            ]
            j = torch.arange(bs.x.size(2), dtype=bs.x.dtype, device=bs.x.device)[
                None, :
            ]
            k = j % 2
            self.pe = torch.sin(
                t / (self.len_max ** ((j - k) / bs.x.size(2))) + math.pi / 2 * k
            )
            self.cache_y = bs.x.new(bs.x.size())

        self.cache_y[:, bs.first : bs.first + bs.nb] = (
            bs.slice() + self.pe[bs.first : bs.first + bs.nb]
        )

        return BracketedSequence(self.cache_y, bs.first, bs.nb, bs.init_cache)


import pscan


# X is /.../xTxD   A is /.../xT   Y_init is /.../xD


def pscan_dim(A, X, Y_init, dim=-2):
    s = X.size()
    a, T, b = s[:dim].numel(), s[dim], s[dim + 1 :].numel()

    A = A.reshape(a, T, *s[dim + 1 : -1])
    X = X.reshape(a, T, *s[dim + 1 : -1], -1)

    if Y_init is None:
        Y_init = X.new_zeros(a, *s[dim + 1 : -1], X.size(-1))
    else:
        Y_init = Y_init.reshape(a, *s[dim + 1 : -1], -1)

    Y = pscan.pscan(A, X, Y_init).reshape(s)

    return Y


def pscan_shape(A, X, Y_init):
    s = X.size()
    A = A.reshape(-1, s[-2])
    X = X.reshape(-1, s[-2], s[-1])

    if Y_init is None:
        Y_init = X.new_zeros(X.size(0), s[-1])
    else:
        Y_init = Y_init.reshape(-1, s[-1])

    Y = pscan.pscan(A, X, Y_init).reshape(s)

    return Y


def nsum_shape(X, Y_init):
    s = X.size()
    X = X.reshape(-1, s[-2], s[-1])  # ntd

    Y = 0 if Y_init is None else Y_init.reshape(-1, s[-1])
    result = []

    for k in range(X.size(1)):
        Y = Y + X[:, k]
        Y = Y / Y.norm(dim=-1, keepdim=True).clamp(min=1)
        result.append(Y)

    return torch.cat(result, dim=1).reshape(s)


##############################


class DumbRec(nn.Module):
    def __init__(
        self,
        dim_model,
        dim_qk,
        dim_v,
        nb_heads,
        nb_lines,
        attention_dropout=0.0,
        len_max=1e5,
        logger=print,
        **kwargs,
    ):
        super().__init__()

        def randw(*d):
            return nn.Parameter(torch.randn(*d) / math.sqrt(d[-1]))

        self.nb_lines = nb_lines
        self.attention_dropout = attention_dropout

        self.k_star = randw(nb_lines, dim_qk)

        self.w_qw = randw(nb_heads, dim_qk, dim_model)
        self.w_qr = randw(nb_heads, dim_qk, dim_model)
        # self.w_k = randw(nb_heads, dim_qk, dim_model)
        self.w_v = randw(nb_heads, dim_v, dim_model)
        self.w_o = randw(dim_v * nb_heads, dim_model)

    def reset_inner_loss(self):
        self.acc_attention = 0
        self.acc_nb = 0

    def get_inner_loss(self):
        warnings.warn("l2 regularization", RuntimeWarning)
        return (self.acc_attention / self.acc_nb).pow(2).sum()
        # return torch.tensor([0], device=self.w_qw.device)

    def forward(self, bs):
        x_q, t0, t1 = bs.x, bs.first, bs.first + bs.nb

        if bs.init_cache:
            self.rec_v = x_q.new_zeros(
                x_q.size(0), self.nb_lines, x_q.size(1), self.w_v.size(1)
            )
            # self.rec_k = x_q.new_zeros(
            # x_q.size(0), self.nb_lines, x_q.size(1), self.w_k.size(1)
            # )
            self.cache_y = x_q.new_zeros(x_q.size(0), x_q.size(1), self.w_o.size(1))

        ######################################################################
        # Prepare the keys

        k_star = self.k_star[:, None, :].expand(-1, t1 - t0, -1)

        warnings.warn("rotating key barrel", RuntimeWarning)
        k_star = self.k_star[:, None, :].expand(-1, x_q.size(1), -1)
        t_barrel = torch.arange(t0, t1, device=k_star.device)
        t_barrel = t_barrel[None, :].expand(k_star.size(0), t1 - t0)
        l_barrel = (
            torch.arange(k_star.size(0), device=k_star.device)[:, None] + t_barrel
        ) % k_star.size(0)
        k_star = k_star[l_barrel, t_barrel]

        ######################################################################
        # Compute the recurrent state

        qw = torch.einsum("ntc,hdc->nhtd", x_q[:, t0:t1], self.w_qw)

        v = torch.einsum("ntc,hdc->nhtd", x_q[:, t0:t1], self.w_v)
        # k = torch.einsum("ntc,hdc->nhtd", x_q[:, t0:t1], self.w_k)

        aw = torch.einsum(
            "nhtd,ltd->nhlt",
            qw,
            k_star,
        ) / math.sqrt(self.w_qw.size(1))

        aw = aw.softmax(dim=2)  # nhlt

        if self.train:
            self.acc_attention += aw.sum(dim=(0, 1, 3))
            self.acc_nb += aw.size(0) * aw.size(1) * aw.size(3)

        aw = F.dropout(aw, self.attention_dropout, self.training)

        A = 1 - aw.sum(dim=1)  # nlt

        V = torch.einsum("nhlt,nhtd->nltd", aw, v).contiguous()
        # K = torch.einsum("nhlt,nhtd->nltd", aw, k).contiguous()

        if t0 == 0:
            V0 = None
            # K0 = None
        else:
            V0 = self.rec_v[:, :, t0 - 1]
            # K0 = self.rec_k[:, :, t0 - 1]

        self.rec_v[:, :, t0:t1] = pscan_shape(A, V, V0)
        # self.rec_k[:, :, t0:t1] = pscan_shape(A, K, K0)

        ######################################################################
        # compute the readout

        qr = torch.einsum("ntc,hdc->nhtd", x_q[:, t0:t1], self.w_qr)

        ar = torch.einsum(
            "nhtd,ld->nhlt",
            qr,
            # self.rec_k[:, :, t0:t1],
            self.k_star,
        ) / math.sqrt(self.w_qr.size(1))

        ar = ar.softmax(dim=2)  # nhlt

        ar = F.dropout(ar, self.attention_dropout, self.training)

        y = torch.einsum(
            "nhlt,nltd->nthd",
            ar,
            self.rec_v[:, :, t0:t1],
        ).flatten(2)

        self.cache_y[:, t0:t1] = y @ self.w_o

        return BracketedSequence(self.cache_y, t0, t1 - t0, bs.init_cache)


##############################


class KVRec(nn.Module):
    def __init__(
        self,
        dim_model,
        dim_qk,
        dim_v,
        nb_heads,
        nb_lines,
        attention_dropout=0.0,
        len_max=1e5,
        logger=print,
        **kwargs,
    ):
        super().__init__()

        def randw(*d):
            return nn.Parameter(torch.randn(*d) / math.sqrt(d[-1]))

        self.nb_lines = nb_lines
        self.attention_dropout = attention_dropout

        self.k_star = randw(nb_lines, dim_qk)

        self.w_qw = randw(nb_heads, dim_qk, dim_model)
        self.w_qr = randw(nb_heads, dim_qk, dim_model)
        self.w_k = randw(nb_heads, dim_qk, dim_model)
        self.w_v = randw(nb_heads, dim_v, dim_model)
        self.w_o = randw(dim_v * nb_heads, dim_model)

    def reset_inner_loss(self):
        self.acc_attention = 0
        self.acc_nb = 0

    def get_inner_loss(self):
        warnings.warn("l2 regularization", RuntimeWarning)
        return (self.acc_attention / self.acc_nb).pow(2).sum()
        # return torch.tensor([0], device=self.w_qw.device)
        # warnings.warn("side regularization", RuntimeWarning)
        # return (
        # (0.5 / self.nb_lines - self.acc_attention / self.acc_nb).clamp(min=0).sum()
        # )
        # return torch.tensor([0], device=self.w_qw.device)

    def forward(self, bs):
        x_q, t0, t1 = bs.x, bs.first, bs.first + bs.nb

        if bs.init_cache:
            self.rec_v = x_q.new_zeros(
                x_q.size(0), self.nb_lines, x_q.size(1), self.w_v.size(1)
            )
            self.rec_k = x_q.new_zeros(
                x_q.size(0), self.nb_lines, x_q.size(1), self.w_k.size(1)
            )
            self.cache_y = x_q.new_zeros(x_q.size(0), x_q.size(1), self.w_o.size(1))

        ######################################################################
        # Prepare the keys

        k_star = self.k_star[:, None, :].expand(-1, t1 - t0, -1)

        warnings.warn("rotating key barrel", RuntimeWarning)
        k_star = self.k_star[:, None, :].expand(-1, x_q.size(1), -1)
        t_barrel = torch.arange(t0, t1, device=k_star.device)
        t_barrel = t_barrel[None, :].expand(k_star.size(0), t1 - t0)
        l_barrel = (
            torch.arange(k_star.size(0), device=k_star.device)[:, None] + t_barrel
        ) % k_star.size(0)
        k_star = k_star[l_barrel, t_barrel]

        ######################################################################
        # Compute the recurrent state

        qw = torch.einsum("ntc,hdc->nhtd", x_q[:, t0:t1], self.w_qw)

        v = torch.einsum("ntc,hdc->nhtd", x_q[:, t0:t1], self.w_v)
        k = torch.einsum("ntc,hdc->nhtd", x_q[:, t0:t1], self.w_k)

        aw = torch.einsum(
            "nhtd,ltd->nhlt",
            qw,
            k_star,
        ) / math.sqrt(self.w_qw.size(1))

        aw = aw.softmax(dim=2)  # nhlt

        if self.train:
            # We want all the memory lines to be used similarly
            self.acc_attention += aw.sum(dim=(0, 1, 3))  # Sum accross NxHx_xT
            self.acc_nb += aw.size(0) * aw.size(1) * aw.size(3)

        aw = F.dropout(aw, self.attention_dropout, self.training)

        A = 1 - aw.sum(dim=1)  # nlt

        V = torch.einsum("nhlt,nhtd->nltd", aw, v).contiguous()
        K = torch.einsum("nhlt,nhtd->nltd", aw, k).contiguous()

        if t0 == 0:
            V0 = None
            K0 = None
        else:
            V0 = self.rec_v[:, :, t0 - 1]
            K0 = self.rec_k[:, :, t0 - 1]

        self.rec_v[:, :, t0:t1] = pscan_shape(A, V, V0)
        self.rec_k[:, :, t0:t1] = pscan_shape(A, K, K0)

        ######################################################################
        # compute the readout

        qr = torch.einsum("ntc,hdc->nhtd", x_q[:, t0:t1], self.w_qr)

        ar = torch.einsum(
            "nhtd,nltd->nhlt",
            qr,
            self.rec_k[:, :, t0:t1],
        ) / math.sqrt(self.w_qr.size(1))

        ar = ar.softmax(dim=2)  # nhlt

        ar = F.dropout(ar, self.attention_dropout, self.training)

        y = torch.einsum(
            "nhlt,nltd->nthd",
            ar,
            self.rec_v[:, :, t0:t1],
        ).flatten(2)

        self.cache_y[:, t0:t1] = y @ self.w_o

        return BracketedSequence(self.cache_y, t0, t1 - t0, bs.init_cache)


##############################


# Returns a tensor with an additional index at rank win_dim, that move
# along the same dimension as dim, on a domain {0...win_size-1}, and
# dim is restricted on a domain reduced by win_size-1 values.


def moving_window(x, dim, win_dim, win_size):
    size, stride = x.size(), x.stride()
    size = size[:dim] + (size[dim] - win_size + 1,) + size[dim + 1 :]
    size = size[:win_dim] + (win_size,) + size[win_dim:]
    stride = stride[:win_dim] + (stride[dim],) + stride[win_dim:]

    return x.as_strided(size=size, stride=stride)


##############################


class Caterpillar(nn.Module):
    def __init__(
        self,
        dim_model,
        dim_qk,
        dim_v,
        nb_heads,
        caterpillar_length,
        caterpillar_height,
        attention_dropout=0.0,
        len_max=1e5,
        logger=print,
        **kwargs,
    ):
        super().__init__()

        warnings.warn("Caterpillar", RuntimeWarning)

        def randw(*d, amplitude=None):
            if amplitude is None:
                amplitude = 1 / math.sqrt(d[-1])
            return nn.Parameter(amplitude * torch.randn(*d))

        self.caterpillar_length = caterpillar_length
        self.caterpillar_height = caterpillar_height
        self.attention_dropout = attention_dropout

        ######################################################################
        # sup_args

        x = kwargs.get("gate_dropout")
        if x is None:
            self.proba_gate_dropout = 0.0
        else:
            self.proba_gate_dropout = float(x)

        logger(f"self.proba_gate_dropout {self.proba_gate_dropout}")

        x = kwargs.get("default_bg")
        if x is None:
            default_bg = -math.log(caterpillar_height - 1)
        else:
            default_bg = float(x)

        logger(f"default_bg {default_bg}")

        ######################################################################

        self.w_G = randw(nb_heads, caterpillar_height, dim_model)
        self.b_G = nn.Parameter(torch.full((nb_heads, caterpillar_height), default_bg))

        self.w_K = randw(nb_heads, dim_qk, dim_model)
        self.w_V = randw(nb_heads, dim_v, dim_model)
        self.w_Q = randw(nb_heads, dim_qk, dim_model)
        self.w_O = randw(dim_v * nb_heads, dim_model)

        self.init_K_rec = randw(
            caterpillar_height,
            caterpillar_length,
            dim_qk,
        )
        self.init_V_rec = randw(
            caterpillar_height,
            caterpillar_length,
            dim_v,
        )

    def reset_inner_loss(self):
        self.acc_attention = 0
        self.acc_nb = 0

    def get_inner_loss(self):
        # warnings.warn("l2 regularization", RuntimeWarning)
        # return (self.acc_attention / self.acc_nb).pow(2).sum()
        return torch.tensor([0], device=self.w_Q.device)

    def forward(self, bs):
        # Dimensions to make the source a bit clearer, that's needed

        X, t0, t1 = bs.slice(), bs.first, bs.first + bs.nb

        N = bs.x.size(0)
        T = bs.x.size(1)
        H = self.w_V.size(0)
        DV = self.w_V.size(1)
        DK = self.w_K.size(1)
        DM = self.w_O.size(1)
        R = self.caterpillar_height
        L = self.caterpillar_length

        assert (
            t0 >= L and (t1 - t0) % L == 0
        ), f"bs.first should be greater than caterpillar_length, and bs.nb should be a multiple of caterpillar_length"

        # We cache values to deal efficiently with auto-regression

        if bs.init_cache:
            self.rec_V = X.new_zeros(N, R, T, DV)
            self.rec_K = X.new_zeros(N, R, T, DK)
            # We start the recurrent sequences with optimizable
            # initial values. No idea if it helps.
            self.rec_V[:, :, t0 - L : t0] = self.init_V_rec[None, :, :, :]
            self.rec_K[:, :, t0 - L : t0] = self.init_K_rec[None, :, :, :]

            self.cache_Y = X.new_zeros(N, T, DM)

        V = torch.einsum("ntc,hdc->nhtd", X, self.w_V)
        K = torch.einsum("ntc,hdc->nhtd", X, self.w_K)

        ######################################################################
        # Compute the recurrent state

        # This is the Gating sequence that modulates the storing of
        # the new key and value in the R pairs of the current
        # stack. There are R independent gating values, which means
        # that the current K/V may be stored in multiple pairs of the
        # recurrent state, or not at all.

        G = (
            torch.einsum("ntc,hrc->nhrt", X, self.w_G) + self.b_G[None, :, :, None]
        ).sigmoid()

        # warnings.warn("softmax gating", RuntimeWarning)

        # G = (
        # torch.einsum("ntc,hrc->nhrt", X, self.w_G) + self.b_G[None, :, :, None]
        # ).softmax(dim=2)

        ######################################################################
        # The "flashbacks"

        if self.training and self.proba_gate_dropout > 0.0:
            # This is a better implementation of "flashbacks".

            # G is NxHxExT where e is the caterpillar's row.

            warnings.warn("gate dropout", RuntimeWarning)

            kill = (
                torch.rand(G.size(), device=G.device) <= self.proba_gate_dropout
            ).float()

            alpha = G / (1 - self.proba_gate_dropout)

            G = alpha * (1 - kill)

        ######################################################################
        # Clip the gating to avoid values greater than 1 when several
        # heads hit the same row

        G = G / G.sum(1, keepdim=True).clamp(min=1)

        ######################################################################
        # Roll the gating indexes

        # warnings.warn("rotating barrel", RuntimeWarning)

        # r_barrel = torch.arange(R, device=G.device)[None, None, :, None]
        # t_barrel = torch.arange(t1 - t0, device=G.device)[None, None, None, :]
        # r_barrel = (r_barrel + (t_barrel + t0) // L) % R
        # G = G.gather(dim=2, index=r_barrel.expand_as(G))

        # We prepare the arguments for the parallel scan

        A = 1 - G.sum(1)

        # warnings.warn("harmonic recurrence", RuntimeWarning)
        # har = torch.arange(t0, t1, device = G.device).float() + 1
        # A = har / (har + 1)
        # G = G / har

        gated_V = torch.einsum("nhrt,nhtd->nrtd", G, V)
        gated_K = torch.einsum("nhrt,nhtd->nrtd", G, K)

        # We start from cached values, which matters in inference

        init_rec_V = self.rec_V[:, :, t0 - L : t0]
        init_rec_K = self.rec_K[:, :, t0 - L : t0]

        #################################################################
        # Associative scan

        # Here there is a trick: Since the stack at position t is
        # computed by updating that at position t-L, the parallel
        # scan operates with a period of L. To do so we split the
        # sequence indexing in two axes, the second of size L, and
        # run the parallel scan using the first as the sequence index.

        A = A.unflatten(2, (-1, L))
        gated_V = gated_V.unflatten(2, (-1, L))
        gated_K = gated_K.unflatten(2, (-1, L))

        next_V = pscan_dim(A, gated_V, init_rec_V, dim=2)
        next_K = pscan_dim(A, gated_K, init_rec_K, dim=2)

        self.rec_V[:, :, t0:t1] = next_V.flatten(2, 3)
        self.rec_K[:, :, t0:t1] = next_K.flatten(2, 3)

        ######################################################################
        # compute the readout

        Q = torch.einsum("ntc,hdc->nhtd", X, self.w_Q)

        # We build tensors NxHxTxFxL where N is the sample index, H
        # the head, T the time, F the row in the caterpillar, and L
        # the column in the caterpillar

        windowed_V = moving_window(
            self.rec_V[:, :, t0 - L + 1 : t1], dim=2, win_dim=3, win_size=L
        )

        windowed_K = moving_window(
            self.rec_K[:, :, t0 - L + 1 : t1], dim=2, win_dim=3, win_size=L
        )

        # We have an attention score for each of the RxL values

        ar = torch.einsum(
            "nhtd,nftld->nhtfl",
            Q,
            windowed_K,
        ) / math.sqrt(DK)

        # softmax can operate only on one dimension, hence the
        # flattening

        ar = ar.flatten(3).softmax(dim=3).view(ar.size())

        ar = F.dropout(ar, self.attention_dropout, self.training)

        # Compute the output for each head, flatten to concatenate

        Y = torch.einsum(
            "nhtfl,nftld->nthd",
            ar,
            windowed_V,
        ).flatten(2)

        # Compute the final output

        self.cache_Y[:, t0:t1] = Y @ self.w_O

        return BracketedSequence(self.cache_Y, t0, t1 - t0, bs.init_cache)


##############################


class QKVAttention(nn.Module):
    def __init__(
        self,
        dim_model,
        dim_qk,
        dim_v,
        nb_heads=1,
        causal=False,
        attention_dropout=0.0,
        logger=print,
        **kwargs,
    ):
        super().__init__()

        def randw(*d):
            return nn.Parameter(torch.randn(*d) / math.sqrt(d[-1]))

        self.causal = causal
        self.attention_dropout = attention_dropout
        self.record_attention = False

        self.w_q = randw(nb_heads, dim_qk, dim_model)
        self.w_k = randw(nb_heads, dim_qk, dim_model)
        self.w_v = randw(nb_heads, dim_v, dim_model)
        self.w_o = randw(dim_v * nb_heads, dim_model)

    def forward(self, bs):
        x_q = bs.x

        assert (
            self.causal or bs.complete()
        ), "Partial evaluation is only possible for causal models"

        if bs.init_cache:
            self.cache_k = x_q.new_zeros(
                x_q.size(0), self.w_k.size(0), x_q.size(1), self.w_k.size(1)
            )
            self.cache_v = x_q.new_zeros(
                x_q.size(0), self.w_v.size(0), x_q.size(1), self.w_v.size(1)
            )
            self.cache_y = x_q.new_zeros(x_q.size(0), x_q.size(1), self.w_o.size(1))

        q = torch.einsum("ntc,hdc->nhtd", x_q[:, bs.first : bs.first + bs.nb], self.w_q)

        self.cache_k[:, :, bs.first : bs.first + bs.nb] = torch.einsum(
            "ntc,hdc->nhtd", x_q[:, bs.first : bs.first + bs.nb], self.w_k
        )
        self.cache_v[:, :, bs.first : bs.first + bs.nb] = torch.einsum(
            "ntc,hdc->nhtd", x_q[:, bs.first : bs.first + bs.nb], self.w_v
        )

        a = torch.einsum(
            "nhtd,nhsd->nhts", q, self.cache_k[:, :, : bs.first + bs.nb]
        ) / math.sqrt(self.w_q.size(1))

        if self.causal:
            if bs.init_cache:
                self.cache_attzero = (
                    torch.arange(x_q.size(1), device=q.device)[None, None, :, None]
                    < torch.arange(x_q.size(1), device=q.device)[None, None, None, :]
                )
            a = a.masked_fill(
                self.cache_attzero[
                    :, :, bs.first : bs.first + bs.nb, : bs.first + bs.nb
                ],
                float("-inf"),
            )

        a = a.softmax(dim=3)

        if self.record_attention:
            self.a = a

        a = F.dropout(a, self.attention_dropout, self.training)

        y = torch.einsum(
            "nhts,nhsd->nthd", a, self.cache_v[:, :, : bs.first + bs.nb]
        ).flatten(2)

        self.cache_y[:, bs.first : bs.first + bs.nb] = y @ self.w_o

        return BracketedSequence(self.cache_y, bs.first, bs.nb, bs.init_cache)


##############################


class MyGPT(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        dim_model,
        dim_keys,
        dim_hidden,
        nb_heads,
        nb_blocks,
        nb_lines=None,
        caterpillar_height=None,
        causal=False,
        dropout=0.0,
        len_max=1e5,
        attention_layer="kvrec",
        logger=print,
        **kwargs,
    ):
        super().__init__()

        assert attention_layer in {
            "mha",
            "dumbrec",
            "kvrec",
            "caterpillar",
        }, f"Unknown attention operator {attention_layer}."

        if attention_layer == "caterpillar":
            assert nb_lines % caterpillar_height == 0
            self.caterpillar_length = nb_lines // caterpillar_height
            self.caterpillar_height = caterpillar_height
        else:
            self.caterpillar_length = -1
            self.caterpillar_height = -1

        assert dim_model % nb_heads == 0

        self.embedding = nn.Sequential(
            CacheWrapper(nn.Embedding(vocabulary_size, dim_model), nn.Dropout(dropout)),
            AddPositionalEncoding(len_max),
        )

        trunk_blocks = []

        def attlayer():
            if attention_layer == "mha":
                return QKVAttention(
                    dim_model=dim_model,
                    dim_qk=dim_keys,
                    dim_v=dim_model // nb_heads,
                    nb_heads=nb_heads,
                    causal=causal,
                    attention_dropout=dropout,
                    logger=logger,
                    **kwargs,
                )
            elif attention_layer == "dumbrec":
                return DumbRec(
                    dim_model=dim_model,
                    dim_qk=dim_keys,
                    dim_v=dim_model // nb_heads,
                    nb_heads=nb_heads,
                    nb_lines=nb_lines,
                    attention_dropout=dropout,
                    logger=logger,
                    **kwargs,
                )
            elif attention_layer == "kvrec":
                return KVRec(
                    dim_model=dim_model,
                    dim_qk=dim_keys,
                    dim_v=dim_model // nb_heads,
                    nb_heads=nb_heads,
                    nb_lines=nb_lines,
                    attention_dropout=dropout,
                    logger=logger,
                    **kwargs,
                )
            elif attention_layer == "caterpillar":
                return Caterpillar(
                    dim_model=dim_model,
                    dim_qk=dim_keys,
                    dim_v=dim_model // nb_heads,
                    nb_heads=nb_heads,
                    caterpillar_length=self.caterpillar_length,
                    caterpillar_height=self.caterpillar_height,
                    attention_dropout=dropout,
                    logger=logger,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown attention type {attention_layer}.")

        for b in range(nb_blocks):
            trunk_blocks += [
                WithResidual(
                    CacheWrapper(nn.LayerNorm((dim_model,))),
                    attlayer(),
                ),
                WithResidual(
                    CacheWrapper(
                        nn.LayerNorm((dim_model,)),
                        nn.Linear(in_features=dim_model, out_features=dim_hidden),
                        nn.ReLU(),
                        nn.Linear(in_features=dim_hidden, out_features=dim_model),
                        nn.Dropout(dropout),
                    ),
                ),
            ]

        self.trunk = nn.Sequential(*trunk_blocks)

        self.readout = CacheWrapper(
            nn.Linear(in_features=dim_model, out_features=vocabulary_size)
        )

        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Embedding):
                    m.weight.normal_(mean=0, std=2e-2)
                elif isinstance(m, nn.LayerNorm):
                    m.bias.zero_()
                    m.weight.fill_(1.0)

        self.reset_inner_loss()

    def forward(self, bs):
        bs = BracketedSequence(F.pad(bs.x, (1, -1)), bs.first, bs.nb, bs.init_cache)

        # To make the code simpler in the Caterpillar layer, we pad
        # here. It's unclear if/how much it hurts computationaly by
        # increasing the sequence length for the other layers

        if self.caterpillar_length > 0:
            original_nb = bs.nb
            if bs.nb % self.caterpillar_length > 0:
                bs.nb += self.caterpillar_length - bs.nb % self.caterpillar_length

            bs = BracketedSequence(
                F.pad(bs.x, (self.caterpillar_length, self.caterpillar_length)),
                bs.first + self.caterpillar_length,
                bs.nb,
                bs.init_cache,
            )

        bs = self.embedding(bs)
        bs = self.trunk(bs)
        bs = self.readout(bs)

        if self.caterpillar_length > 0:
            bs = BracketedSequence(
                F.pad(bs.x, (0, 0, -self.caterpillar_length, -self.caterpillar_length)),
                bs.first - self.caterpillar_length,
                original_nb,
                bs.init_cache,
            )

        return bs

    # ar_mask is a tensor with 0s and 1s, of same shape as input, with
    # 1s where tokens should be generated. The others are kept
    # unchanged.

    def masked_inplace_autoregression(
        self,
        input_src,
        ar_mask_src,
        forbidden_tokens=None,
        deterministic_synthesis=False,
    ):
        input = input_src.to(self.readout.f.weight.device)
        ar_mask = ar_mask_src.to(self.readout.f.weight.device)
        to_generate = (ar_mask.sum(0) > 0).nonzero()
        if to_generate.min() > 0:
            self(
                BracketedSequence(input, 0, to_generate.min(), True)
            )  # Needed to initialize the model's cache
        for s in range(to_generate.min(), to_generate.max() + 1):
            output = self(BracketedSequence(input, s, 1, s == 0)).x
            logits = output[:, s]
            if forbidden_tokens is not None:
                logits = logits.masked_fill(forbidden_tokens, float("-inf"))
            if deterministic_synthesis:
                t_next = logits.argmax(1)
            else:
                dist = torch.distributions.categorical.Categorical(logits=logits)
                t_next = dist.sample()
            input[:, s] = ar_mask[:, s] * t_next + (1 - ar_mask[:, s]) * input[:, s]

        input_src.copy_(input)

    def reset_inner_loss(self):
        for m in self.modules():
            if m is not self and hasattr(m, "reset_inner_loss"):
                m.reset_inner_loss()

    def get_inner_loss(self):
        l = torch.tensor([0.0], device=self.readout.f.weight.device)
        for m in self.modules():
            if m is not self and hasattr(m, "get_inner_loss"):
                l += m.get_inner_loss()
        return l

    def record_attention(self, v=True):
        for m in self.modules():
            if isinstance(m, QKVAttention):
                m.record_attention = v

    def retrieve_attention(self):
        a = []
        for m in self.modules():
            if isinstance(m, QKVAttention):
                a.append(m.a)
        return a


######################################################################

if __name__ == "__main__":
    print("Basic check.")

    m = Caterpillar(
        dim_model=4,
        dim_qk=3,
        dim_v=7,
        nb_heads=1,
        caterpillar_length=7,
        caterpillar_height=3,
        attention_dropout=0.0,
    )

    m.reset_inner_loss()
    x = torch.randn(1, 21 + 2 * 7, 4)
    y1 = m(BracketedSequence(x, first=7, nb=21, init_cache=True)).x[:, 7:28]
    y2 = m(BracketedSequence(x, first=7, nb=21, init_cache=True)).x[:, 7:28]
    y3a = m(BracketedSequence(x, first=7, nb=14, init_cache=True)).x[:, 7:21]
    y3b = m(BracketedSequence(x, first=21, nb=7, init_cache=False)).x[:, 21:28]
    print((y1 - y2).abs().max())
    print((y1 - torch.cat([y3a, y3b], dim=1)).abs().max())
    exit(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocabulary_size = 128
    x = torch.randint(vocabulary_size, (6, 1024))

    model = MyGPT(
        vocabulary_size=vocabulary_size,
        dim_model=512,
        dim_keys=64,
        dim_hidden=2048,
        nb_heads=8,
        nb_lines=128,
        nb_blocks=12,
        dropout=0.1,
        causal=True,
    )

    x = x.to(device)
    model.to(device)

    import time, sys

    # import torchvision.models as models
    # from torch.profiler import profile, record_function, ProfilerActivity

    # with profile(activities=[ProfilerActivity.CPU,  ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
    # with record_function("model_inference"):

    model.eval()
    for i in range(3):
        start_time = time.perf_counter()
        for k in range(10):
            model(BracketedSequence(x))
        duration = time.perf_counter() - start_time
        print(duration)
        sys.stdout.flush()

    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # print("##############################################################")
    # y2 = torch.randn_like(y1)
    # for s in range(x.size(1)):
    # z = model(BracketedSequence(x, s, 1))
    # y2[:, s : s + 1] = z.slice()

    # print(f"error={((y1 - y2).norm() / (y1.norm() + y2.norm())).item()}")

######################################################################
