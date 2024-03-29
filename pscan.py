#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

import torch

######################################################################


class PScan(torch.autograd.Function):
    # Given A is NxTxMx1 and X is NxTxMxD, expands A and X in
    # place in O(T), and O(log(T)) if not core-bounded, so that
    #
    # Y[:, 0] = Y_init
    # Y[:, t] = A[:, t] * Y[:, t-1] + X[:, t]
    #
    # can be computed as
    #
    # Y[:, t] = A[:, t] * Y_init + X[:, t]

    @staticmethod
    def expand_(A, X):
        # Unrolling gains ~8% speed

        if A.size(1) > 4:
            T = 2 * (A.size(1) // 2)
            Aa = A[:, :T].view(A.size(0), T // 2, 2, -1, 1)
            Xa = X[:, :T].view(X.size(0), T // 2, 2, -1, X.size(-1))
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Aa[:, :, 1].mul_(Aa[:, :, 0])
            PScan.expand_(Aa[:, :, 1], Xa[:, :, 1])
            Xa[:, 1:, 0].add_(Aa[:, 1:, 0].mul(Xa[:, :-1, 1]))
            Aa[:, 1:, 0].mul_(Aa[:, :-1, 1])
            if T < A.size(1):
                X[:, -1].add_(A[:, -1].mul(X[:, -2]))
                A[:, -1].mul_(A[:, -2])
        elif A.size(1) == 2:
            X[:, 1].add_(A[:, 1].mul(X[:, 0]))
            A[:, 1].mul_(A[:, 0])
        elif A.size(1) == 3:
            X[:, 1].add_(A[:, 1].mul(X[:, 0]))
            A[:, 1].mul_(A[:, 0])
            X[:, 2].add_(A[:, 2].mul(X[:, 1]))
            A[:, 2].mul_(A[:, 1])
        elif A.size(1) == 4:
            X[:, 1].add_(A[:, 1].mul(X[:, 0]))
            A[:, 1].mul_(A[:, 0])
            X[:, 2].add_(A[:, 2].mul(X[:, 1]))
            A[:, 2].mul_(A[:, 1])
            X[:, 3].add_(A[:, 3].mul(X[:, 2]))
            A[:, 3].mul_(A[:, 2])

    @staticmethod
    def acc_rev_(A, X):
        if A.size(1) > 4:
            T = 2 * (X.size(1) // 2)
            Aa = A[:, -T:].view(A.size(0), T // 2, 2, -1, 1)
            Xa = X[:, -T:].view(X.size(0), T // 2, 2, -1, X.size(-1))
            Xa[:, :, 0].add_(Aa[:, :, 1].mul(Xa[:, :, 1]))
            B = Aa[:, :, 0].clone()
            B[:, 1:].mul_(Aa[:, :-1, 1])
            PScan.acc_rev_(B, Xa[:, :, 0])
            Xa[:, :-1, 1].add_(Aa[:, 1:, 0].mul(Xa[:, 1:, 0]))
            if T < A.size(1):
                X[:, 0].add_(A[:, 1].mul(X[:, 1]))
        elif A.size(1) == 2:
            X[:, 0].add_(A[:, 1].mul(X[:, 1]))
        elif A.size(1) == 3:
            X[:, 1].add_(A[:, 2].mul(X[:, 2]))
            X[:, 0].add_(A[:, 1].mul(X[:, 1]))
        elif A.size(1) == 4:
            X[:, 2].add_(A[:, 3].mul(X[:, 3]))
            X[:, 1].add_(A[:, 2].mul(X[:, 2]))
            X[:, 0].add_(A[:, 1].mul(X[:, 1]))

    # A is NxT, X is NxTxD, Y_init is NxD
    #
    # returns Y of same shape as X, with
    #
    # Y[:, t] = A[:, 0] * Y_init   + X[:, 0] if t == 0
    #         = A[:, t] * Y[:, t-1] + X[:, t] otherwise

    @staticmethod
    def forward(ctx, A, X, Y_init):
        ctx.A = A.unsqueeze(-1).clone()
        ctx.Y_init = Y_init[:, None].clone()
        ctx.A_star = ctx.A.clone()
        ctx.X_star = X.clone()
        PScan.expand_(ctx.A_star, ctx.X_star)
        return ctx.A_star * ctx.Y_init + ctx.X_star

    @staticmethod
    def backward(ctx, grad_output):
        U = grad_output * ctx.A_star
        A = ctx.A.clone()
        R = grad_output.clone()
        PScan.acc_rev_(A, R)
        Q = ctx.Y_init.expand_as(ctx.X_star).clone()
        Q[:, 1:].mul_(ctx.A_star[:, :-1]).add_(ctx.X_star[:, :-1])
        return (Q * R).sum(-1), R, U.sum(dim=1)


pscan = PScan.apply


def naive_pscan(A, X, Y_init):
    y = Y_init
    s = 0

    for k in range(A.size(1)):
        y = A[:, k, None] * y + X[:, k]
        s = s + y

    s = s.sum()


######################################################################

if __name__ == "__main__":
    import time, sys

    ######################################################################

    N, T, D = 16, 4096, 32

    for r in range(timing.size(0)):
        A = 0.9 + 0.1 * torch.rand(N, T, dtype=torch.float64).requires_grad_()
        X = torch.randn(N, T, D, dtype=torch.float64).requires_grad_()
        Y_init = torch.randn(N, D, dtype=torch.float64).requires_grad_()

        start_time = time.perf_counter()
        for _ in range(1000):
            Y = pscan(A, X, Y_init)
        duration = time.perf_counter() - start_time

    ######################################################################

    # A = torch.rand(17, 12, 3)
    # X = torch.rand(17, 12, 3, 11)
    # Y_init = torch.rand(17, 3, 11)
    # Y = pscan(A, X, Y_init)

    # exit(0)

    err = 0
    timing = torch.empty(10)

    for r in range(timing.size(0)):
        N, T, D = 2, 1120, 3

        # T = torch.randint(10, (1,)).item() + 1

        A = 0.9 + 0.1 * torch.rand(N, T, dtype=torch.float64).requires_grad_()
        X = torch.randn(N, T, D, dtype=torch.float64).requires_grad_()
        Y_init = torch.randn(N, D, dtype=torch.float64).requires_grad_()

        # Iterative implementation

        y = Y_init
        s = 0

        for k in range(A.size(1)):
            y = A[:, k, None] * y + X[:, k]
            s = s + y

        s = s.sum()

        gA_ref, gX_ref, gY_init_ref = torch.autograd.grad(
            s, (A, X, Y_init), retain_graph=True
        )

        # parallel scan

        start_time = time.perf_counter()
        for _ in range(1000):
            Y = pscan(A, X, Y_init)
        duration = time.perf_counter() - start_time

        print(f"duration {duration}")
        timing[r] = duration

        s = Y.sum()

        gA, gX, gY_init = torch.autograd.grad(s, (A, X, Y_init), retain_graph=True)

        err = max(
            [
                err,
                (gA - gA_ref).abs().max(),
                (gX - gX_ref).abs().max(),
                (gY_init - gY_init_ref).abs().max(),
            ]
        )

        # Y1 = pscan(A[:, : T // 2], X[:, : T // 2], Y_init)
        # Y2 = pscan(A[:, T // 2 :], X[:, T // 2 :], Y1[:, -1])

        # print((Y - torch.cat([Y1, Y2], dim=1)).abs().max())

    print(f"err={err:.2e} duration={timing.mean():.2e} (+/- {timing.std():.2e})")
