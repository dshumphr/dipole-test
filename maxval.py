#!/usr/bin/env python

import torch

######################################################################


def baseline(X, V):
    Y = X.new(X.size())
    W = V.new(V.size())
    for t in range(X.size(1)):
        if t == 0:
            Y[:, t] = X[:, t]
            W[:, t] = V[:, t]
        else:
            m = (V[:, t] >= W[:, t - 1] - 1).long()
            Y[:, t] = m * X[:, t] + (1 - m) * Y[:, t - 1]
            W[:, t] = m * V[:, t] + (1 - m) * (W[:, t - 1] - 1)

    return Y, W


######################################################################


def pscan(X, V, s=1):
    if X.size(1) == 1:
        return

    T = 2 * (X.size(1) // 2)

    Xf = X[:, :T].view(X.size(0), X.size(1) // 2, 2, X.size(2))
    Vf = V[:, :T].view(V.size(0), V.size(1) // 2, 2)

    # [:, :, 0] < [:, :, 1]
    m = (Vf[:, :, 0] - s >= Vf[:, :, 1]).long()
    Vf[:, :, 1] = m * (Vf[:, :, 0] - s) + (1 - m) * Vf[:, :, 1]
    m = m[:, :, None]
    Xf[:, :, 1] = m * Xf[:, :, 0] + (1 - m) * Xf[:, :, 1]

    pscan(Xf[:, :, 1], Vf[:, :, 1], s * 2)

    # [:, :-1, 1] < [:, 1:, 0]
    m = (Vf[:, :-1, 1] - s >= Vf[:, 1:, 0]).long()
    Vf[:, 1:, 0] = m * (Vf[:, :-1, 1] - s) + (1 - m) * Vf[:, 1:, 0]
    m = m[:, :, None]
    Xf[:, 1:, 0] = m * Xf[:, :-1, 1] + (1 - m) * Xf[:, 1:, 0]

    if T < X.size(1):
        # [:, -2] < [:, -1]
        m = (V[:, -2] - s >= V[:, -1]).long()
        V[:, -1] = m * (V[:, -2] - s) + (1 - m) * V[:, -1]
        m = m[:, None]
        X[:, -1] = m * X[:, -2] + (1 - m) * X[:, -1]


######################################################################


def pscan_diff(X, V, s=1):
    if X.size(1) == 1:
        return X, V

    T = 2 * (X.size(1) // 2)

    Xf = X[:, :T].view(X.size(0), X.size(1) // 2, 2, X.size(2))
    Vf = V[:, :T].view(V.size(0), V.size(1) // 2, 2)

    Xr = X.new(X.size())
    Vr = V.new(V.size())
    Xrf = Xr[:, :T].view(Xr.size(0), Xr.size(1) // 2, 2, Xr.size(2))
    Vrf = Vr[:, :T].view(Vr.size(0), Vr.size(1) // 2, 2)

    # [:, :, 0] < [:, :, 1]
    m = (Vf[:, :, 0] - s >= Vf[:, :, 1]).long()
    Vv = m * (Vf[:, :, 0] - s) + (1 - m) * Vf[:, :, 1]
    m = m[:, :, None]
    Xx = m * Xf[:, :, 0] + (1 - m) * Xf[:, :, 1]

    Xrf[:, :, 1], Vrf[:, :, 1] = pscan_diff(Xx, Vv, s * 2)

    Xr[:, 0] = X[:, 0]
    Vr[:, 0] = V[:, 0]

    # [:, :-1, 1] < [:, 1:, 0]
    m = (Vrf[:, :-1, 1] - s >= Vf[:, 1:, 0]).long()
    Vrf[:, 1:, 0] = m * (Vrf[:, :-1, 1] - s) + (1 - m) * Vf[:, 1:, 0]
    m = m[:, :, None]
    Xrf[:, 1:, 0] = m * Xrf[:, :-1, 1] + (1 - m) * Xf[:, 1:, 0]

    if T < X.size(1):
        # [:, -2] < [:, -1]
        m = (V[:, -2] - s >= V[:, -1]).long()
        Vr[:, -1] = m * (V[:, -2] - s) + (1 - m) * V[:, -1]
        m = m[:, None]
        Xr[:, -1] = m * X[:, -2] + (1 - m) * X[:, -1]

    return Xr, Vr


######################################################################

if __name__ == "__main__":
    N = 1
    T = 513
    D = 2

    X = torch.randn(N, T, D, dtype=torch.float64).requires_grad_()
    V = torch.rand(N, T, dtype=torch.float64) * 10

    X0, V0 = baseline(X, V)

    # print("########### X0 V0 ###########################################")
    # print(V0)
    # print(X0)

    X1, V1 = pscan_diff(X, V)

    # print("########### X V ############################################")
    # print(V)
    # print(X)

    print("ERROR", ((X0 - X1).abs().max() + (V0 - V1).abs().max()).item())

    exit(0)

    # s = X1.sum()
    # print(torch.autograd.grad(s, X))

    # with open("/tmp/v.dat", "w") as f:
    # for t in range(T):
    # f.write(f"{V1[0,t].item()}\n")

    Y = torch.randn(1, 1, D)
    X = torch.randn(
        N, T, D
    )  # * 0.1 + (torch.rand(N,T,1).sort(dim=1).indices==0).float() * Y
    V = torch.rand(N, T).requires_grad_()

    optimizer = torch.optim.SGD([V], lr=1e-2)

    for k in range(1000):
        X1, V1 = X.clone(), V.clone()
        pscan(X, V, X1, V1)
        # X1=X1*(1+V1-V1.detach())[:,:,None]
        loss = (X1[:, -1:] - Y).pow(2).mean()
        print(k, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
