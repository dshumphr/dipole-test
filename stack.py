#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

import torch, torchvision

######################################################################

# CODE_OP=[0 for push, 1 for pop] + 2 * n_stack
# CODE_VAL=val + 2 * nb_stacks


def generate_sequences(
    nb, nb_steps, nb_stacks, nb_digits, values=None, device=torch.device("cpu")
):
    stack = torch.empty(nb, nb_stacks, nb_steps, dtype=torch.int64)
    stack_counts = torch.zeros(nb, nb_stacks, dtype=torch.int64)
    k = torch.arange(nb)
    result = torch.empty(nb, (1 + nb_digits) * nb_steps, dtype=torch.int64)
    recorded_stack_counts = torch.zeros(
        nb, (1 + nb_digits) * nb_steps, dtype=torch.int64
    )

    for t in range(nb_steps):
        op = torch.randint(2, (nb,))
        st = torch.randint(nb_stacks, (nb,))
        op = op * (stack_counts[k, st] > 0)
        if values is None:
            val_push = torch.randint(10**nb_digits, (nb,))
        else:
            val_push = values[torch.randint(values.size(0), (nb,))]
        val_pop = stack[
            k,
            st,
            (stack_counts[k, st] - 1).clamp(min=0),
        ]
        stack[k, st, stack_counts[k, st]] = val_push
        recorded_stack_counts[:, (1 + nb_digits) * t] = stack_counts[k, st]
        stack_counts[k[op == 0], st[op == 0]] += 1
        stack_counts[k[op == 1], st[op == 1]] -= 1
        result[:, (1 + nb_digits) * t] = st * 2 + op
        for d in range(nb_digits):
            result[:, (1 + nb_digits) * t + 1 + d] = (
                (op * val_pop + (1 - op) * val_push) // (10**d)
            ) % 10 + 2 * nb_stacks

    return result.to(device), recorded_stack_counts.to(device)


def remove_popped_values(seq, nb_stacks, nb_digits):
    m = torch.logical_and(seq % 2 == 1, seq < 2 * nb_stacks).long()
    for d in range(nb_digits):
        k = d + 1
        seq[:, k:] = -m[:, :-k] + (1 - m[:, :-k]) * seq[:, k:]


def seq_to_str(seq, nb_stacks, nb_digits, recorded_stack_counts=None):
    assert seq.size(0) % (1 + nb_digits) == 0
    s = ""
    for t in range(seq.size(0) // (1 + nb_digits)):
        n_op = seq[(1 + nb_digits) * t]
        if t > 0:
            s += " "
        if recorded_stack_counts is not None:
            s += f"[{recorded_stack_counts[(1 + nb_digits)*t]}] "
        s += f"POP" if n_op % 2 == 1 else f"PSH"
        if nb_stacks > 1:
            s += f"_{n_op//2}"
        for d in range(nb_digits):
            if seq[(1 + nb_digits) * t + 1 + d] == -1:
                s += " ?"
            else:
                s += f" {seq[(1 + nb_digits) * t + 1 + d] - 2 * nb_stacks:1d}"
    return s


######################################################################

if __name__ == "__main__":
    nb, nb_steps, nb_stacks, nb_digits = 150000, 20, 2, 1
    seq, recorded_stack_counts = generate_sequences(
        nb=nb,
        nb_steps=nb_steps,
        nb_stacks=nb_stacks,
        nb_digits=nb_digits,
    )

    for n in range(min(10, seq.size(0))):
        print(
            seq_to_str(
                seq[n],
                nb_stacks=nb_stacks,
                nb_digits=nb_digits,
                recorded_stack_counts=recorded_stack_counts[n],
            )
        )
        # print(seq_to_str(seq[n], nb_stacks=nb_stacks, nb_digits=nb_digits))

    print("-- PREPARED FOR TEST -----------------")

    remove_popped_values(seq, nb_stacks, nb_digits)

    for n in range(min(10, seq.size(0))):
        print(seq_to_str(seq[n], nb_stacks=nb_stacks, nb_digits=nb_digits))
