#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

import math

import torch, torchvision

from torch import nn
from torch.nn import functional as F

######################################################################


def rpl_exec(program, stack):
    stack = stack.copy()
    for op in program:
        if op == "add":
            if len(stack) > 1:
                a, b = stack.pop(), stack.pop()
                stack.append(a + b)
        elif op == "min":
            if len(stack) > 1:
                a, b = stack.pop(), stack.pop()
                stack.append(min(a, b))
        elif op == "max":
            if len(stack) > 1:
                a, b = stack.pop(), stack.pop()
                stack.append(max(a, b))
        elif op == "swp":
            if len(stack) > 1:
                a, b = stack.pop(), stack.pop()
                stack.append(a)
                stack.append(b)
        elif op == "rep":
            if len(stack) > 1:
                a, b = stack.pop(), stack.pop()
                stack += [b] * a
        elif op == "dup":
            if len(stack) > 0:
                a = stack.pop()
                stack.append(a)
                stack.append(a)
        elif op == "del":
            if len(stack) > 0:
                a = stack.pop()
        else:
            raise ValueError(f"Unknown instruction {op}")

    return stack


rpl_ops = ["add", "min", "max", "swp", "rep", "dup", "del"]

######################################################################


def generate(
    nb_starting_values=3, nb_result_values_max=None, max_input=9, prog_len=6, nb_runs=5
):
    prog_len = (1 + torch.randint(2 * prog_len, (1,))).clamp(max=prog_len).item()

    while True:
        no_empty_stack = True
        prog = [rpl_ops[k] for k in torch.randint(len(rpl_ops), (prog_len,))]

        result = []
        for _ in range(nb_runs):
            stack = [
                x.item() for x in torch.randint(max_input + 1, (nb_starting_values,))
            ]
            result_stack = rpl_exec(prog, stack)
            if len(result_stack) == 0:
                no_empty_stack = False
            result = result + ["<in>"] + stack + ["<out>"] + result_stack

        result = result + ["<prg>"] + prog
        result = result + ["<end>"]

        if no_empty_stack and (
            nb_result_values_max is None or len(result_stack) <= nb_result_values_max
        ):
            break

    return result


def next_marker(seq, tokens, start=0):
    pos = None
    for t in tokens:
        try:
            i = seq.index(t, start)
            if pos is None or i < pos:
                pos = i
        except ValueError:
            pass
    return pos


def decompose(seq):
    io = []
    k = 0
    while seq[k] == "<in>":
        o = next_marker(seq, ["<out>"], start=k + 1)
        if o is None:
            raise ValueError("Missing output markers (should be correct in the prompt)")
        e = next_marker(seq, ["<in>", "<prg>"], start=o)
        if e is None:
            raise ValueError(
                "Missing input/output markers (should be correct in the prompt)"
            )
        try:
            io.append(
                ([int(x) for x in seq[k + 1 : o]], [int(x) for x in seq[o + 1 : e]])
            )
        except ValueError:
            raise ValueError(
                "Invalid input/output value (should be correct in the prompt)"
            )

        k = e

    if seq[k] == "<prg>":
        e = next_marker(seq, ["<end>"], start=k)
        if e is None:
            prog = []
        else:
            prog = seq[k + 1 : e]
    else:
        raise ValueError("Missing <prg> (it should be in the prompt)")

    return prog, io


def stack_distance(target_stack, result_stack):
    return abs(len(result_stack) - len(target_stack)) + sum(
        [0 if x == y else 1 for x, y in zip(result_stack, target_stack)]
    )


def compute_nb_errors(seq):
    prog, io = decompose(seq)

    nb_total, nb_errors = 0, 0

    stacks = []

    if len(set(prog) - set(rpl_ops)) > 0:
        # Program is not valid, we count 100% error
        for start_stack, target_stack in io:
            stacks.append((start_stack, target_stack, ["N/A"], False))
            nb_total += len(target_stack)
            nb_errors += len(target_stack)

    else:
        # Program is valid
        for start_stack, target_stack in io:
            result_stack = rpl_exec(prog, start_stack)
            nb_total += len(target_stack)
            e = stack_distance(target_stack, result_stack)
            nb_errors += e
            stacks.append((start_stack, target_stack, result_stack, e == 0))

    return nb_total, nb_errors, prog, stacks


######################################################################

if __name__ == "__main__":
    seq = generate()
    print(seq)
    seq[3] = 7
    print(seq)
    print(compute_nb_errors(seq))
