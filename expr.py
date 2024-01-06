#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

import math, re

import torch, torchvision

from torch import nn
from torch.nn import functional as F


def random_var(nb_variables=None, variables=None):
    if variables is None:
        return chr(ord("A") + torch.randint(nb_variables, (1,)).item())
    else:
        l = list(variables)
        return l[torch.randint(len(l), (1,)).item()]


def random_expr(variables, operand_max, budget):
    if budget <= 5:
        op = torch.randint(2, (1,)).item()
        if op == 0 and len(variables) > 0:
            return random_var(variables=variables)
        else:
            return str(torch.randint(operand_max + 1, (1,)).item())
    else:
        op = torch.randint(3, (1,)).item()
        if op == 0:
            e = random_expr(variables, operand_max, budget - 2)
            if ("+" in e or "-" in e or "*" in e) and (e[0] != "(" or e[-1] != ")"):
                return "(" + e + ")"
            else:
                return e
        else:
            b = 2 + torch.randint(budget - 5, (1,)).item()
            e1 = random_expr(variables, operand_max, b)
            e2 = random_expr(variables, operand_max, budget - b - 1)
            if op == 1:
                return e1 + "+" + e2
            elif op == 2:
                return e1 + "*" + e2


def generate_program(nb_variables, operand_max, length):
    s = ""
    variables = set()

    while len(s) < length:
        v = random_var(nb_variables=nb_variables)
        s += v + "=" + random_expr(variables, operand_max, budget=20) + ";"
        variables.add(v)

    return s, variables


def generate_sequences(nb, nb_variables=5, length=20, operand_max=9, result_max=99):
    assert nb_variables <= 26
    sequences = []

    for n in range(nb):
        # We take length itself half of the time, and uniform between
        # 1 and length otherwise. The actual length can be slightly
        # greater

        l = min(length, 1 + torch.randint(length * 2, (1,)).item())
        result = None
        while result == None or max(result.values()) > result_max:
            p, v = generate_program(nb_variables, operand_max, l)
            v = ", ".join(['"' + v + '": ' + v for v in v])
            ldict = {}
            exec(p + "result={" + v + "}", globals(), ldict)
            result = ldict["result"]

        k = list(result.keys())
        k.sort()
        sequences.append(p + " " + "".join([v + ":" + str(result[v]) + ";" for v in k]))

    return sequences


def extract_results(seq):
    f = lambda a: (a[0], -1 if a[1] == "" else int(a[1]))
    results = [
        dict([f(tuple(x.split(":"))) for x in re.findall("[A-Z]:[0-9]*", s)])
        for s in seq
    ]
    return results


if __name__ == "__main__":
    import time

    start_time = time.perf_counter()
    sequences = generate_sequences(1000, length=40)
    end_time = time.perf_counter()
    for s in sequences[:10]:
        print(s)
    print(f"{len(sequences) / (end_time - start_time):.02f} samples per second")

    print(extract_results(sequences[:10]))
