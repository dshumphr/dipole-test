#!/usr/bin/env python

import math

import torch, torchvision

from torch import nn
from torch.nn import functional as F

######################################################################


class Problem:
    def generate_sequences(self, nb):
        pass

    def seq2str(self, seq):
        return "[NOT IMPLEMENTED]"

    def compute_nb_correct(self, input, ar_mask, result):
        nb_total = ar_mask.sum().item()
        nb_correct = ((result == input).long() * ar_mask).sum().item()
        return nb_total, nb_correct


####################


class ProblemDegradation(Problem):
    def __init__(self, nb_state_tokens=5, nb_time_steps=12, value_max=25, hard=False):
        assert value_max // nb_state_tokens >= 2
        self.nb_state_tokens = nb_state_tokens
        self.nb_time_steps = nb_time_steps
        self.value_max = value_max
        self.hard = hard

    def generate_sequences(self, nb):
        x = (
            torch.rand(nb, self.nb_state_tokens).sort(dim=-1).indices == 0
        ).long() * self.value_max
        seq = [x]

        for t in range(self.nb_time_steps - 1):
            v = (torch.rand(x.size()).sort(dim=-1).indices + 1) * (x >= 2).long()
            u = (v.max(dim=-1, keepdim=True).values == v).long()
            n = (
                (u * x)
                .minimum(2 + torch.randint(self.value_max // 4 - 2, x.size()))
                .sum(dim=-1, keepdim=True)
            )
            m = 1 + ((n - 1) * torch.rand(n.size())).long()
            x = (
                x
                + m * u.roll(shifts=-1, dims=-1)
                - n * u
                + (n - m) * u.roll(shifts=1, dims=-1)
            )
            seq.append(x)

        if self.hard:
            seq.reverse()

        seq = torch.cat(seq, dim=1)
        return seq, seq.new_full(seq.size(), 1, dtype=torch.int64)

    def compute_nb_correct(self, input, ar_mask, result):
        nb_total = result.size(0)
        nb_correct = 0
        e = result.new_zeros(self.nb_state_tokens)

        for seq in result:
            states = list(seq.split(self.nb_state_tokens))
            if self.hard:
                states.reverse()

            d = states[0]
            j = d.sort(descending=True).indices[0]
            e.zero_()
            e[j] = self.value_max
            if (d - e).abs().sum() == 0:
                nb_errors = 0
                for k in range(len(states) - 1):
                    d = states[k + 1] - states[k]
                    j = d.sort(descending=False).indices[0]
                    if (
                        d[j] == 0
                        or d[j] > self.value_max // 4
                        or d[(j + 1) % e.size(0)] <= 0
                        or d[(j + 1) % e.size(0)] >= -d[j]
                    ):
                        nb_errors += 1
                    else:
                        e.zero_()
                        e[j] = d[j]
                        e[(j + 1) % e.size(0)] = d[(j + 1) % e.size(0)]
                        e[(j - 1) % e.size(0)] = -d[(j + 1) % e.size(0)] - d[j]
                        if (d - e).abs().sum() > 0:
                            nb_errors += 1
                if nb_errors == 0:
                    nb_correct += 1

        return nb_total, nb_correct

    def seq2str(self, seq):
        return " | ".join(
            [" ".join([f"{x:02d}" for x in s]) for s in seq.split(self.nb_state_tokens)]
        )


####################


class ProblemMemory(Problem):
    def __init__(self, len_total=32):
        self.len_total = len_total
        self.max_len_pattern = 5
        self.nb_noise_tokens = 10
        self.start_pattern_token = 0
        self.end_pattern_token = 1
        self.start_result_token = 2
        self.end_result_token = 3
        self.token_string = "[]<>" + "".join(
            [chr(ord("a") + k) for k in range(self.nb_noise_tokens)]
        )

    def generate_sequences(self, nb):
        sequences = (
            torch.randint(self.nb_noise_tokens, (nb, self.len_total))
            + self.end_result_token
            + 1
        )
        len_patterns = torch.randint(self.max_len_pattern, (nb,)) + 1
        pattern_positions = torch.randint(
            self.len_total - (5 + 2 * self.max_len_pattern), (nb,)
        )
        k = self.len_total - (3 + self.max_len_pattern)
        for i in range(nb):
            l = len_patterns[i]
            j = pattern_positions[i]
            sequences[i, j] = self.start_pattern_token
            sequences[i, j + l + 2] = self.end_pattern_token
            sequences[i, k] = self.start_result_token
            sequences[i, k + l + 2] = self.end_result_token
            sequences[i, k + 1 : k + 2 + l] = sequences[i, j + 1 : j + 2 + l]

        j = torch.arange(self.len_total)[None, :]
        ar_mask = (j > k).long() * (j <= k + 1 + len_patterns[:, None]).long()

        return sequences, ar_mask

    def seq2str(self, seq):
        return "".join(self.token_string[x.item()] for x in seq)


class ProblemTwoTargets(Problem):
    def __init__(self, len_total=10, len_targets=3):
        assert len_targets >= 3
        assert len_total >= 3 * len_targets - 1
        self.len_total = len_total
        self.len_targets = len_targets

    def generate_sequences(self, nb):
        k = torch.arange(self.len_total)[None, :]
        s = torch.randint(10, (nb, self.len_total))
        l = torch.rand(nb, self.len_total)
        l = l * (k <= self.len_total - self.len_targets).long()
        k1 = l.argmax(dim=1, keepdim=True)
        m = (k != k1).long() * (k != k1 + self.len_targets - 1).long()
        s = s * m + 10 * (1 - m)
        l = l * (
            1
            - (k + self.len_targets - 1 >= k1).long()
            * (k < k1 + self.len_targets).long()
        )
        k2 = l.argmax(dim=1, keepdim=True)
        m = (k != k2).long() * (k != k2 + self.len_targets - 1).long()
        s = s * m + 11 * (1 - m)
        a1 = s.gather(dim=1, index=k1 + 1 + torch.arange(self.len_targets - 2)[None, :])
        a2 = s.gather(dim=1, index=k2 + 1 + torch.arange(self.len_targets - 2)[None, :])
        sequences = torch.cat(
            (
                s,
                torch.full((nb, 1), 12),
                a1,
                torch.full((nb, 1), 12),
                a2,
                torch.full((nb, 1), 12),
            ),
            1,
        )
        ar_mask = (sequences == 12).long()
        ar_mask = (ar_mask.cumsum(1) - ar_mask).clamp(max=1)
        return sequences, ar_mask

    def seq2str(self, seq):
        return "".join("0123456789-+|"[x.item()] for x in seq)


####################


class ProblemByHeart(Problem):
    def __init__(self, nb_sentences=100, len_prompt=8, len_result=8):
        self.seq = torch.randint(10, (nb_sentences, len_prompt + 1 + len_result))
        self.seq[:, len_prompt] = 10

    def generate_sequences(self, nb):
        sequences = self.seq[torch.randint(self.seq.size(0), (nb,))]
        ar_mask = (sequences == 10).long()
        ar_mask = (ar_mask.cumsum(1) - ar_mask).clamp(max=1)
        return sequences, ar_mask

    def seq2str(self, seq):
        return "".join("0123456789|"[x.item()] for x in seq)


####################


class ProblemLearnOperator(Problem):
    def __init__(self, nb_operators=100, len_source=6, len_result=9):
        self.len_source = len_source
        self.len_result = len_result
        self.len_nb_operator = int(math.log(nb_operators) / math.log(10)) + 1
        self.operators = F.one_hot(
            torch.rand(nb_operators, len_result, len_source).argmax(-1),
            num_classes=len_source,
        )

    def generate_sequences(self, nb):
        nb_operators = torch.randint(self.operators.size(0), (nb,))
        operators = self.operators[nb_operators]
        nb_operators = (
            nb_operators[:, None]
            // 10 ** torch.arange(self.len_nb_operator - 1, -1, -1)
        ) % 10
        marker1 = torch.full((nb, 1), 10)
        source = torch.rand(nb, 10).sort(dim=1).indices[:, : self.len_source]
        marker2 = torch.full((nb, 1), 11)
        result = operators.bmm(source[:, :, None]).squeeze(-1)
        sequences = torch.cat((nb_operators, marker1, source, marker2, result), 1)
        ar_mask = (sequences == 11).long()
        ar_mask = (ar_mask.cumsum(1) - ar_mask).clamp(max=1)
        return sequences, ar_mask

    def seq2str(self, seq):
        return "".join("0123456789|>"[x.item()] for x in seq)


####################


class ProblemGuessOperator(Problem):
    def __init__(self, len_source=5, len_result=8):
        self.len_source = len_source
        self.len_result = len_result

    def generate_sequences(self, nb):
        operators = F.one_hot(
            torch.rand(nb, self.len_result, self.len_source).argmax(-1),
            num_classes=self.len_source,
        )
        source1 = torch.rand(nb, 10).sort(dim=1).indices[:, : self.len_source]
        marker1 = torch.full((nb, 1), 10)
        result1 = operators.bmm(source1[:, :, None]).squeeze(-1)
        marker2 = torch.full((nb, 1), 11)
        source2 = torch.randint(10, (nb, self.len_source))
        marker3 = torch.full((nb, 1), 12)
        result2 = operators.bmm(source2[:, :, None]).squeeze(-1)

        sequences = torch.cat(
            (source1, marker1, result1, marker2, source2, marker3, result2), 1
        )
        ar_mask = (sequences == 12).long()
        ar_mask = (ar_mask.cumsum(1) - ar_mask).clamp(max=1)
        return sequences, ar_mask

    def seq2str(self, seq):
        return "".join("0123456789>|~"[x.item()] for x in seq)


####################


class ProblemAddition(Problem):
    def __init__(self, nb_digits=10, zero_padded=False, inverted_result=False):
        self.nb_digits = nb_digits
        self.zero_padded = zero_padded
        self.inverted_result = inverted_result
        self.char2id = dict([(c, n) for n, c in enumerate("0123456789+=$")])
        self.id2char = dict([(n, c) for c, n in self.char2id.items()])

    def tensorize(self, strings):
        len_max = max([len(x) for x in strings])
        return torch.cat(
            [
                torch.tensor(
                    [
                        [self.char2id[c] for c in s + "$" * (len_max - len(s))]
                        for s in strings
                    ]
                )
            ],
            0,
        )

    def generate_sequences(self, nb):
        sequences = []
        for k in range(nb):
            a, b = torch.randint(10**self.nb_digits, (2,))
            c = a + b
            a, b, c = str(a.item()), str(b.item()), str(c.item())
            if self.zero_padded:
                a = "0" * (self.nb_digits - len(a)) + a
                b = "0" * (self.nb_digits - len(b)) + b
                c = "0" * (self.nb_digits + 1 - len(c)) + c
            if self.inverted_result:
                c = c[::-1]
            sequences.append(f"{a}+{b}={c}$")

        sequences = self.tensorize(sequences)
        ar_mask = (sequences == self.char2id["="]).long()
        ar_mask = (ar_mask.cumsum(1) - ar_mask).clamp(max=1)
        return sequences, ar_mask

    def seq2str(self, seq):
        return "".join(self.id2char[x.item()] for x in seq)


####################


class ProblemMixing(Problem):
    def __init__(
        self, height=4, width=4, nb_time_steps=9, hard=False, random_start=True
    ):
        self.height = height
        self.width = width
        self.nb_time_steps = nb_time_steps
        self.hard = hard
        self.random_start = random_start

    def start_random(self, nb):
        y = torch.arange(self.height * self.width).reshape(1, -1).expand(nb, -1)

        if self.random_start:
            i = (
                torch.arange(self.height)
                .reshape(1, -1, 1)
                .expand(nb, self.height, self.width)
            )
            j = (
                torch.arange(self.width)
                .reshape(1, 1, -1)
                .expand(nb, self.height, self.width)
            )

            ri = torch.randint(self.height, (nb,)).reshape(nb, 1, 1)
            rj = torch.randint(self.width, (nb,)).reshape(nb, 1, 1)

            m = 1 - torch.logical_or(i == ri, j == rj).long().flatten(1)

            y = y * m + self.height * self.width * (1 - m)

        y = y.reshape(nb, self.height, self.width)

        return y

    def start_error(self, x):
        if self.random_start:
            i = (
                torch.arange(self.height, device=x.device)
                .reshape(1, -1, 1)
                .expand_as(x)
            )
            j = torch.arange(self.width, device=x.device).reshape(1, 1, -1).expand_as(x)

            ri = (
                (x == self.height * self.width)
                .long()
                .sum(dim=-1)
                .argmax(-1)
                .view(-1, 1, 1)
            )
            rj = (
                (x == self.height * self.width)
                .long()
                .sum(dim=-2)
                .argmax(-1)
                .view(-1, 1, 1)
            )

            m = 1 - torch.logical_or(i == ri, j == rj).long().flatten(1)
        else:
            m = 1

        x = x.flatten(1)
        u = torch.arange(self.height * self.width, device=x.device).reshape(1, -1)

        d = (x - (m * u + (1 - m) * self.height * self.width)).abs().sum(-1)

        return d

    def moves(self, x):
        y = (
            x[:, None, :, :]
            .expand(-1, self.height * 2 + self.width * 2, -1, -1)
            .clone()
        )
        k = 0

        for i in range(self.height):
            y[:, k, i, :] = y[:, k, i, :].roll(dims=-1, shifts=-1)
            k += 1
            y[:, k, i, :] = y[:, k, i, :].roll(dims=-1, shifts=1)
            k += 1

        for j in range(self.width):
            y[:, k, :, j] = y[:, k, :, j].roll(dims=-1, shifts=-1)
            k += 1
            y[:, k, :, j] = y[:, k, :, j].roll(dims=-1, shifts=1)
            k += 1

        return y

    def generate_sequences(self, nb):
        x = self.start_random(nb)

        seq = [x.flatten(1)]

        for t in range(self.nb_time_steps - 1):
            y = self.moves(x)
            x = y[torch.arange(nb), torch.randint(y.size(1), (nb,))]
            seq.append(x.flatten(1))

        if self.hard:
            seq.reverse()

        seq = torch.cat(seq, dim=1)
        return seq, seq.new_full(seq.size(), 1, dtype=torch.int64)

    def compute_nb_correct(self, input, ar_mask, result):
        a = [
            x.reshape(result.size(0), self.height, self.width)
            for x in result.split(self.height * self.width, dim=1)
        ]
        if self.hard:
            a.reverse()

        x = a[0]

        d = self.start_error(x)

        for t in range(self.nb_time_steps - 1):
            x0, x = a[t], a[t + 1]
            y = self.moves(x0)
            d = d + (x[:, None] - y).abs().sum((-1, -2)).min(dim=-1).values

        nb_total, nb_correct = result.size(0), (d == 0).long().sum().item()

        return nb_total, nb_correct

    def seq2str(self, seq):
        return " | ".join(
            [
                " ".join(
                    [
                        "-".join(
                            [
                                f"{x:02d}" if x < self.height * self.width else "**"
                                for x in s
                            ]
                        )
                        for s in r.split(self.width)
                    ]
                )
                for r in seq.split(self.height * self.width)
            ]
        )


####################

if __name__ == "__main__":
    p = ProblemMixing(height=3, width=3, random_start=False)

    s, m = p.generate_sequences(10000)
    for x in s[:5]:
        print(p.seq2str(x))
    print(p.compute_nb_correct(None, None, s))
