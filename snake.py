#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

import torch, torchvision
import torch.nn.functional as F


def generate_sequences(
    nb, height, width, nb_colors, length, prompt_length, device=torch.device("cpu")
):
    worlds = torch.randint(nb_colors, (nb, height, width), device=device)
    world_prior_visits = torch.zeros(nb, height, width, device=device)

    # nb x 2
    snake_position = torch.cat(
        (
            torch.randint(height, (nb, 1), device=device),
            torch.randint(width, (nb, 1), device=device),
        ),
        1,
    )
    snake_direction = torch.randint(4, (nb,), device=device)
    sequences = torch.empty(nb, 2 * length, device=device, dtype=torch.int64)
    sequences_prior_visits = torch.zeros(
        nb, 2 * length, device=device, dtype=torch.int64
    )
    i = torch.arange(nb, device=device)  # [:,None]

    for l in range(length):
        # nb x 3
        snake_next_direction = torch.cat(
            (
                (snake_direction[:, None] - 1) % 4,
                snake_direction[:, None],
                (snake_direction[:, None] + 1) % 4,
            ),
            1,
        )

        # nb x 3
        vh = (snake_next_direction + 1) % 2 * (snake_next_direction - 1)
        vw = snake_next_direction % 2 * (snake_next_direction - 2)

        # nb x 3 x 2
        snake_next_speed = torch.cat((vh[:, :, None], vw[:, :, None]), 2)
        snake_next_position = snake_position[:, None, :] + snake_next_speed

        # nb x 3
        val = torch.logical_and(
            torch.logical_and(
                snake_next_position[:, :, 0] >= 0, snake_next_position[:, :, 0] < height
            ),
            torch.logical_and(
                snake_next_position[:, :, 1] >= 0, snake_next_position[:, :, 1] < width
            ),
        ).float()
        val = (
            # The multiplicative factors bias toward moving forward
            torch.rand_like(val)
            * val
            * torch.tensor([[1.0, 2.0, 1.0]], device=device)
        )

        # nb
        j = val.argmax(1)
        snake_direction = snake_next_direction[i, j]

        sequences[:, 2 * l] = worlds[i, snake_position[:, 0], snake_position[:, 1]] + 4
        sequences_prior_visits[:, 2 * l] = world_prior_visits[
            i, snake_position[:, 0], snake_position[:, 1]
        ]
        if l < prompt_length:
            world_prior_visits[i, snake_position[:, 0], snake_position[:, 1]] += 1
        sequences[:, 2 * l + 1] = snake_direction

        # nb x 2
        snake_position = snake_next_position[i, j]

    return sequences, sequences_prior_visits, worlds, world_prior_visits


# generate_snake_sequences(nb=1, height=4, width=6, nb_colors=3, length=20)
# exit(0)


def solver(input, ar_mask):
    for n in range(input.size(0)):
        i, j, memory = 0, 0, {}
        # print(input[n])
        # print(ar_mask[n])
        for l in range(input.size(1) // 2):
            if ar_mask[n, 2 * l] == 1:
                if memory.get((i, j)) is None:
                    input[n, 2 * l] = -1
                else:
                    input[n, 2 * l] = memory[(i, j)]
            else:
                # print(f'@3 {memory=}')
                if memory.get((i, j)) is None:
                    memory[(i, j)] = input[n, 2 * l]
                else:
                    assert memory[(i, j)] == input[n, 2 * l], f"n={n} l={l}"
            # print(f'@1 {i=} {j=}')
            d = input[n, 2 * l + 1].item()
            i += (d + 1) % 2 * (d - 1)
            j += d % 2 * (d - 2)
            # print(f'@2 {i=} {j=}')


def seq2str(seq):
    return "".join(["NESW123456789"[i] for i in seq])


######################################################################

if __name__ == "__main__":
    train_input, train_prior_visits, _, _ = generate_sequences(
        nb=20,
        height=9,
        width=12,
        nb_colors=5,
        length=50,
        prompt_length=100,
    )

    print([seq2str(s) for s in train_input])

######################################################################
