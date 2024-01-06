#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

import torch, torchvision

######################################################################

v_empty, v_wall, v_start, v_goal, v_path = 0, 1, 2, 3, 4


def create_maze(h=11, w=17, nb_walls=8):
    assert h % 2 == 1 and w % 2 == 1

    a, k = 0, 0

    while k < nb_walls:
        while True:
            if a == 0:
                m = torch.zeros(h, w, dtype=torch.int64)
                m[0, :] = 1
                m[-1, :] = 1
                m[:, 0] = 1
                m[:, -1] = 1

            r = torch.rand(4)

            if r[0] <= 0.5:
                i1, i2, j = (
                    int((r[1] * h).item()),
                    int((r[2] * h).item()),
                    int((r[3] * w).item()),
                )
                i1, i2, j = i1 - i1 % 2, i2 - i2 % 2, j - j % 2
                i1, i2 = min(i1, i2), max(i1, i2)
                if i2 - i1 > 1 and i2 - i1 <= h / 2 and m[i1 : i2 + 1, j].sum() <= 1:
                    m[i1 : i2 + 1, j] = 1
                    break
            else:
                i, j1, j2 = (
                    int((r[1] * h).item()),
                    int((r[2] * w).item()),
                    int((r[3] * w).item()),
                )
                i, j1, j2 = i - i % 2, j1 - j1 % 2, j2 - j2 % 2
                j1, j2 = min(j1, j2), max(j1, j2)
                if j2 - j1 > 1 and j2 - j1 <= w / 2 and m[i, j1 : j2 + 1].sum() <= 1:
                    m[i, j1 : j2 + 1] = 1
                    break
            a += 1

            if a > 10 * nb_walls:
                a, k = 0, 0

        k += 1

    return m


######################################################################


def compute_distance(walls, goal_i, goal_j):
    max_length = walls.numel()
    dist = torch.full_like(walls, max_length)

    dist[goal_i, goal_j] = 0
    pred_dist = torch.empty_like(dist)

    while True:
        pred_dist.copy_(dist)
        d = (
            torch.cat(
                (
                    dist[None, 1:-1, 0:-2],
                    dist[None, 2:, 1:-1],
                    dist[None, 1:-1, 2:],
                    dist[None, 0:-2, 1:-1],
                ),
                0,
            ).min(dim=0)[0]
            + 1
        )

        dist[1:-1, 1:-1] = torch.min(dist[1:-1, 1:-1], d)
        dist = walls * max_length + (1 - walls) * dist

        if dist.equal(pred_dist):
            return dist * (1 - walls)


######################################################################


def compute_policy(walls, goal_i, goal_j):
    distance = compute_distance(walls, goal_i, goal_j)
    distance = distance + walls.numel() * walls

    value = distance.new_full((4,) + distance.size(), walls.numel())
    value[0, :, 1:] = distance[:, :-1]  # <
    value[1, :, :-1] = distance[:, 1:]  # >
    value[2, 1:, :] = distance[:-1, :]  # ^
    value[3, :-1, :] = distance[1:, :]  # v

    proba = (value.min(dim=0)[0][None] == value).float()
    proba = proba / proba.sum(dim=0)[None]
    proba = proba * (1 - walls) + walls.float() / 4

    return proba


def stationary_densities(mazes, policies):
    policies = policies * (mazes != v_goal)[:, None]
    start = (mazes == v_start).nonzero(as_tuple=True)
    probas = mazes.new_zeros(mazes.size(), dtype=torch.float32)
    pred_probas = probas.clone()
    probas[start] = 1.0

    while not pred_probas.equal(probas):
        pred_probas.copy_(probas)
        probas.zero_()
        probas[:, 1:, :] += pred_probas[:, :-1, :] * policies[:, 3, :-1, :]
        probas[:, :-1, :] += pred_probas[:, 1:, :] * policies[:, 2, 1:, :]
        probas[:, :, 1:] += pred_probas[:, :, :-1] * policies[:, 1, :, :-1]
        probas[:, :, :-1] += pred_probas[:, :, 1:] * policies[:, 0, :, 1:]
        probas[start] = 1.0

    return probas


######################################################################


def mark_path(walls, i, j, goal_i, goal_j, policy):
    action = torch.distributions.categorical.Categorical(
        policy.permute(1, 2, 0)
    ).sample()
    n, nmax = 0, walls.numel()
    while i != goal_i or j != goal_j:
        di, dj = [(0, -1), (0, 1), (-1, 0), (1, 0)][action[i, j]]
        i, j = i + di, j + dj
        assert walls[i, j] == 0
        walls[i, j] = v_path
        n += 1
        assert n < nmax


def path_optimality(ref_paths, paths):
    return (ref_paths == v_path).long().flatten(1).sum(1) == (
        paths == v_path
    ).long().flatten(1).sum(1)


def path_correctness(mazes, paths):
    still_ok = (mazes - (paths * (paths != v_path))).view(mazes.size(0), -1).abs().sum(
        1
    ) == 0
    reached = still_ok.new_zeros(still_ok.size())
    current, pred_current = paths.clone(), paths.new_zeros(paths.size())
    goal = (mazes == v_goal).long()
    while not pred_current.equal(current):
        pred_current.copy_(current)
        u = (current == v_start).long()
        possible_next = (
            u[:, 2:, 1:-1] + u[:, 0:-2, 1:-1] + u[:, 1:-1, 2:] + u[:, 1:-1, 0:-2] > 0
        ).long()
        u = u[:, 1:-1, 1:-1]
        reached += ((goal[:, 1:-1, 1:-1] * possible_next).sum((1, 2)) == 1) * (
            (current == v_path).sum((1, 2)) == 0
        )
        current[:, 1:-1, 1:-1] = (1 - u) * current[:, 1:-1, 1:-1] + (
            v_start - v_path
        ) * (possible_next * (current[:, 1:-1, 1:-1] == v_path))
        still_ok *= (current == v_start).sum((1, 2)) <= 1

    return still_ok * reached


######################################################################


def create_maze_data(
    nb, height=11, width=17, nb_walls=8, dist_min=10, progress_bar=lambda x: x
):
    mazes = torch.empty(nb, height, width, dtype=torch.int64)
    paths = torch.empty(nb, height, width, dtype=torch.int64)
    policies = torch.empty(nb, 4, height, width)

    for n in progress_bar(range(nb)):
        maze = create_maze(height, width, nb_walls)
        i = (maze == v_empty).nonzero()
        while True:
            start, goal = i[torch.randperm(i.size(0))[:2]]
            if (start - goal).abs().sum() >= dist_min:
                break
        start_i, start_j, goal_i, goal_j = start[0], start[1], goal[0], goal[1]

        policy = compute_policy(maze, goal_i, goal_j)
        path = maze.clone()
        mark_path(path, start_i, start_j, goal_i, goal_j, policy)
        maze[start_i, start_j] = v_start
        maze[goal_i, goal_j] = v_goal
        path[start_i, start_j] = v_start
        path[goal_i, goal_j] = v_goal

        mazes[n] = maze
        paths[n] = path
        policies[n] = policy

    return mazes, paths, policies


######################################################################


def save_image(
    name,
    mazes,
    target_paths=None,
    predicted_paths=None,
    path_correct=None,
    path_optimal=None,
):
    colors = torch.tensor(
        [
            [255, 255, 255],  # empty
            [0, 0, 0],  # wall
            [0, 255, 0],  # start
            [127, 127, 255],  # goal
            [255, 0, 0],  # path
        ]
    )

    mazes = mazes.cpu()

    c_mazes = (
        colors[mazes.reshape(-1)].reshape(mazes.size() + (-1,)).permute(0, 3, 1, 2)
    )

    imgs = c_mazes.unsqueeze(1)

    if target_paths is not None:
        target_paths = target_paths.cpu()

        c_target_paths = (
            colors[target_paths.reshape(-1)]
            .reshape(target_paths.size() + (-1,))
            .permute(0, 3, 1, 2)
        )

        imgs = torch.cat((imgs, c_target_paths.unsqueeze(1)), 1)

    if predicted_paths is not None:
        predicted_paths = predicted_paths.cpu()
        c_predicted_paths = (
            colors[predicted_paths.reshape(-1)]
            .reshape(predicted_paths.size() + (-1,))
            .permute(0, 3, 1, 2)
        )
        imgs = torch.cat((imgs, c_predicted_paths.unsqueeze(1)), 1)

    img = torch.tensor([255, 255, 0]).view(1, -1, 1, 1)

    # NxKxCxHxW
    if path_optimal is not None:
        path_optimal = path_optimal.cpu().long().view(-1, 1, 1, 1)
        img = (
            img * (1 - path_optimal)
            + torch.tensor([0, 255, 0]).view(1, -1, 1, 1) * path_optimal
        )

    if path_correct is not None:
        path_correct = path_correct.cpu().long().view(-1, 1, 1, 1)
        img = img * path_correct + torch.tensor([255, 0, 0]).view(1, -1, 1, 1) * (
            1 - path_correct
        )

    img = img.expand(
        -1, -1, imgs.size(3) + 2, 1 + imgs.size(1) * (1 + imgs.size(4))
    ).clone()

    print(f"{img.size()=} {imgs.size()=}")

    for k in range(imgs.size(1)):
        img[
            :,
            :,
            1 : 1 + imgs.size(3),
            1 + k * (1 + imgs.size(4)) : 1 + k * (1 + imgs.size(4)) + imgs.size(4),
        ] = imgs[:, k]

    img = img.float() / 255.0

    torchvision.utils.save_image(img, name, nrow=4, padding=1, pad_value=224.0 / 256)


######################################################################

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mazes, paths, policies = create_maze_data(8)
    mazes, paths = mazes.to(device), paths.to(device)
    save_image("test.png", mazes=mazes, target_paths=paths, predicted_paths=paths)
    print(path_correctness(mazes, paths))

######################################################################
