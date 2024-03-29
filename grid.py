#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

import math
import torch, torchvision
import torch.nn.functional as F

######################################################################


class GridFactory:
    def __init__(
        self,
        size=6,
        max_nb_items=4,
        max_nb_transformations=3,
        nb_questions=4,
        nb_shapes=6,
        nb_colors=6,
    ):
        assert size % 2 == 0
        self.size = size
        self.max_nb_items = max_nb_items
        self.max_nb_transformations = max_nb_transformations
        self.nb_questions = nb_questions
        self.name_shapes = [chr(ord("A") + k) for k in range(nb_shapes)]
        self.name_colors = [
            "red",
            "yellow",
            "blue",
            "green",
            "white",
            "black",
            "maroon",
            "dark_red",
            "brown",
            "firebrick",
            "crimson",
            "tomato",
            "coral",
            "indian_red",
            "light_coral",
            "dark_salmon",
            "salmon",
            "light_salmon",
            "orange_red",
            "dark_orange",
            "orange",
            "gold",
            "dark_golden_rod",
            "golden_rod",
            "pale_golden_rod",
            "dark_khaki",
            "khaki",
            "olive",
            "yellow_green",
            "dark_olive_green",
            "olive_drab",
            "lawn_green",
            "chartreuse",
            "green_yellow",
            "dark_green",
            "forest_green",
            "lime",
            "lime_green",
            "light_green",
            "pale_green",
            "dark_sea_green",
            "medium_spring_green",
            "spring_green",
            "sea_green",
            "medium_aqua_marine",
            "medium_sea_green",
            "light_sea_green",
            "dark_slate_gray",
            "teal",
            "dark_cyan",
            "aqua",
            "cyan",
            "light_cyan",
            "dark_turquoise",
            "turquoise",
            "medium_turquoise",
            "pale_turquoise",
            "aqua_marine",
            "powder_blue",
            "cadet_blue",
            "steel_blue",
            "corn_flower_blue",
            "deep_sky_blue",
            "dodger_blue",
            "light_blue",
            "sky_blue",
            "light_sky_blue",
            "midnight_blue",
            "navy",
            "dark_blue",
            "medium_blue",
            "royal_blue",
            "blue_violet",
            "indigo",
            "dark_slate_blue",
            "slate_blue",
            "medium_slate_blue",
            "medium_purple",
            "dark_magenta",
            "dark_violet",
            "dark_orchid",
            "medium_orchid",
            "purple",
            "thistle",
            "plum",
            "violet",
            "magenta",
            "orchid",
            "medium_violet_red",
            "pale_violet_red",
            "deep_pink",
            "hot_pink",
            "light_pink",
            "pink",
            "antique_white",
            "beige",
            "bisque",
            "blanched_almond",
            "wheat",
            "corn_silk",
            "lemon_chiffon",
            "light_golden_rod_yellow",
            "light_yellow",
            "saddle_brown",
            "sienna",
            "chocolate",
            "peru",
            "sandy_brown",
            "burly_wood",
            "tan",
            "rosy_brown",
            "moccasin",
            "navajo_white",
            "peach_puff",
            "misty_rose",
            "lavender_blush",
            "linen",
            "old_lace",
            "papaya_whip",
            "sea_shell",
            "mint_cream",
            "slate_gray",
            "light_slate_gray",
            "light_steel_blue",
            "lavender",
            "floral_white",
            "alice_blue",
            "ghost_white",
            "honeydew",
            "ivory",
            "azure",
            "snow",
            "silver",
            "gainsboro",
            "white_smoke",
        ][:nb_colors]

    def generate_scene(self):
        nb_items = torch.randint(self.max_nb_items - 1, (1,)).item() + 2
        col = torch.full((self.size * self.size,), -1)
        shp = torch.full((self.size * self.size,), -1)
        a = torch.randperm(len(self.name_colors) * len(self.name_shapes))[:nb_items]
        col[:nb_items] = a % len(self.name_colors)
        shp[:nb_items] = a // len(self.name_colors)
        i = torch.randperm(self.size * self.size)
        col = col[i]
        shp = shp[i]
        return col.reshape(self.size, self.size), shp.reshape(self.size, self.size)

    def random_transformations(self, scene):
        col, shp = scene

        descriptions = []
        nb_transformations = torch.randint(self.max_nb_transformations + 1, (1,)).item()
        transformations = torch.randint(5, (nb_transformations,))

        for t in transformations:
            if t == 0:
                col, shp = col.flip(0), shp.flip(0)
                descriptions += ["<chg> vertical flip"]
            elif t == 1:
                col, shp = col.flip(1), shp.flip(1)
                descriptions += ["<chg> horizontal flip"]
            elif t == 2:
                col, shp = col.flip(0).t(), shp.flip(0).t()
                descriptions += ["<chg> rotate 90 degrees"]
            elif t == 3:
                col, shp = col.flip(0).flip(1), shp.flip(0).flip(1)
                descriptions += ["<chg> rotate 180 degrees"]
            elif t == 4:
                col, shp = col.flip(1).t(), shp.flip(1).t()
                descriptions += ["<chg> rotate 270 degrees"]

            col, shp = col.contiguous(), shp.contiguous()

        return (col, shp), descriptions

    def print_scene(self, scene):
        col, shp = scene

        # for i in range(self.size):
        # for j in range(self.size):
        # if col[i,j] >= 0:
        # print(f"at ({i},{j}) {self.name_colors[col[i,j]]} {self.name_shapes[shp[i,j]]}")

        for i in range(self.size):
            for j in range(self.size):
                if col[i, j] >= 0:
                    print(
                        f"{self.name_colors[col[i,j]][0]}{self.name_shapes[shp[i,j]]}",
                        end="",
                    )
                elif j == 0:
                    print(" +", end="")
                else:
                    print("-+", end="")
                if j < self.size - 1:
                    print("--", end="")
                else:
                    print("")
            if i < self.size - 1:
                for j in range(self.size - 1):
                    print(" |  ", end="")
                print(" |")

    def grid_positions(self, scene):
        col, shp = scene

        properties = []

        for i in range(self.size):
            for j in range(self.size):
                if col[i, j] >= 0:
                    n = f"{self.name_colors[col[i,j]]} {self.name_shapes[shp[i,j]]}"
                    properties += [f"a {n} at {i} {j}"]

        return properties

    def all_properties(self, scene):
        col, shp = scene

        properties = []

        for i1 in range(self.size):
            for j1 in range(self.size):
                if col[i1, j1] >= 0:
                    n1 = (
                        f"{self.name_colors[col[i1,j1]]} {self.name_shapes[shp[i1,j1]]}"
                    )
                    properties += [f"there is a {n1}"]
                    if i1 < self.size // 2:
                        properties += [f"a {n1} is in the top half"]
                    if i1 >= self.size // 2:
                        properties += [f"a {n1} is in the bottom half"]
                    if j1 < self.size // 2:
                        properties += [f"a {n1} is in the left half"]
                    if j1 >= self.size // 2:
                        properties += [f"a {n1} is in the right half"]
                    for i2 in range(self.size):
                        for j2 in range(self.size):
                            if col[i2, j2] >= 0:
                                n2 = f"{self.name_colors[col[i2,j2]]} {self.name_shapes[shp[i2,j2]]}"
                                if i1 > i2:
                                    properties += [f"a {n1} is below a {n2}"]
                                if i1 < i2:
                                    properties += [f"a {n1} is above a {n2}"]
                                if j1 > j2:
                                    properties += [f"a {n1} is right of a {n2}"]
                                if j1 < j2:
                                    properties += [f"a {n1} is left of a {n2}"]
                                if abs(i1 - i2) + abs(j1 - j2) == 1:
                                    properties += [f"a {n1} is next to a {n2}"]

        return properties

    def generate_scene_and_questions(self):
        while True:
            while True:
                start_scene = self.generate_scene()
                scene, transformations = self.random_transformations(start_scene)
                true = self.all_properties(scene)
                if len(true) >= self.nb_questions:
                    break

            for a in range(10):
                col, shp = scene
                col, shp = col.view(-1), shp.view(-1)
                p = torch.randperm(col.size(0))
                col, shp = col[p], shp[p]
                other_scene = (
                    col.view(self.size, self.size),
                    shp.view(self.size, self.size),
                )

                false = self.all_properties(other_scene)

                # We sometime add properties from a totally different
                # scene to have negative "there is a xxx xxx"
                # properties
                if torch.rand(1).item() < 0.2:
                    other_scene = self.generate_scene()
                    false += self.all_properties(other_scene)

                false = list(set(false) - set(true))
                if len(false) >= self.nb_questions:
                    break

            if a < 10:
                break

        true = [true[k] for k in torch.randperm(len(true))[: self.nb_questions]]
        false = [false[k] for k in torch.randperm(len(false))[: self.nb_questions]]
        true = ["<prop> " + q + " <ans> true" for q in true]
        false = ["<prop> " + q + " <ans> false" for q in false]

        union = true + false
        questions = [union[k] for k in torch.randperm(len(union))[: self.nb_questions]]

        result = " ".join(
            ["<obj> " + x for x in self.grid_positions(start_scene)]
            + transformations
            + questions
        )

        return start_scene, scene, result

    def generate_samples(self, nb, progress_bar=None):
        result = []

        r = range(nb)
        if progress_bar is not None:
            r = progress_bar(r)

        for _ in r:
            result.append(self.generate_scene_and_questions()[2])

        return result


######################################################################

if __name__ == "__main__":
    import time

    grid_factory = GridFactory()

    # start_time = time.perf_counter()
    # samples = grid_factory.generate_samples(10000)
    # end_time = time.perf_counter()
    # print(f"{len(samples) / (end_time - start_time):.02f} samples per second")

    start_scene, scene, questions = grid_factory.generate_scene_and_questions()
    print()
    print("-- Original scene -----------------------------")
    print()
    grid_factory.print_scene(start_scene)
    print()
    print("-- Transformed scene --------------------------")
    print()
    grid_factory.print_scene(scene)
    print()
    print("-- Sequence -----------------------------------")
    print()
    print(questions)

######################################################################
