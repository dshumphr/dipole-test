#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

import math
import torch, torchvision
import torch.nn.functional as F

color_name2rgb = {
    "white": [255, 255, 255],
    "red": [255, 0, 0],
    "green": [0, 128, 0],
    "blue": [0, 0, 255],
    "yellow": [255, 255, 0],
    "black": [0, 0, 0],
    "maroon": [128, 0, 0],
    "dark_red": [139, 0, 0],
    "brown": [165, 42, 42],
    "firebrick": [178, 34, 34],
    "crimson": [220, 20, 60],
    "tomato": [255, 99, 71],
    "coral": [255, 127, 80],
    "indian_red": [205, 92, 92],
    "light_coral": [240, 128, 128],
    "dark_salmon": [233, 150, 122],
    "salmon": [250, 128, 114],
    "light_salmon": [255, 160, 122],
    "orange_red": [255, 69, 0],
    "dark_orange": [255, 140, 0],
    "orange": [255, 165, 0],
    "gold": [255, 215, 0],
    "dark_golden_rod": [184, 134, 11],
    "golden_rod": [218, 165, 32],
    "pale_golden_rod": [238, 232, 170],
    "dark_khaki": [189, 183, 107],
    "khaki": [240, 230, 140],
    "olive": [128, 128, 0],
    "yellow_green": [154, 205, 50],
    "dark_olive_green": [85, 107, 47],
    "olive_drab": [107, 142, 35],
    "lawn_green": [124, 252, 0],
    "chartreuse": [127, 255, 0],
    "green_yellow": [173, 255, 47],
    "dark_green": [0, 100, 0],
    "forest_green": [34, 139, 34],
    "lime": [0, 255, 0],
    "lime_green": [50, 205, 50],
    "light_green": [144, 238, 144],
    "pale_green": [152, 251, 152],
    "dark_sea_green": [143, 188, 143],
    "medium_spring_green": [0, 250, 154],
    "spring_green": [0, 255, 127],
    "sea_green": [46, 139, 87],
    "medium_aqua_marine": [102, 205, 170],
    "medium_sea_green": [60, 179, 113],
    "light_sea_green": [32, 178, 170],
    "dark_slate_gray": [47, 79, 79],
    "teal": [0, 128, 128],
    "dark_cyan": [0, 139, 139],
    "aqua": [0, 255, 255],
    "cyan": [0, 255, 255],
    "light_cyan": [224, 255, 255],
    "dark_turquoise": [0, 206, 209],
    "turquoise": [64, 224, 208],
    "medium_turquoise": [72, 209, 204],
    "pale_turquoise": [175, 238, 238],
    "aqua_marine": [127, 255, 212],
    "powder_blue": [176, 224, 230],
    "cadet_blue": [95, 158, 160],
    "steel_blue": [70, 130, 180],
    "corn_flower_blue": [100, 149, 237],
    "deep_sky_blue": [0, 191, 255],
    "dodger_blue": [30, 144, 255],
    "light_blue": [173, 216, 230],
    "sky_blue": [135, 206, 235],
    "light_sky_blue": [135, 206, 250],
    "midnight_blue": [25, 25, 112],
    "navy": [0, 0, 128],
    "dark_blue": [0, 0, 139],
    "medium_blue": [0, 0, 205],
    "royal_blue": [65, 105, 225],
    "blue_violet": [138, 43, 226],
    "indigo": [75, 0, 130],
    "dark_slate_blue": [72, 61, 139],
    "slate_blue": [106, 90, 205],
    "medium_slate_blue": [123, 104, 238],
    "medium_purple": [147, 112, 219],
    "dark_magenta": [139, 0, 139],
    "dark_violet": [148, 0, 211],
    "dark_orchid": [153, 50, 204],
    "medium_orchid": [186, 85, 211],
    "purple": [128, 0, 128],
    "thistle": [216, 191, 216],
    "plum": [221, 160, 221],
    "violet": [238, 130, 238],
    "magenta": [255, 0, 255],
    "orchid": [218, 112, 214],
    "medium_violet_red": [199, 21, 133],
    "pale_violet_red": [219, 112, 147],
    "deep_pink": [255, 20, 147],
    "hot_pink": [255, 105, 180],
    "light_pink": [255, 182, 193],
    "pink": [255, 192, 203],
    "antique_white": [250, 235, 215],
    "beige": [245, 245, 220],
    "bisque": [255, 228, 196],
    "blanched_almond": [255, 235, 205],
    "wheat": [245, 222, 179],
    "corn_silk": [255, 248, 220],
    "lemon_chiffon": [255, 250, 205],
    "light_golden_rod_yellow": [250, 250, 210],
    "light_yellow": [255, 255, 224],
    "saddle_brown": [139, 69, 19],
    "sienna": [160, 82, 45],
    "chocolate": [210, 105, 30],
    "peru": [205, 133, 63],
    "sandy_brown": [244, 164, 96],
    "burly_wood": [222, 184, 135],
    "tan": [210, 180, 140],
    "rosy_brown": [188, 143, 143],
    "moccasin": [255, 228, 181],
    "navajo_white": [255, 222, 173],
    "peach_puff": [255, 218, 185],
    "misty_rose": [255, 228, 225],
    "lavender_blush": [255, 240, 245],
    "linen": [250, 240, 230],
    "old_lace": [253, 245, 230],
    "papaya_whip": [255, 239, 213],
    "sea_shell": [255, 245, 238],
    "mint_cream": [245, 255, 250],
    "slate_gray": [112, 128, 144],
    "light_slate_gray": [119, 136, 153],
    "light_steel_blue": [176, 196, 222],
    "lavender": [230, 230, 250],
    "floral_white": [255, 250, 240],
    "alice_blue": [240, 248, 255],
    "ghost_white": [248, 248, 255],
    "honeydew": [240, 255, 240],
    "ivory": [255, 255, 240],
    "azure": [240, 255, 255],
    "snow": [255, 250, 250],
    "silver": [192, 192, 192],
    "gainsboro": [220, 220, 220],
    "white_smoke": [245, 245, 245],
}

color_name2id = dict([(n, k) for k, n in enumerate(color_name2rgb.keys())])
color_id2name = dict([(k, n) for k, n in enumerate(color_name2rgb.keys())])

######################################################################


def all_properties(height, width, nb_squares, square_i, square_j, square_c):
    s = []

    for r, c_r in [(k, color_id2name[square_c[k].item()]) for k in range(nb_squares)]:
        s += [f"there is {c_r}"]

        if square_i[r] >= height - height // 3:
            s += [f"{c_r} bottom"]
        if square_i[r] < height // 3:
            s += [f"{c_r} top"]
        if square_j[r] >= width - width // 3:
            s += [f"{c_r} right"]
        if square_j[r] < width // 3:
            s += [f"{c_r} left"]

        for t, c_t in [
            (k, color_id2name[square_c[k].item()]) for k in range(nb_squares)
        ]:
            if square_i[r] > square_i[t]:
                s += [f"{c_r} below {c_t}"]
            if square_i[r] < square_i[t]:
                s += [f"{c_r} above {c_t}"]
            if square_j[r] > square_j[t]:
                s += [f"{c_r} right of {c_t}"]
            if square_j[r] < square_j[t]:
                s += [f"{c_r} left of {c_t}"]

    return s


######################################################################

# Generates sequences


def generate(
    nb,
    height,
    width,
    max_nb_squares=5,
    max_nb_properties=10,
    nb_colors=5,
    pruner=None,
):
    assert nb_colors >= max_nb_squares and nb_colors <= len(color_name2rgb) - 1

    descr = []

    for n in range(nb):
        # we want uniform over the combinations of 1 to max_nb_squares
        # pixels of nb_colors
        logits = math.log(nb_colors) * torch.arange(1, max_nb_squares + 1).float()
        dist = torch.distributions.categorical.Categorical(logits=logits)
        nb_squares = dist.sample((1,)) + 1
        # nb_squares = torch.randint(max_nb_squares, (1,)) + 1
        square_position = torch.randperm(height * width)[:nb_squares]

        # color 0 is white and reserved for the background
        square_c = torch.randperm(nb_colors)[:nb_squares] + 1
        square_i = square_position.div(width, rounding_mode="floor")
        square_j = square_position % width

        img = torch.zeros(height * width, dtype=torch.int64)
        for k in range(nb_squares):
            img[square_position[k]] = square_c[k]

        # generates all the true properties

        s = all_properties(height, width, nb_squares, square_i, square_j, square_c)

        if pruner is not None:
            s = list(filter(pruner, s))

        # picks at most max_nb_properties at random

        nb_properties = torch.randint(max_nb_properties, (1,)) + 1
        s = (
            " <sep> ".join([s[k] for k in torch.randperm(len(s))[:nb_properties]])
            + " <img> "
            + " ".join([f"{color_id2name[n.item()]}" for n in img])
        )

        descr += [s]

    return descr


######################################################################

# Extracts the image after <img> in descr as a 1x3xHxW tensor


def descr2img(descr, height, width):
    result = []

    def token2color(t):
        try:
            return color_name2rgb[t]
        except KeyError:
            return [128, 128, 128]

    for d in descr:
        d = d.split("<img>")[1]
        d = d.strip().split(" ")[: height * width]
        d = d + ["<unk>"] * (height * width - len(d))
        d = [token2color(t) for t in d]
        img = torch.tensor(d).permute(1, 0).reshape(1, 3, height, width)
        result.append(img)

    return torch.cat(result, 0)


######################################################################

# Returns all the properties of the image after <img> in descr


def descr2properties(descr, height, width):
    if type(descr) == list:
        return [descr2properties(d, height, width) for d in descr]

    d = descr.split("<img>")
    img_tokens = d[-1] if len(d) > 1 else ""
    img_tokens = img_tokens.strip().split(" ")[: height * width]
    if len(img_tokens) != height * width:
        return []

    seen = {}
    for k, x in enumerate(img_tokens):
        if x != color_id2name[0]:
            if x in color_name2rgb:
                if x in seen:
                    return []
            else:
                return []
            seen[x] = (color_name2id[x], k // width, k % width)

    square_infos = tuple(zip(*seen.values()))

    if square_infos:
        square_c = torch.tensor(square_infos[0])
        square_i = torch.tensor(square_infos[1])
        square_j = torch.tensor(square_infos[2])
    else:
        square_c = torch.tensor([])
        square_i = torch.tensor([])
        square_j = torch.tensor([])

    s = all_properties(height, width, len(seen), square_i, square_j, square_c)

    return s


######################################################################

# Returns a triplet composed of (1) the total number of properties
# before <img> in descr, (2) the total number of properties the image
# after <img> verifies, and (3) the number of properties in (1) not in
# (2)


def nb_properties(descr, height, width, pruner=None):
    if type(descr) == list:
        return [nb_properties(d, height, width, pruner) for d in descr]

    d = descr.split("<img>", 1)
    if len(d) == 0:
        return 0
    d = d[0].strip().split("<sep>")
    d = [x.strip() for x in d]

    all_properties = set(descr2properties(descr, height, width))

    if pruner is None:
        requested_properties = set(d)
    else:
        requested_properties = set(filter(pruner, d))

    missing_properties = requested_properties - all_properties

    return (len(requested_properties), len(all_properties), len(missing_properties))


######################################################################

if __name__ == "__main__":
    for n in range(16):
        descr = generate(nb=1, height=12, width=16)

        print(nb_properties(descr, height=12, width=16))

        with open(f"picoclvr_example_{n:02d}.txt", "w") as f:
            for d in descr:
                f.write(f"{d}\n\n")

        img = descr2img(descr, height=12, width=16)
        if img.size(0) == 1:
            img = F.pad(img, (1, 1, 1, 1), value=64)

        torchvision.utils.save_image(
            img / 255.0,
            f"picoclvr_example_{n:02d}.png",
            padding=1,
            nrow=4,
            pad_value=0.8,
        )

    import time

    start_time = time.perf_counter()
    descr = generate(nb=1000, height=12, width=16)
    end_time = time.perf_counter()
    print(f"{len(descr) / (end_time - start_time):.02f} samples per second")

######################################################################
