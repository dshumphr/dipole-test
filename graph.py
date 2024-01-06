#!/usr/bin/env python

import math

import torch, torchvision

from torch import nn
from torch.nn import functional as F

import cairo


######################################################################


def save_attention_image(
    # image to save
    filename,
    tokens_input,
    tokens_output,
    # list of 2d tensors T2xT1, T3xT2, ..., TkxTk-1
    attention_matrices,
    # do not draw links with a lesser attention
    min_link_attention=0,
    # draw only the strongest links necessary so that their summed
    # attention is above min_total_attention
    min_total_attention=None,
    # draw only the top k links
    k_top=None,
    # the purely graphical settings
    curved=True,
    pixel_scale=8,
    token_gap=15,
    layer_gap=25,
    y_eps=0.5,
    padding=10,
):
    if k_top is not None:
        am = []
        for m in attention_matrices:
            am.append(m * (m.sort(dim=-1, descending=True).indices < k_top))
        attention_matrices = am

    if min_total_attention is not None:
        am = []
        for m in attention_matrices:
            s = m.sort(dim=-1)
            m = 1 - (s.values.cumsum(-1) < 1 - min_total_attention).long()
            b = m.new(m.size()).scatter_(dim=-1, index=s.indices, src=m)
            am.append(m * b)

    surface = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None)

    ctx = cairo.Context(surface)
    ctx.scale(pixel_scale, pixel_scale)

    ctx.set_source_rgb(0.0, 0.0, 0.0)
    ctx.set_font_size(4.0)
    # ctx.select_font_face("Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)

    x, y = 0, 0

    ctx.set_line_width(0.25)
    for d in range(len(attention_matrices)):
        at = attention_matrices[d].to("cpu")
        ni = torch.arange(at.size(0))[:, None].expand_as(at)
        nj = torch.arange(at.size(1))[None, :].expand_as(at)
        at = at.flatten()
        o = at.sort().indices
        at = at[o]
        ni = ni.flatten()[o]
        nj = nj.flatten()[o]
        for i, j, a in zip(ni, nj, at):
            if a > 0 and a >= min_link_attention:
                c = 1 - a.item()
                ctx.set_source_rgb(c, c, c)
                ax, ay = j * token_gap, y - y_eps
                ctx.move_to(ax, ay)
                dx, dy = i * token_gap, y - layer_gap + y_eps
                if curved:
                    bx, by = ax, ay - layer_gap * 0.5
                    cx, cy = dx, dy + layer_gap * 0.5
                    ctx.curve_to(bx, by, cx, cy, dx, dy)
                else:
                    ctx.line_to(dx, dy)
                ctx.stroke()
        y -= layer_gap

    for d in range(0, len(attention_matrices) + 1):
        n = (
            attention_matrices[0].size(-1)
            if d == 0
            else attention_matrices[d - 1].size(-2)
        )
        for n in range(n):
            xc, yc = n * token_gap, -d * layer_gap
            ctx.set_source_rgb(1.0, 1.0, 1.0)
            ctx.arc(xc, yc, token_gap / 10, 0, 2 * math.pi)
            ctx.fill()
            ctx.set_source_rgb(0.0, 0.0, 0.0)
            ctx.arc(xc, yc, token_gap / 20, 0, 2 * math.pi)
            ctx.fill()

    ctx.set_source_rgb(0.0, 0.0, 0.0)

    for k, t in enumerate(tokens_input):
        s = str(t)
        (
            x_bearing,
            y_bearing,
            width_t,
            height_t,
            x_advance,
            y_advance,
        ) = ctx.text_extents(s)
        ctx.move_to(k * token_gap - width_t / 2, 2 * token_gap / 5)
        ctx.show_text(s)

    for k, t in enumerate(tokens_output):
        s = str(t)
        (
            x_bearing,
            y_bearing,
            width_t,
            height_t,
            x_advance,
            y_advance,
        ) = ctx.text_extents(s)
        ctx.move_to(
            k * token_gap - width_t / 2,
            -token_gap / 5 - len(attention_matrices) * layer_gap,
        )
        ctx.show_text(s)

    x, y, width, height = surface.ink_extents()
    x -= padding
    y -= padding
    width += 2 * padding
    height += 2 * padding
    pdf_surface = cairo.PDFSurface(filename, width, height)
    ctx_pdf = cairo.Context(pdf_surface)
    ctx_pdf.set_source_surface(surface, -x, -y)
    ctx_pdf.paint()
    pdf_surface.finish()


######################################################################

if __name__ == "__main__":
    import mygpt

    tokens_output = ["<wat>", "-", 3, 4, "<end>"]
    tokens_input = [""] + tokens_output[:-1]

    vocabulary_size = 3
    x = torch.randint(vocabulary_size, (1, len(tokens_input)))

    model = mygpt.MyGPT(
        vocabulary_size=vocabulary_size,
        dim_model=4,
        dim_keys=2,
        dim_hidden=2,
        nb_heads=2,
        nb_blocks=5,
        dropout=0.1,
        causal=True,
    )

    model.eval()
    model.record_attention()

    y1 = model(mygpt.BracketedSequence(x)).x

    attention_matrices = [m[0, 0] for m in model.retrieve_attention()]

    # attention_matrices = [torch.rand(*s) for s in [ (4,5),(3,4),(8,3),(5,8) ]]

    save_attention_image(
        "attention.pdf",
        tokens_input,
        tokens_output,
        attention_matrices,
        # k_top=2,
        min_total_attention=0.9,
    )
