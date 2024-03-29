#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

import math, sys, argparse, time, tqdm, os, datetime, warnings

import torch, torchvision
from torch import nn
from torch.nn import functional as F

# torch.autograd.set_detect_anomaly(True) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import ffutils
import mygpt, tasks, problems

######################################################################


def str2bool(x):
    x = x.lower()
    if x in {"1", "true", "yes"}:
        return True
    elif x in {"0", "false", "no"}:
        return False
    else:
        raise ValueError


parser = argparse.ArgumentParser(
    description="An implementation of GPT with cache.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--task",
    type=str,
    default="twotargets",
    help="byheart, learnop, guessop, mixing, memory, twotargets, addition, picoclvr, mnist, maze, snake, stack, expr, rpl, grid, qmlp",
)

parser.add_argument("--log_filename", type=str, default="train.log", help=" ")

parser.add_argument("--result_dir", type=str, default=None)

parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--max_percents_of_test_in_train", type=int, default=1)

parser.add_argument("--force_cpu", type=str2bool, default=False)

########################################

parser.add_argument("--nb_epochs", type=int, default=25)

parser.add_argument("--physical_batch_size", type=int, default=None)

parser.add_argument("--batch_size", type=int, default=25)

parser.add_argument("--nb_train_samples", type=int, default=None)

parser.add_argument("--nb_test_samples", type=int, default=None)

parser.add_argument("--optim", type=str, default="adam")

########################################

parser.add_argument("--nb_warmup_iter", type=int, default=100)

parser.add_argument("--nb_decay_iter", type=int, default=5000)

parser.add_argument("--learning_rate", type=float, default=6e-4)

parser.add_argument("--min_learning_rate", type=float, default=6e-5)

# legacy

parser.add_argument("--legacy_lr_schedule", type=str2bool, default=True)

parser.add_argument("--legacy_large_lr", type=float, default=1e-4)

parser.add_argument("--legacy_small_lr", type=float, default=2e-5)

parser.add_argument("--legacy_nb_epoch_large_lr", type=float, default=10)

########################################

parser.add_argument("--model", type=str, default=None)

parser.add_argument("--attention", type=str, default=None)

parser.add_argument("--memex_proba", type=float, default=0)

parser.add_argument("--memex_nb_epochs", type=float, default=None)

parser.add_argument("--dim_model", type=int, default=None)

parser.add_argument("--dim_keys", type=int, default=None)

parser.add_argument("--dim_hidden", type=int, default=None)

parser.add_argument("--nb_heads", type=int, default=None)

parser.add_argument("--nb_lines", type=int, default=None)

parser.add_argument("--caterpillar_height", type=int, default=None)

parser.add_argument("--gate_dropout_proba", type=float, default=0.0)

parser.add_argument("--gate_dropout_sync", type=str2bool, default=False)

parser.add_argument("--gate_dropout_replace", type=str2bool, default=False)

parser.add_argument("--rho_inner_loss", type=float, default=0.0)

parser.add_argument("--nb_blocks", type=int, default=None)

parser.add_argument("--dropout", type=float, default=0.1)

########################################

parser.add_argument("--deterministic_synthesis", action="store_true", default=False)

parser.add_argument("--no_checkpoint", action="store_true", default=False)

parser.add_argument("--continue_training", action="store_true", default=False)

parser.add_argument("--checkpoint_name", type=str, default="checkpoint.pth")

##############################
# rpl options

parser.add_argument("--rpl_nb_starting_values", type=int, default=3)

parser.add_argument("--rpl_max_input", type=int, default=9)

parser.add_argument("--rpl_prog_len", type=int, default=8)

parser.add_argument("--rpl_nb_runs", type=int, default=5)

parser.add_argument("--rpl_no_prog", action="store_true", default=False)

##############################
# grid options

parser.add_argument("--grid_size", type=int, default=6)

parser.add_argument("--grid_nb_colors", type=int, default=6)

parser.add_argument("--grid_nb_shapes", type=int, default=6)

##############################
# picoclvr options

parser.add_argument("--picoclvr_nb_colors", type=int, default=5)

parser.add_argument("--picoclvr_height", type=int, default=12)

parser.add_argument("--picoclvr_width", type=int, default=16)

parser.add_argument("--picocvlr_prune_properties", type=str, default="none")

##############################
# Maze options

parser.add_argument("--maze_height", type=int, default=13)

parser.add_argument("--maze_width", type=int, default=21)

parser.add_argument("--maze_nb_walls", type=int, default=15)

##############################
# Snake options

parser.add_argument("--snake_height", type=int, default=9)

parser.add_argument("--snake_width", type=int, default=12)

parser.add_argument("--snake_nb_colors", type=int, default=5)

parser.add_argument("--snake_length", type=int, default=200)

##############################
# Stack options

parser.add_argument("--stack_nb_steps", type=int, default=100)

parser.add_argument("--stack_nb_stacks", type=int, default=3)

parser.add_argument("--stack_nb_digits", type=int, default=3)

parser.add_argument("--stack_fraction_values_for_train", type=float, default=0.75)

##############################
# Expr options

parser.add_argument("--expr_nb_variables", type=int, default=5)

parser.add_argument("--expr_sequence_length", type=int, default=40)

parser.add_argument("--expr_operand_max", type=int, default=9)

parser.add_argument("--expr_result_max", type=int, default=99)

parser.add_argument("--expr_input_file", type=str, default=None)

##############################
# Memory

parser.add_argument("--memory_len_total", type=int, default=32)

##############################
# Mixing

parser.add_argument("--mixing_hard", action="store_true", default=False)

parser.add_argument("--mixing_deterministic_start", action="store_true", default=False)

######################################################################

# args = parser.parse_args()

args, sup_args = parser.parse_known_args()

sup_args = dict([x.removeprefix("--").split("=") for x in sup_args])

if args.result_dir is None:
    args.result_dir = f"results_{args.task}_{args.model}"

######################################################################

if not args.force_cpu and torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
else:
    device = torch.device("cpu")

######################################################################

default_task_args = {
    "addition": {
        "model": "352M",
        "physical_batch_size": 25,
        "nb_train_samples": 250000,
        "nb_test_samples": 10000,
    },
    "byheart": {
        "model": "37M",
        "physical_batch_size": 25,
        "nb_train_samples": 50000,
        "nb_test_samples": 10000,
    },
    "expr": {
        "model": "352M",
        "physical_batch_size": 25,
        "nb_train_samples": 2500000,
        "nb_test_samples": 10000,
    },
    "grid": {
        "model": "37M",
        "physical_batch_size": 25,
        "nb_train_samples": 250000,
        "nb_test_samples": 10000,
    },
    "qmlp": {
        "model": "37M",
        "physical_batch_size": 10,
        "nb_train_samples": 100000,
        "nb_test_samples": 1000,
    },
    "guessop": {
        "model": "352M",
        "physical_batch_size": 25,
        "nb_train_samples": 1000000,
        "nb_test_samples": 10000,
    },
    "learnop": {
        "model": "37M",
        "physical_batch_size": 25,
        "nb_train_samples": 50000,
        "nb_test_samples": 10000,
    },
    "maze": {
        "model": "37M",
        "physical_batch_size": 5,
        "nb_train_samples": 100000,
        "nb_test_samples": 10000,
    },
    "picoclvr": {
        "model": "37M",
        "physical_batch_size": 25,
        "nb_train_samples": 250000,
        "nb_test_samples": 10000,
    },
    "rpl": {
        "model": "352M",
        "physical_batch_size": 5,
        "nb_train_samples": 2500000,
        "nb_test_samples": 10000,
    },
    "snake": {
        "model": "37M",
        "physical_batch_size": 25,
        "nb_train_samples": 250000,
        "nb_test_samples": 10000,
    },
    "stack": {
        "model": "37M",
        "physical_batch_size": 25,
        "nb_train_samples": 100000,
        "nb_test_samples": 1000,
    },
    "twotargets": {
        "model": "37M",
        "physical_batch_size": 25,
        "nb_train_samples": 50000,
        "nb_test_samples": 10000,
    },
    "memory": {
        "model": "37M",
        "physical_batch_size": 25,
        "nb_train_samples": 25000,
        "nb_test_samples": 10000,
    },
    "mixing": {
        "model": "37M",
        "physical_batch_size": 25,
        "nb_train_samples": 250000,
        "nb_test_samples": 10000,
    },
    "mnist": {
        "model": "37M",
        "physical_batch_size": 5,
        "nb_train_samples": 60000,
        "nb_test_samples": 10000,
    },
}

if args.task in default_task_args:
    for k, v in default_task_args[args.task].items():
        if getattr(args, k) is None:
            setattr(args, k, v)

######################################################################

default_model_args = {
    "17K": {
        "attention": "mha",
        "dim_model": 32,
        "dim_keys": 32,
        "dim_hidden": 32,
        "nb_heads": 2,
        "nb_blocks": 2,
    },
    "17K-C": {
        "attention": "caterpillar",
        "dim_model": 32,
        "dim_keys": 32,
        "dim_hidden": 32,
        "nb_heads": 2,
        "nb_lines": 16,
        "caterpillar_height": 4,
        "nb_blocks": 2,
    },
    "4M": {
        "attention": "mha",
        "dim_model": 256,
        "dim_keys": 32,
        "dim_hidden": 1024,
        "nb_heads": 4,
        "nb_blocks": 6,
    },
    "4M-C": {
        "attention": "caterpillar",
        "dim_model": 256,
        "dim_keys": 32,
        "dim_hidden": 1024,
        "nb_heads": 4,
        "nb_lines": 32,
        "caterpillar_height": 4,
        "nb_blocks": 6,
    },
    "37M": {
        "attention": "mha",
        "dim_model": 512,
        "dim_keys": 64,
        "dim_hidden": 2048,
        "nb_heads": 8,
        "nb_blocks": 12,
    },
    "37M-C": {
        "attention": "caterpillar",
        "dim_model": 512,
        "dim_keys": 64,
        "dim_hidden": 2048,
        "nb_heads": 8,
        "nb_lines": 256,
        "caterpillar_height": 32,
        "nb_blocks": 12,
    },
    "122M": {
        "attention": "mha",
        "dim_model": 768,
        "dim_keys": 64,
        "dim_hidden": 2048,
        "nb_heads": 8,
        "nb_blocks": 24,
    },
    "122M-C": {
        "attention": "caterpillar",
        "dim_model": 768,
        "dim_keys": 64,
        "dim_hidden": 2048,
        "nb_heads": 8,
        "nb_lines": 128,
        "nb_blocks": 24,
    },
    "352M": {
        "attention": "mha",
        "dim_model": 1024,
        "dim_keys": 64,
        "dim_hidden": 2048,
        "nb_heads": 8,
        "nb_blocks": 48,
    },
    "352M-C": {
        "attention": "caterpillar",
        "dim_model": 1024,
        "dim_keys": 64,
        "dim_hidden": 2048,
        "nb_heads": 8,
        "nb_lines": 128,
        "nb_blocks": 48,
    },
}

if args.model in default_model_args:
    for k, v in default_model_args[args.model].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
else:
    raise ValueError(f"Unknown model {args.model}")

######################################################################

try:
    os.mkdir(args.result_dir)
except FileExistsError:
    if not args.continue_training:
        print(f"result directory {args.result_dir} already exists")
        exit(1)

loss_file = open(os.path.join(args.result_dir, "loss.dat"), "a")
lambda_file = open(os.path.join(args.result_dir, "lambda.dat"), "a")

log_file = open(os.path.join(args.result_dir, args.log_filename), "a")

if args.seed >= 0:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

######################################################################


def log_string(s):
    t = time.strftime("%Y%m%d-%H:%M:%S ", time.localtime())

    if log_file is not None:
        log_file.write(t + s + "\n")
        log_file.flush()

    print(t + s)
    sys.stdout.flush()


with os.popen("sha256sum *.py") as f:
    for l in f:
        log_string(f"sha256sum {l.strip()}")

now = time.strftime("%Y%m%d-%H%M%S", time.localtime())
os.system(f"tar zcvf {args.result_dir}/src-{now}.tgz *.py *.sh")

log_string(f"argv {' '.join(sys.argv)}")

for n in vars(args):
    log_string(f"args.{n} {getattr(args, n)}")

for k, v in sup_args.items():
    log_string(f'sup_args["{k}"] "{v}"')


######################################################################


def get_lr(n_epoch, it):
    if args.legacy_lr_schedule:
        # my crude scheduling to compare to previous baseline, added
        # warmup though

        if it < args.nb_warmup_iter:
            return args.legacy_large_lr * it / args.nb_warmup_iter
        elif n_epoch < args.legacy_nb_epoch_large_lr:
            return args.legacy_large_lr
        else:
            return args.legacy_small_lr

    # from nanoGPT

    # 1) linear warmup for warmup_iter steps
    if it < args.nb_warmup_iter:
        return args.learning_rate * it / args.nb_warmup_iter
    # 2) if it > nb_decay_iter, return min learning rate
    if it > args.nb_decay_iter:
        return args.min_learning_rate
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - args.nb_warmup_iter) / (
        args.nb_decay_iter - args.nb_warmup_iter
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return args.min_learning_rate + coeff * (
        args.learning_rate - args.min_learning_rate
    )


######################################################################


def add_memex_v1(batches, memex_proba, marker_token):
    for input in batches:
        if torch.rand(1).item() < memex_proba:
            t = (
                torch.arange(1 + 2 * input.size(1), device=input.device)[None, :]
                .expand(input.size(0), -1)
                .clone()
            )

            u0 = torch.randint(input.size(1), (input.size(0), 1), device=input.device)
            caterpillar_length = args.nb_lines // args.caterpillar_height
            u1 = (
                u0
                + torch.randint(
                    caterpillar_length, (input.size(0), 1), device=input.device
                )
                + 1
            )

            m0 = (t < u0).long()
            m1 = (t >= u1).long() * (t < u1 + input.size(1)).long()

            t = t * m0 + ((-1) * (1 - m0) * (1 - m1)) + (t - u1) * m1
            m = (t < 0).long()
            n = torch.arange(input.size(0), device=input.device)[:, None].expand(
                -1, t.size(1)
            )

            new_input = input[n, t.clamp(min=0)]
            new_input = (1 - m) * new_input + m * (marker_token)

            memex_mask = new_input.new_zeros(new_input.size())
            memex_mask[:, input.size(1) :] = 1.0

            yield new_input, memex_mask

        yield input


# The marker token is not used for this one
def add_memex_v2(batches, memex_proba, marker_token):
    for input in batches:
        if torch.rand(1).item() < memex_proba:
            t = torch.arange(input.size(1) // 4, device=input.device)[None, :].expand(
                input.size(0), -1
            )
            t = t + torch.randint(
                input.size(1) - t.size(1), (t.size(0), 1), device=t.device
            )
            n = torch.arange(input.size(0), device=input.device)[:, None].expand(
                -1, t.size(1)
            )

            flash = input[n, t]
            new_input = torch.cat([input, flash], dim=1)

            memex_mask = new_input.new_zeros(new_input.size())
            memex_mask[:, input.size(1) :] = 1.0

            yield new_input, memex_mask

        else:
            yield input


def add_memex_v3(batches, memex_proba, marker_token):
    for input in batches:
        if torch.rand(1).item() < memex_proba:
            memex_len = input.size(1) // 4

            t = torch.arange(input.size(1) + memex_len, device=input.device)[
                None, :
            ].expand(input.size(0), -1)
            n = torch.arange(input.size(0), device=input.device)[:, None].expand(
                -1, t.size(1)
            )

            # Call me the tensor-spaghetti master

            trigger = torch.rand(t.size(), device=t.device)
            trigger[:, -memex_len:] = 2.0
            trigger[:, 0] = 2.0
            trigger = (trigger == trigger.min(dim=1, keepdim=True).values).long()
            memex_mask = trigger.clone()
            memex_mask[:, memex_len:] -= trigger[:, :-memex_len]
            memex_mask = memex_mask.cumsum(dim=1)

            u = 1 - memex_mask
            u[:, 0] = 0
            u = u.cumsum(dim=1)
            assert u.min() == 0
            assert u.max() == input.size(1) - 1

            v = (
                (trigger.cumsum(dim=1) - trigger).cumsum(dim=1)
                + torch.randint(
                    input.size(1) - memex_len, (input.size(0), 1), device=t.device
                )
            ) * memex_mask
            assert v.min() >= 0
            assert v.max() < input.size(1)
            u = u * (1 - memex_mask) + v * memex_mask

            new_input = input[n, u]
            assert input.max() < vocabulary_size
            assert new_input.max() < vocabulary_size
            limits = trigger.clone()
            limits[:, memex_len - 1 :] += limits[:, : -(memex_len - 1)]
            assert limits.min() == 0
            assert limits.max() == 1
            new_input = new_input * (1 - limits) + marker_token * limits
            assert marker_token < vocabulary_size
            assert new_input.max() < vocabulary_size

            yield new_input, memex_mask

        else:
            yield input


######################################################################

assert args.picocvlr_prune_properties in {"none", "train+eval", "eval"}

assert args.batch_size % args.physical_batch_size == 0


def picoclvr_pruner_horizontal_green(p):
    return not ("green" in p and ("left" in p or "right" in p))


picoclvr_pruner_train = (
    picoclvr_pruner_horizontal_green
    if args.picocvlr_prune_properties in {"train+eval"}
    else None
)

picoclvr_pruner_eval = (
    (lambda p: not picoclvr_pruner_horizontal_green(p))
    if args.picocvlr_prune_properties in {"train+eval", "eval"}
    else None
)

######################################################################

device_data = device

if args.task == "byheart":
    task = tasks.SandBox(
        problem=problems.ProblemByHeart(),
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.physical_batch_size,
        logger=log_string,
        device=device_data,
    )
    args.max_percents_of_test_in_train = -1

elif args.task == "learnop":
    task = tasks.SandBox(
        problem=problems.ProblemLearnOperator(),
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.physical_batch_size,
        logger=log_string,
        device=device_data,
    )


elif args.task == "guessop":
    task = tasks.SandBox(
        problem=problems.ProblemGuessOperator(),
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.physical_batch_size,
        logger=log_string,
        device=device_data,
    )


elif args.task == "twotargets":
    task = tasks.SandBox(
        problem=problems.ProblemTwoTargets(),
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.physical_batch_size,
        logger=log_string,
        device=device_data,
    )

elif args.task == "memory":
    task = tasks.SandBox(
        problem=problems.ProblemMemory(len_total=args.memory_len_total),
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.physical_batch_size,
        logger=log_string,
        device=device_data,
    )

elif args.task == "mixing":
    task = tasks.SandBox(
        problem=problems.ProblemMixing(
            hard=args.mixing_hard, random_start=not args.mixing_deterministic_start
        ),
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.physical_batch_size,
        logger=log_string,
        device=device_data,
    )

elif args.task == "addition":
    task = tasks.SandBox(
        problem=problems.ProblemAddition(),
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.physical_batch_size,
        logger=log_string,
        device=device_data,
    )

elif args.task == "picoclvr":
    task = tasks.PicoCLVR(
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.physical_batch_size,
        height=args.picoclvr_height,
        width=args.picoclvr_width,
        nb_colors=args.picoclvr_nb_colors,
        logger=log_string,
        device=device_data,
        pruner_train=picoclvr_pruner_train,
        pruner_eval=picoclvr_pruner_eval,
    )

elif args.task == "mnist":
    task = tasks.MNIST(
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.physical_batch_size,
        device=device_data,
    )

elif args.task == "maze":
    task = tasks.Maze(
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.physical_batch_size,
        height=args.maze_height,
        width=args.maze_width,
        nb_walls=args.maze_nb_walls,
        device=device_data,
    )

elif args.task == "snake":
    task = tasks.Snake(
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.physical_batch_size,
        height=args.snake_height,
        width=args.snake_width,
        nb_colors=args.snake_nb_colors,
        length=args.snake_length,
        prompt_length=args.snake_length // 2,
        device=device_data,
    )

elif args.task == "stack":
    task = tasks.Stack(
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.physical_batch_size,
        logger=log_string,
        nb_steps=args.stack_nb_steps,
        nb_stacks=args.stack_nb_stacks,
        nb_digits=args.stack_nb_digits,
        fraction_values_for_train=args.stack_fraction_values_for_train,
        device=device_data,
    )

elif args.task == "expr":
    task = tasks.Expr(
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        nb_variables=args.expr_nb_variables,
        sequence_length=args.expr_sequence_length,
        operand_max=args.expr_operand_max,
        result_max=args.expr_result_max,
        batch_size=args.physical_batch_size,
        device=device_data,
    )

elif args.task == "rpl":
    task = tasks.RPL(
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.physical_batch_size,
        nb_starting_values=args.rpl_nb_starting_values,
        max_input=args.rpl_max_input,
        prog_len=args.rpl_prog_len,
        nb_runs=args.rpl_nb_runs,
        no_prog=args.rpl_no_prog,
        logger=log_string,
        device=device_data,
    )

elif args.task == "grid":
    task = tasks.Grid(
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.physical_batch_size,
        size=args.grid_size,
        nb_shapes=args.grid_nb_shapes,
        nb_colors=args.grid_nb_colors,
        logger=log_string,
        device=device_data,
    )

elif args.task == "qmlp":
    task = tasks.QMLP(
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.physical_batch_size,
        result_dir=args.result_dir,
        logger=log_string,
        device=device_data,
    )

else:
    raise ValueError(f"Unknown task {args.task}")

######################################################################

log_string(f"device {device}")

vocabulary_size = task.vocabulary_size()

if args.memex_proba > 0:
    memex_marker = vocabulary_size
    vocabulary_size += 1

log_string(f"vocabulary_size {vocabulary_size}")

##############################

model = mygpt.MyGPT(
    vocabulary_size=vocabulary_size,
    dim_model=args.dim_model,
    dim_keys=args.dim_keys,
    dim_hidden=args.dim_hidden,
    nb_heads=args.nb_heads,
    nb_lines=args.nb_lines,
    caterpillar_height=args.caterpillar_height,
    nb_blocks=args.nb_blocks,
    causal=True,
    dropout=args.dropout,
    attention_layer=args.attention,
    logger=log_string,
    args=args,
)

model.to(device)

nb_parameters = sum(p.numel() for p in model.parameters())
log_string(f"nb_parameters {nb_parameters} ({int(nb_parameters/1e6)}M)")

######################################################################

nb_epochs_finished = 0

if args.no_checkpoint:
    log_string(f"not trying to load checkpoint.")

else:
    try:
        checkpoint_name = os.path.join(args.result_dir, args.checkpoint_name)
        checkpoint = torch.load(checkpoint_name)
        nb_epochs_finished = checkpoint["nb_epochs_finished"]
        model.load_state_dict(checkpoint["model_state"])
        torch.set_rng_state(checkpoint["rng_state"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])

        log_string(f"checkpoint loaded with {nb_epochs_finished} epochs finished.")

    except FileNotFoundError:
        log_string("starting from scratch.")

    except:
        log_string("error when loading the checkpoint.")
        exit(1)

######################################################################

if args.task == "expr" and args.expr_input_file is not None:
    task.produce_results(
        n_epoch=nb_epochs_finished,
        model=model,
        result_dir=args.result_dir,
        logger=log_string,
        deterministic_synthesis=args.deterministic_synthesis,
        input_file=args.expr_input_file,
    )

    exit(0)

######################################################################

nb_epochs = args.nb_epochs if args.nb_epochs > 0 else nb_epochs_default

# Compute the entropy of the training tokens

token_count = 0
for input in task.batches(split="train"):
    token_count += F.one_hot(input, num_classes=task.vocabulary_size()).sum((0, 1))
token_probas = token_count / token_count.sum()
entropy = -torch.xlogy(token_probas, token_probas).sum()
train_set_perplexity = math.exp(entropy)

######################################################################
# A bit of paranoia never hurts

if args.max_percents_of_test_in_train >= 0:

    def subsets_as_tuples(batches, cs):
        s = set()
        for batch in batches:
            for x in batch:
                s.add(tuple([v.item() for v in x]))
                if len(s) == cs:
                    yield s
                    s = set()
        yield s

    nb_test, nb_in_train = 0, 0
    for test_subset in subsets_as_tuples(task.batches(split="test"), 25000):
        in_train = set()
        for train_subset in subsets_as_tuples(task.batches(split="train"), 25000):
            in_train.update(test_subset.intersection(train_subset))
        nb_in_train += len(in_train)
        nb_test += len(test_subset)

    log_string(
        f"data_check {nb_in_train*100/nb_test:.02f}% ({nb_in_train}/{nb_test}) of test samples are in the train set"
    )

    assert (
        nb_in_train <= args.max_percents_of_test_in_train * nb_test / 100
    ), f"More than {args.max_percents_of_test_in_train}% of test samples are in the train set"

##############################

if "calibrate" in sup_args:
    for input in task.batches(split="train", desc="calibrate"):
        input = input.to(device)
        output = model(mygpt.BracketedSequence(input)).x

    for n, m in model.named_modules():
        for a in dir(m):
            x = getattr(m, a)
            if isinstance(x, mygpt.Calibrator):
                print(f"####### ${n} | ${a} ########################")
                mean, std = x.moments()
                print("mean\n", mean, "\n")
                print("std\n", std, "\n")
                print(f"############################################\n\n")

    exit(0)

##############################

nb_samples_seen = 0

if nb_epochs_finished >= nb_epochs:
    task.produce_results(
        n_epoch=nb_epochs_finished,
        model=model,
        result_dir=args.result_dir,
        logger=log_string,
        deterministic_synthesis=args.deterministic_synthesis,
    )

time_pred_result = datetime.datetime.now()

it = 0

n_batch = 0


def the_dot_products(value1, value2, params):
    g1g1, g1g2, g2g2 = 0, 0, 0
    for p in params:
        g1 = torch.autograd.grad(value1, p, retain_graph=True)[0]
        g2 = torch.autograd.grad(value2, p, retain_graph=True)[0]
        g1g1 += g1.pow(2).sum()[None]
        g2g2 += g2.pow(2).sum()[None]
        g1g2 += (g1 * g2).sum()[None]
    return torch.cat([g1g1, g1g2, g2g2])


def update_ave_grad(value, params, name, eps=1e-3):
    for p in params:
        g = torch.autograd.grad(value, p, retain_graph=True)[0]
        ag = getattr(p, name) if hasattr(p, name) else 0
        setattr(p, name, (1 - eps) * ag + eps * g)


def norm(params, name):
    s = 0
    for p in params:
        s += getattr(p, name).pow(2).sum()
    return s


for n_epoch in range(nb_epochs_finished, nb_epochs):
    if args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optim == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer {args.optim}.")

    model.train()

    nb_train_samples, acc_train_loss, acc_train_inner_loss = 0, 0.0, 0.0

    memex_proba = (
        args.memex_proba
        if args.memex_nb_epochs is None or n_epoch < args.memex_nb_epochs
        else 0.0
    )

    log_string(f"memex_proba {memex_proba}")

    warnings.warn("memex v3", RuntimeWarning)
    train_batches = task.batches(split="train")
    #add_memex_v3(
    #    batches=task.batches(split="train"),
    #    memex_proba=memex_proba,
    #    marker_token=memex_marker,
    #)

    def add_none(it):
        for x in it:
            yield x
        yield None

    nb_acc_samples = 0

    for input in add_none(train_batches):
        if input is not None:
            if type(input) is tuple:
                input, memex_mask = input
                memex_mask = memex_mask.to(device)
            else:
                memex_mask = None

            model.reset_inner_loss()
            input = input.to(device)

            output = model(mygpt.BracketedSequence(input)).x

            if memex_mask is None:
                loss = F.cross_entropy(output.transpose(1, 2), input)
            else:
                loss = F.cross_entropy(output.transpose(1, 2), input, reduction="none")
                loss_regular = (loss * (1 - memex_mask)).mean()
                loss_memex = (loss * memex_mask).mean()

                if it < 100 or torch.rand(1) < 0.01:
                    update_ave_grad(loss_regular, model.parameters(), "grad_regular")
                    update_ave_grad(loss_memex, model.parameters(), "grad_memex")
                    norm_regular = norm(model.parameters(), "grad_regular")
                    norm_memex = norm(model.parameters(), "grad_memex")
                    l_memex = (
                        max(norm_regular, norm_memex) - norm_regular
                    ) / norm_memex

                loss = loss_regular + l_memex * loss_memex

            inner_loss = model.get_inner_loss()

            acc_train_loss += loss.item() * input.size(0)
            acc_train_inner_loss += inner_loss.item() * input.size(0)

            nb_train_samples += input.size(0)
            nb_samples_seen += input.size(0)

            total_loss = loss + (
                args.rho_inner_loss * inner_loss if args.rho_inner_loss > 0 else 0.0
            )

            it += 1
            lr = get_lr(n_epoch, it)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

                # log_string(f"learning_rate {lr}")

            total_loss.backward()
            nb_acc_samples += input.size(0)

        if (input is None and nb_acc_samples > 0) or nb_acc_samples == args.batch_size:
            assert nb_acc_samples <= args.batch_size
            optimizer.step()
            grad_norm = sum([p.grad.pow(2).sum() for p in model.parameters()]).sqrt()
            loss_file.write(f"{n_epoch} {n_batch} {loss.item()} {grad_norm.item()}\n")
            lambda_file.write(
                f"{n_epoch} {n_batch} \n"
            )
            optimizer.zero_grad()
            nb_acc_samples = 0
            n_batch += 1

    with torch.autograd.no_grad():
        model.eval()

        nb_test_samples, acc_test_loss = 0, 0.0

        for input in task.batches(split="test"):
            input = input.to(device)

            output = model(mygpt.BracketedSequence(input)).x
            loss = F.cross_entropy(output.transpose(1, 2), input)
            acc_test_loss += loss.item() * input.size(0)
            nb_test_samples += input.size(0)

        log_string(
            f"loss {n_epoch} train_loss {acc_train_loss/nb_train_samples} train_inner_loss {acc_train_inner_loss/nb_train_samples} test_prediction {acc_test_loss/nb_test_samples}"
        )

        task.produce_results(
            n_epoch=n_epoch,
            model=model,
            result_dir=args.result_dir,
            logger=log_string,
            deterministic_synthesis=args.deterministic_synthesis,
        )

        train_perplexity = math.exp(min(100, acc_train_loss / nb_train_samples))
        test_perplexity = math.exp(min(100, acc_test_loss / nb_test_samples))

        log_string(
            f"perplexity {n_epoch} train_set {train_set_perplexity} train_prediction {train_perplexity} test_prediction {test_perplexity}"
        )

        time_current_result = datetime.datetime.now()
        log_string(
            f"next_result {time_current_result + (time_current_result - time_pred_result)}"
        )
        time_pred_result = time_current_result

    checkpoint = {
        "nb_epochs_finished": n_epoch + 1,
        "model_state": model.state_dict(),
        "rng_state": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state()

    checkpoint_name = os.path.join(args.result_dir, args.checkpoint_name)
    torch.save(checkpoint, checkpoint_name)
    log_string(f"saved checkpoint {checkpoint_name}")

######################################################################
