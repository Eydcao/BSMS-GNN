from typing import Optional
import torch
from functools import wraps
import time
import matplotlib.pyplot as plt
import io
import pytz
from datetime import datetime, timedelta
import re
import numpy as np
import torch.optim as optim
import random
from PIL import Image
import wandb
import matplotlib

# see https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
linestyles = {
    "solid": "solid",
    "dotted": "dotted",
    "dashed": "dashed",
    "dashdot": "dashdot",
    "loosely dotted": (0, (1, 10)),
    "dotted": (0, (1, 1)),
    "densely dotted": (0, (1, 1)),
    "long dash with offset": (5, (10, 3)),
    "loosely dashed": (0, (5, 10)),
    "dashed": (0, (5, 5)),
    "densely dashed": (0, (5, 1)),
    "loosely dashdotted": (0, (3, 10, 1, 10)),
    "dashdotted": (0, (3, 5, 1, 5)),
    "densely dashdotted": (0, (3, 1, 1, 1)),
    "dashdotdotted": (0, (3, 5, 1, 5, 1, 5)),
    "loosely dashdotdotted": (0, (3, 10, 1, 10, 1, 10)),
    "densely dashdotdotted": (0, (3, 1, 1, 1, 1, 1)),
}


def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def strip_ansi_codes(s):
    return re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", s)


def get_git_hash():
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def get_sentence_from_ids(ids, tokenizer, clean=True):
    tokens = tokenizer.convert_ids_to_tokens(ids)
    if clean:
        tokens_clean = [token for token in tokens if token not in tokenizer.all_special_tokens]
        sentence = tokenizer.convert_tokens_to_string(tokens_clean).replace(" ##", "")
    else:
        sentence = tokenizer.convert_tokens_to_string(tokens)
    return tokens, sentence


def find_sublist(a, b):
    a = list(a)
    b = list(b)
    len_a, len_b = len(a), len(b)
    for i in range(len_b - len_a + 1):
        if b[i : i + len_a] == a:
            return i + len_a  # add the length of a to the index
    return -1


def print_dot(i, freq=100, marker="."):
    if i % freq == 0:
        print(i, end="", flush=True)
    print(marker, end="", flush=True)
    if (i + 1) % freq == 0:
        print("", flush=True)


def timeit_full(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds", flush=True)
        return result

    return timeit_wrapper


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} Took {total_time:.4f} seconds", flush=True)
        return result

    return timeit_wrapper


def get_days_hours_mins_seconds(time_consumed_in_seconds):
    time_consumed = time_consumed_in_seconds
    days_consumed = int(time_consumed // (24 * 3600))
    time_consumed %= 24 * 3600
    hours_consumed = int(time_consumed // 3600)
    time_consumed %= 3600
    minutes_consumed = int(time_consumed // 60)
    seconds_consumed = int(time_consumed % 60)
    return days_consumed, hours_consumed, minutes_consumed, seconds_consumed


class TicToc:
    def __init__(self):
        self.start_time = {}
        self.end_time = {}

    def tic(self, name):
        self.start_time[name] = time.perf_counter()

    def toc(self, name):
        self.end_time[name] = time.perf_counter()
        total_time = self.end_time[name] - self.start_time[name]
        print(f"{name} Took {total_time:.4f} seconds", flush=True)

    def estimate_time(self, name, ratio, samples_processed=None, timezone_str="America/Los_Angeles"):
        print("==========================Time Estimation Starts==========================")
        timezone = pytz.timezone(timezone_str)
        current_time = datetime.now(timezone)
        current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Current time in {timezone_str}:", current_time_str)
        self.end_time[name] = time.perf_counter()
        time_consumed = self.end_time[name] - self.start_time[name]
        days_consumed, hours_consumed, minutes_consumed, seconds_consumed = get_days_hours_mins_seconds(time_consumed)
        print(f"Time consumed: {days_consumed}-{hours_consumed:02d}:{minutes_consumed:02d}:{seconds_consumed:02d}")
        if samples_processed is not None:
            samples_processed_per_second = samples_processed / time_consumed
            print(f"Samples processed per second: {samples_processed_per_second:.2f}")
        time_remaining = time_consumed * (1 - ratio) / ratio
        days_remaining, hours_remaining, minutes_remaining, seconds_remaining = get_days_hours_mins_seconds(
            time_remaining
        )
        print(
            f"Estimated remaining time: {days_remaining}-{hours_remaining:02d}:{minutes_remaining:02d}:{seconds_remaining:02d}"
        )
        time_total = time_consumed / ratio
        days_total, hours_total, minutes_total, seconds_total = get_days_hours_mins_seconds(time_total)
        print(f"Estimated total time: {days_total}-{hours_total:02d}:{minutes_total:02d}:{seconds_total:02d}")
        finish_time = current_time + timedelta(seconds=time_remaining)
        finish_time_str = finish_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Percentage finished: {ratio * 100 :.2f}%")
        print(f"Estimated finishing time in {timezone_str}:", finish_time_str)
        print("==========================Time Estimation Ends==========================", flush=True)


timer = TicToc()


class WarmupCosineDecayScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        if epoch <= self.warmup:
            lr_factor = epoch * 1.0 / self.warmup
        else:
            progress = (epoch - self.warmup) / (self.max_num_iters - self.warmup)
            lr_factor = 0.5 * (1 + np.cos(np.pi * progress))
        return lr_factor


def plt_to_wandb(fig, cfg=None):
    """
    Converts a Matplotlib figure to a wandb.Image.
    Parameters:
    - fig: Matplotlib figure to be converted.
    Returns:
    - wandb.Image object for logging.
    """
    # Save the figure to a BytesIO object
    if cfg is None:
        cfg = {}

    if type(fig) == matplotlib.figure.Figure:
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        # Use PIL to open the image from the BytesIO object
        image = Image.open(buf)
        # Close the buffer
        buf.close()
    else:  # already a PIL image
        image = fig
    # Convert to wandb.Image
    wandb_image = wandb.Image(image, **cfg)
    return wandb_image


def make_image(array, wandb=True, title=None, cfg=None):
    fig = plt.figure(figsize=(4, 4))
    cmap = "bwr"
    vmax = np.max(np.abs(array))
    plt.imshow(array, cmap=cmap, vmin=-vmax, vmax=vmax)
    plt.colorbar()
    if title is not None:
        plt.title(title)
    if not wandb:
        return fig
    wandb_image = plt_to_wandb(fig, cfg)
    plt.close("all")
    return wandb_image


def merge_images(figs_2d, spacing=0):
    """
    Converts a 2D list of Matplotlib figures to a single PIL image arranged in a grid.
    Parameters:
    - figs_2d: 2D list of Matplotlib figures to be merged and converted.
    - spacing: Space between images in pixels.
    Returns:
    - Merged and converted PIL image.
    """
    # Store the merged images of each row
    row_images = []
    max_row_height = 0
    total_width = 0

    # Process each row
    for figs_row in figs_2d:
        imgs = []
        bufs = []
        for fig in figs_row:
            buf = io.BytesIO()
            bufs.append(buf)
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            img = Image.open(buf)
            imgs.append(img)

        # Determine the total size for this row
        row_width = sum(img.width for img in imgs) + spacing * (len(imgs) - 1)
        row_height = max(img.height for img in imgs)
        max_row_height = max(max_row_height, row_height)
        total_width = max(total_width, row_width)

        # Create row image and paste figures
        row_image = Image.new("RGB", (row_width, row_height))
        x_offset = 0
        for img in imgs:
            row_image.paste(img, (x_offset, 0))
            x_offset += img.width + spacing

        row_images.append(row_image)

        # Close all the buffers
        for buf in bufs:
            buf.close()

    # Determine total size for the final merged image
    total_height = sum(img.height for img in row_images) + spacing * (len(row_images) - 1)

    # Create final merged image and paste row images
    merged_img = Image.new("RGB", (total_width, total_height))
    y_offset = 0
    for img in row_images:
        merged_img.paste(img, (0, y_offset))
        y_offset += img.height + spacing

    return merged_img


def degree(index: torch.Tensor, num_nodes: Optional[int] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    r"""Computes the (unweighted) degree of a given one-dimensional index
    tensor.

    Args:
        index (LongTensor): Index tensor.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dtype (:obj:`torch.dtype`, optional): The desired data type of the
            returned tensor.

    :rtype: :class:`Tensor`

    Example:
        >>> row = torch.tensor([0, 1, 0, 2, 0])
        >>> degree(row, dtype=torch.long)
        tensor([3, 1, 1])
    """
    N = torch.max(index) + 1
    N = int(N)
    out = torch.zeros((N,), dtype=dtype, device=index.device)
    one = torch.ones((index.size(0),), dtype=out.dtype, device=out.device)
    return out.scatter_add_(0, index, one)


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)
