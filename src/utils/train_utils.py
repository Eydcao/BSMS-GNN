from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
import torch
from .basic import make_image, merge_images, plt_to_wandb
import wandb


def cross_batch_process(pairs, tc_rng, scheme):
    """
    pairs: (bs, pairs, 2, c, x1, ..., xd)
    """
    if scheme == "random":
        bs, ps, *shape = pairs.shape
        pairs = pairs.view(-1, *pairs.shape[2:])  # (bs * pairs, 2, c, x1, ..., xd)
        idx = torch.randperm(pairs.shape[0], generator=tc_rng)
        pairs = pairs[idx]
        pairs = pairs.view(bs, ps, *pairs.shape[1:])  # (bs, pairs, 2, c, x1, ..., xd)
        return pairs
    elif scheme == "no":
        return pairs
    else:
        raise ValueError("scheme {} not supported".format(scheme))


def board_loss(trainer, data, prefix, cfg):
    loss = trainer.get_loss(data)
    print(f"train step: {trainer.train_step}, {prefix}_rmse: {loss}")
    if cfg.board:
        wandb.log({"step": trainer.train_step, f"{prefix}_rmse": loss})


def print_error(trainer, data, prefix):
    # print error
    error_mean, error_std = trainer.get_error(data)
    x = np.arange(len(error_mean))
    list_elements = [x]
    headers = [f"{prefix}"]

    for cid in range(len(error_mean)):
        list_elements.append([error_mean[cid]])
        list_elements.append([error_std[cid]])
        headers.append(f"rel_e_mean, c:{cid}")
        headers.append(f"rel_e_std, c:{cid}")

    # Transpose list_elements to match tabulate input requirements
    table = list(zip(*list_elements))
    print(tabulate(table, headers=headers, tablefmt="grid"))


def _get_label(data):
    """
    Extract the label data from the input data.
    """
    (node_in, node_tar), m_gs, m_ids = data
    return node_in, node_tar


# NOTE currently do not use directly as mesh plot is not trivial here
def eval_plot(trainer, data, prefix, cfg):
    print("currently do not use directly as mesh plot is not trivial here")
    return

    pred, _ = trainer.get_pred(data)  # (bs, N, C)
    pred = pred.detach().cpu().numpy()
    input, target = _get_label(data)
    input = input.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    diff = pred - target
    bs, n, c = diff.shape
    plot_dict = {}
    # always plot for leading 0 batch
    bid = 0
    # 4 as row number because we have input, label, pred, diff
    figs_array = [[None for i in range(4)] for j in range(c)]
    for cid in range(c):
        figs_array[cid][0] = make_image(
            input[bid, :, cid],
            wandb=False,
            title=f"step:{trainer.train_step}, bid:{bid},cid:{cid},input",
        )
        figs_array[cid][1] = make_image(
            target[bid, :, cid],
            wandb=False,
            title=f"step:{trainer.train_step}, bid:{bid},cid:{cid},label",
        )
        figs_array[cid][2] = make_image(
            pred[bid, :, cid],
            wandb=False,
            title=f"step:{trainer.train_step}, bid:{bid},cid:{cid},pred",
        )
        figs_array[cid][3] = make_image(
            diff[bid, :, cid],
            wandb=False,
            title=f"step:{trainer.train_step}, bid:{bid},cid:{cid},diff",
        )
    merged_image = merge_images(figs_array, spacing=0)
    plot_dict[f"{prefix}"] = plt_to_wandb(merged_image, cfg={""})
    plt.close("all")
    if cfg.board:
        wandb.log({"step": trainer.train_step, **plot_dict})


def get_data_from_looper(looper, tc_rng, cfg):
    data = next(looper)

    # print("data is ", data)
    # print("data 0 face, latter half - formmaer half", data[0].face[:len(data[0].face) // 2]-data[0].face[len(data[0].face) // 2:])
    # print("data 1 face, latter half - formmaer half", data[1].face[:len(data[1].face) // 2]-data[1].face[len(data[1].face) // 2:])

    return data
