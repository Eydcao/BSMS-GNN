import wandb
import torch
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pytz
from trainer import Trainer
from datetime import datetime
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from trainer import Trainer
from datasets import DATSET_HANDLER
from models import BSMS_Simulator
from utils import set_seed, Normalizer, restore_model, rollout_one_traj

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def run_rollout(cfg):
    set_seed(cfg.base_seed)
    tc_rng = torch.Generator()
    tc_rng.manual_seed(cfg.base_seed)

    print(OmegaConf.to_yaml(cfg))

    if cfg.board:
        wandb_run = wandb.init(
            project="rollout",
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    else:
        wandb_run = None

    # Model and dataset creation
    dataset_name = cfg.datasets.tf_dataset_name
    model = BSMS_Simulator(cfg.model)
    # restore the model
    ckpt_path = os.path.join(cfg.restore_dir, str(cfg.restore_step) + "_params.pth")
    restore_model(model, ckpt_path)
    model.eval()

    # Trainer creation
    trainer = Trainer(model, cfg.model, cfg.opt)

    # Data loaders creation
    test_datapipe = DATSET_HANDLER[dataset_name](cfg.datasets, 0, cfg.base_seed, "rollout")
    test_loader = DataLoader(test_datapipe, batch_size=1, num_workers=0, pin_memory=True)

    # Printing meta info of the training
    time_stamp = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    print("stamp: {}".format(time_stamp))

    # Printing meta info of the training
    time_stamp = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    stamp = time_stamp
    print("stamp: {}".format(stamp))

    # Initialize error accumulator
    rmse_accumulator = None
    rmse_accumulators_of_channel = None
    rmse_accumulators_of_time = None

    for batch in tqdm(test_loader):
        # to device
        batch = trainer.move_to_device(batch)
        # (1, T-1, N, C+dim+1), (1, T-1, N, C), (1, T-1, N, 1), [m_gs], [m_pooled_ids]
        # where dim + 1 is mesh_pos and node_type
        node_info_inp, node_info_tar, node_mask, m_gs, m_ids = batch
        # remove the batch dimension
        # (T-1, N, C+dim+1)), (T-1, N, C), (T-1, N, 1), [m_gs], [m_pooled_ids]
        node_info_inp = node_info_inp.squeeze(0)
        node_info_tar = node_info_tar.squeeze(0)
        node_mask = node_mask.squeeze(0)

        # Prepare the pred results container
        results = node_info_tar.new_zeros(node_info_tar.shape)
        # Record Initial frame IC
        # (1, N, C+dim+1)
        IC = node_info_inp[0:1].clone()

        # Rollout
        # (T-1, N, C)
        results = rollout_one_traj(trainer, IC, results, node_mask[0], m_gs, m_ids, cfg)

        # If the error accumulators are not initialized, initialize them
        if rmse_accumulator is None:
            rmse_accumulator = Normalizer(size=1, device=device, name="rmse_accumulator")
        if rmse_accumulators_of_channel is None:
            rmse_accumulators_of_channel = Normalizer(
                size=results.shape[-1], device=device, name="rmse_accumulators_of_channel"
            )
        if rmse_accumulators_of_time is None:
            rmse_accumulators_of_time = Normalizer(
                size=results.shape[0], device=device, name="rmse_accumulators_of_time"
            )

        # (T-1, N, C)
        se = (results - node_info_tar) ** 2
        # Get RMSE
        rmse = torch.sqrt((se * node_mask).sum() / node_mask.sum() / se.shape[-1])
        # make shpae (1,1)
        rmse = rmse.unsqueeze(0).unsqueeze(0)
        # GET RMSE error over the nodes only
        # so the shape is (T-1,C)
        rmse_c = torch.sqrt((se * node_mask).sum(dim=1) / node_mask.sum(dim=1))
        rmse_t = rmse_c.detach().clone().T

        # Accumulate the error
        rmse_accumulator(rmse, accumulate=True)
        rmse_accumulators_of_channel(rmse_c, accumulate=True)
        rmse_accumulators_of_time(rmse_t, accumulate=True)

    # Post-processing: compute the overall statistics for each dataset type
    print("\n-------------- Printing error averaged over time and channel --------------\n")
    mean_rmse, std_rmse = rmse_accumulator.mean(), rmse_accumulator.std_with_epsilon()
    # print(mean_rmse.shape)
    # to numpy
    mean_rmse = mean_rmse.cpu().numpy()
    std_rmse = std_rmse.cpu().numpy()

    print(f"Mean error:   {mean_rmse.item():.6f}")
    print(f"Std error:    {std_rmse.item():.6f}")
    print()
    print()

    print("-------------- Printing error for each channel --------------\n")
    mean_rmse_c, std_rmse_c = rmse_accumulators_of_channel.mean(), rmse_accumulators_of_channel.std_with_epsilon()
    # print(mean_rmse_c.shape)
    # to numpy
    mean_rmse_c = mean_rmse_c.cpu().numpy()
    std_rmse_c = std_rmse_c.cpu().numpy()

    print(f"Mean error each channel:   {mean_rmse_c.tolist()}")
    print(f"Std error each channel:    {std_rmse_c.tolist()}")
    print()
    print()

    print("-------------- Printing error for 0, 5, 10, 50, last steps --------------\n")
    mean_rmse_t, std_rmse_t = rmse_accumulators_of_time.mean(), rmse_accumulators_of_time.std_with_epsilon()
    # print(mean_rmse_t.shape)
    # to numpy
    mean_rmse_t = mean_rmse_t.cpu().numpy()
    std_rmse_t = std_rmse_t.cpu().numpy()

    T_len = mean_rmse_t.shape[0]
    if T_len >= 51:
        print(f"Mean error at 0, 5, 10, last steps: {mean_rmse_t[[0, 5, 10, 50, -1]]}")
        print(f"Std error at 0, 5, 10, last steps: {std_rmse_t[[0, 5, 10, 50, -1]]}")
    elif T_len >= 1:
        print(f"Mean error at 0, 5, last steps: {mean_rmse_t[[0, 5, 10, -1]]}")
        print(f"Std error at 0, 5, last steps: {std_rmse_t[[0, 5, 10, -1]]}")
    elif T_len >= 6:
        print(f"Mean error at 0, 5, last steps: {mean_rmse_t[[0, 5, -1]]}")
        print(f"Std error at 0, 5, last steps: {std_rmse_t[[0, 5, -1]]}")
    elif T_len >= 2:
        print(f"Mean error at 0, last steps: {mean_rmse_t[[0, -1]]}")
        print(f"Std error at 0, last steps: {std_rmse_t[[0, -1]]}")
    else:
        print("There is only one time step, no statistics to show.")

    print()
    print()


@hydra.main(version_base=None, config_path="../configs/", config_name="default")
def main(cfg: DictConfig):
    run_rollout(cfg)


if __name__ == "__main__":
    main()
