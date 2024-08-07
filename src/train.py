import os
import torch
import hydra
import wandb
import pytz
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader

from trainer import Trainer
from datasets import DATSET_HANDLER
from models import BSMS_Simulator
from utils import set_seed, timer, board_loss, print_error, eval_plot, get_data_from_looper, InfiniteDataLooper

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # NOTE hardcode now to enforce single gpu


def run_train(cfg):
    """
    Run the training loop.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing training parameters.
    """
    set_seed(cfg.base_seed)
    tc_rng = torch.Generator()
    tc_rng.manual_seed(cfg.base_seed)

    print(OmegaConf.to_yaml(cfg))

    if cfg.board:
        wandb.init(
            project="train",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # Model and dataset creation
    dataset_name = cfg.datasets.tf_dataset_name
    model = BSMS_Simulator(cfg.model)
    train_datapipe = DATSET_HANDLER[dataset_name](cfg.datasets, cfg.dataset_workers, cfg.base_seed, "train")
    test_datapipe = DATSET_HANDLER[dataset_name](cfg.datasets, cfg.dataset_workers, cfg.base_seed, "test")

    # Trainer creation
    trainer = Trainer(model, cfg.model, cfg.opt)

    # Data loaders creation
    train_loader = DataLoader(train_datapipe, batch_size=cfg.batch, num_workers=cfg.dataset_workers, pin_memory=True)
    test_loader = DataLoader(test_datapipe, batch_size=cfg.batch, num_workers=cfg.dataset_workers, pin_memory=True)

    # Printing meta info of the training
    time_stamp = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    print("stamp: {}".format(time_stamp))

    # Infinite data loopers for training and testing
    train_loopers = InfiniteDataLooper(train_loader)
    test_loopers = InfiniteDataLooper(test_loader)

    total_steps = cfg.epochs * cfg.steps_per_epoch
    for _ in range(total_steps + 1):
        train_data = get_data_from_looper(train_loopers, tc_rng, cfg.datasets)

        # Log loss
        if (
            (trainer.train_step % cfg.loss_freq == 0)
            or (trainer.train_step % (cfg.loss_freq // 10) == 0 and trainer.train_step <= cfg.loss_freq)
            or (trainer.train_step % (cfg.loss_freq // 10) == 0 and trainer.train_step >= total_steps - cfg.loss_freq)
        ):
            with torch.no_grad():
                # Train loss and error
                print_error(trainer, train_data, "train")
                board_loss(trainer, train_data, "train", cfg)

                # Test loss and error
                test_data = get_data_from_looper(test_loopers, tc_rng, cfg)
                print_error(trainer, test_data, "test")
                board_loss(trainer, test_data, "test", cfg)

        # Log test error plot
        if cfg.plot and (
            (trainer.train_step % cfg.plot_freq == 0)
            or (trainer.train_step % (cfg.plot_freq // 10) == 0 and trainer.train_step <= cfg.plot_freq)
            or (trainer.train_step % (cfg.plot_freq // 10) == 0 and trainer.train_step >= total_steps - cfg.plot_freq)
        ):
            test_data = get_data_from_looper(test_loopers, tc_rng, cfg)
            eval_plot(trainer, test_data, "test", cfg)

        # Save checkpoint
        if trainer.train_step % cfg.save_freq == 0:
            ckpt_dir = f"{cfg.dump_dir}/ckpts/{cfg.project}/{dataset_name}/{time_stamp}"
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            print("Current time: " + datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S"))
            trainer.save(ckpt_dir)

        # Training iteration
        trainer.iter(train_data)

        # Time estimation
        if trainer.train_step == cfg.time_warm:
            timer.tic("time estimate")
        if trainer.train_step > 0 and (trainer.train_step % cfg.time_freq == 0):
            ratio = (trainer.train_step - cfg.time_warm) / total_steps
            timer.estimate_time("time estimate", ratio)

    if cfg.board:
        wandb.finish()


@hydra.main(version_base=None, config_path="../configs/", config_name="default")
def main(cfg: DictConfig):
    """
    Main function to run the training.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing training parameters.
    """
    run_train(cfg)


if __name__ == "__main__":
    main()
