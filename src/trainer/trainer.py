import torch
import numpy as np
from utils import WarmupCosineDecayScheduler
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, model, model_cfg, opt_cfg):
        print("flash_sdp_enabled", torch.backends.cuda.flash_sdp_enabled())  # True
        self.model = model
        self.model_cfg = model_cfg
        self.opt_cfg = opt_cfg
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            self.model = torch.nn.DataParallel(self.model)
            print("Model wrapped by DataParallel", flush=True)

        self.device = device
        self.model.to(device)
        print("Model moved to {}".format(device), flush=True)

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=opt_cfg.peak_lr,
            weight_decay=opt_cfg.weight_decay,
        )
        self.lr_scheduler = WarmupCosineDecayScheduler(
            optimizer=self.optimizer,
            warmup=opt_cfg.warmup_steps,
            max_iters=opt_cfg.decay_steps,
        )

        print(self.model, flush=True)
        self.train_step = 0

    def _model_forward(self, data):
        """
        A wrapper to call model.forward.

        Parameters
        ----------
        data : PyTorch Geometric Data object
            Contains node_in, node_tar, node_mask, m_ids, m_gs.

        Returns
        -------
        ave_mse : torch.Tensor
            Average mean squared error.
        pred_target : torch.Tensor
            Predicted target.
        """
        pred_target = self.model(
            data, self.model_cfg.consistent_mesh, self.train_step < self.model_cfg.accumulation_steps
        )
        return pred_target

    def get_label_mask(self, data):
        """
        Extract the label data from the input data.

        Parameters
        ----------
        data : PyTorch Geometric Data object
            Contains node_in, node_tar, node_mask, m_ids, m_gs.

        Returns
        -------
        torch.Tensor
            Label data, and label mask.
        """
        if self.model_cfg.consistent_mesh:
            node_in, node_tar, node_mask, m_gs, m_ids = data
        else:
            node_in, node_tar, node_mask = data[0].x, data[0].y, data[0].mask
        return node_tar, node_mask

    def _loss_fn(self, data):
        """
        Calculate the loss function.

        Parameters
        ----------
        data : PyTorch Geometric Data object
            Contains node_in, node_tar, node_mask, m_ids, m_gs.

        Returns
        -------
        torch.Tensor
            Loss value in RMSE
        """
        pred = self.get_pred(data)
        tar, mask = self.get_label_mask(data)

        se = (pred - tar) ** 2
        rmse = torch.sqrt((se * mask).sum() / mask.sum() / se.shape[-1])
        return rmse

    def move_to_device(self, data):
        """
        Move data to the specified device.

        Parameters
        ----------
        data : list, tuple or torch.Tensor
            Data to move to device.

        Returns
        -------
        list or torch.Tensor
            Data moved to device.
        """
        if isinstance(data, (list, tuple)):
            return [self.move_to_device(d) for d in data]
        else:
            return data.to(self.device)

    def summary(self, data):
        """
        Print a summary of the model.

        Parameters
        ----------
        data : list, tuple or torch.Tensor
            Data to use for summary.
        """
        data = self.move_to_device(data)
        if isinstance(data, (list, tuple)):
            summary(self.model, data[0].size()[1:], data[0].size()[1:])
        else:
            summary(self.model, data.size()[1:])

    def iter(self, data):
        """
        Train the model for one iteration.

        Parameters
        ----------
        data : PyTorch Geometric Data object
            Contains node_in, node_tar, node_mask, m_ids, m_gs.
        """
        data = self.move_to_device(data)
        loss = self._loss_fn(data)
        # do nothing during the warmup phase
        if self.train_step >= self.model_cfg.accumulation_steps:
            loss.backward()

            # Gradient clipping
            model = self.model.module if hasattr(self.model, "module") else self.model
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.opt_cfg.gnorm_clip)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        self.train_step += 1

    def save(self, save_dir):
        """
        Save the model parameters.

        Parameters
        ----------
        save_dir : str
            Directory to save the model parameters.
        """
        model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(model.state_dict(), f"{save_dir}/{self.train_step}_params.pth")
        print(f"Saved to {save_dir}, step {self.train_step}")

    def restore(self, save_dir, step, restore_opt_state=True):
        """
        Restore the model parameters.

        Parameters
        ----------
        save_dir : str
            Directory to restore the model parameters from.
        step : int
            Training step to restore.
        restore_opt_state : bool, optional
            Flag to restore optimizer state, by default True.
        """
        params_path = f"{save_dir}/{step}_params.pth"
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.load_state_dict(torch.load(params_path, map_location=device))
        print(f"Restored params from {save_dir}, step {step}")
        if restore_opt_state:
            # opt_path = f"{save_dir}/{step}_opt_state.pth"
            # self.optimizer.load_state_dict(torch.load(opt_path, map_location=device))
            # print(f"Restored optimizer state from {opt_path}")
            pass
            # TODO later

    def get_loss(self, data):
        """
        Get the loss value.

        Parameters
        ----------
        data : PyTorch Geometric Data object
            Contains node_in, node_tar, node_mask, m_ids, m_gs.

        Returns
        -------
        np.ndarray
            Loss value.
        """
        data = self.move_to_device(data)
        loss = self._loss_fn(data)
        return loss

    def get_pred(self, data):
        """
        Get the prediction.

        Parameters
        ----------
        data : PyTorch Geometric Data object
            Contains node_in, node_tar, node_mask, m_ids, m_gs.

        Returns
        -------
        np.ndarray
            Predictions.
        """
        data = self.move_to_device(data)
        predict = self._model_forward(data)
        return predict

    def get_error(self, data, relative=True):
        """
        Calculate the relative error for each channel and output both mean and std.

        Parameters
        ----------
        data : PyTorch Geometric Data object
            Contains node_in, node_tar, node_mask, m_ids, m_gs.
        relative : bool, optional
            Flag to indicate if relative error is to be calculated, by default True.

        Returns
        -------
        tuple
            Mean and std of the error.
        """
        # (B, N, C)
        predict = self.get_pred(data).detach().cpu().numpy()
        target, node_mask = self.get_label_mask(data)
        target = target.detach().cpu().numpy()
        node_mask = node_mask.detach().cpu().numpy()

        # (B, N, C)
        masked_error = np.sqrt(np.where(node_mask, (predict - target) ** 2, 0))

        if relative:
            # for each B, calculate the scale of each C at N axis
            tar_sqr = np.where(node_mask, target**2, 0)
            # (B, 1, C)
            tar_sqr_mean = np.sum(tar_sqr, axis=(1), keepdims=True) / (
                np.sum(node_mask, axis=(1), keepdims=True) + 1e-6
            )
            tar_scale = np.sqrt(tar_sqr_mean) + 1e-6
            # (B, N, C)
            masked_error /= tar_scale

        # get mean and std along the B, N axis
        error_mean = np.mean(masked_error, axis=(0, 1))
        error_std = np.std(masked_error, axis=(0, 1))

        return error_mean, error_std
