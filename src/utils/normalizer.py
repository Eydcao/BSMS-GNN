"""Online data normalization."""

import torch
from torch.nn import parameter as tpm
import torch.distributed as dist
import logging


class Normalizer(torch.nn.Module):
    """Feature normalizer that accumulates statistics online (ONLY WITH torch.DDP)."""

    def __init__(
        self,
        size,
        max_accumulations=10**6,
        std_epsilon=1e-8,
        unit=10**6,
        dtype=torch.float64,
        device="cpu",
        name="Normalizer",
    ):
        super(Normalizer, self).__init__()
        self.name = name
        self.unit = unit
        self.size = size
        self.dtype = dtype
        self.synced = False
        self.std_eps = tpm.Parameter(torch.tensor(std_epsilon, dtype=dtype, device=device), requires_grad=False)

        self._max_accumulations = tpm.Parameter(
            torch.tensor(max_accumulations, dtype=dtype, device=device), requires_grad=False
        )
        self._acc_weight = tpm.Parameter(torch.zeros(1, dtype=dtype, device=device), requires_grad=False)
        self._num_accumulations = tpm.Parameter(torch.zeros(1, dtype=dtype, device=device), requires_grad=False)
        self._E_data = tpm.Parameter(torch.zeros(size, dtype=dtype, device=device), requires_grad=False)
        self._E_data_squared = tpm.Parameter(torch.zeros(size, dtype=dtype, device=device), requires_grad=False)
        # self.synchronize(reduceOp=dist.ReduceOp.AVG)
        print("##### NORMALIZER {}: Max Accumulations set to {} ####".format(name, self._max_accumulations), flush=True)

    def forward(self, batched_data, accumulate=False):
        """Normalizes input data and accumulates statistics."""

        if accumulate and (self._num_accumulations < self._max_accumulations):
            self._accumulate(batched_data)
            if self._num_accumulations >= self._max_accumulations:
                logging.warning(
                    "##### NORMALIZER: Max Accumulations {} reached. ####\n\n".format(self._num_accumulations)
                )

        return ((batched_data - self.mean()) / self.std_with_epsilon()).type(
            dtype=torch.float32, non_blocking=True
        )  # cast back to tf32

    def _accumulate(self, batched_data):
        """Function to perform the accumulation of the batch_data statistics."""
        batched_data = batched_data.view(-1, self.size)
        old_weight = self._acc_weight.data
        delta_weight = torch.tensor(batched_data.shape[0] / self.unit).type(dtype=self.dtype, non_blocking=True)
        delta_mean_data = torch.mean(batched_data, dim=0).type(dtype=self.dtype, non_blocking=True)
        delta_mean_squared_data = torch.mean(batched_data**2, dim=0).type(dtype=self.dtype, non_blocking=True)

        self._acc_weight.data = self._acc_weight.data.add(delta_weight)
        self._E_data.data = (
            self._E_data.data.multiply(old_weight).add(delta_mean_data.multiply(delta_weight)).divide(self._acc_weight)
        )
        self._E_data_squared.data = (
            self._E_data_squared.data.multiply(old_weight)
            .add(delta_mean_squared_data.multiply(delta_weight))
            .divide(self._acc_weight)
        )
        self._num_accumulations.data = self._num_accumulations.data.add(1.0)

    def report(self):
        print(
            "##### NORMALIZER {}: Accumulations {} , mean {} , std {} ####".format(
                self.name, self._num_accumulations, self.mean(), self.std_with_epsilon()
            )
        )

    def inverse(self, normalized_batch_data):
        return ((normalized_batch_data * self.std_with_epsilon()) + self.mean()).type(
            dtype=torch.float32, non_blocking=True
        )  # cast back to tf32

    def mean(self):
        return self._E_data

    def std_with_epsilon(self):
        std = torch.sqrt(self._E_data_squared - (self.mean()) ** 2)
        return torch.max(torch.nan_to_num(std), self.std_eps)

    def synchronize(self, reduceOp):
        # Synchronize with all other ranks (only once! Otherwise wrong data)
        print("##### Synchronize!. ####\n\n", flush=True)
        weight = self._acc_weight.data
        E_data = self._E_data.data
        E_data_squared = self._E_data_squared.data
        num_runs = self._num_accumulations.data
        print(f"{weight=} {E_data=} {E_data_squared=} {num_runs=}", flush=True)

        dist.barrier()
        dist.all_reduce(weight, op=reduceOp)
        dist.all_reduce(E_data, op=reduceOp)
        dist.all_reduce(E_data_squared, op=reduceOp)
        dist.all_reduce(num_runs, op=reduceOp)

        print(f"{weight=} {E_data=} {E_data_squared=} {num_runs=}", flush=True)

        self._acc_weight.data = weight
        self._E_data.data = E_data
        self._E_data_squared.data = E_data_squared
        self._num_accumulations.data = num_runs

        self.synced = True
