import subprocess
import os
import hydra
from omegaconf import DictConfig


def download_dataset(cfg: DictConfig):
    """
    Call the shell script to download the dataset.

    Parameters:
    - cfg: Hydra configuration object containing dataset configuration.

    Returns:
    - output: Output from the shell script.
    - error: Error message, if any.
    """
    script_path = os.path.join(os.path.dirname(__file__), "download_dataset.sh")
    data_set_name = cfg.datasets.tf_dataset_name
    output_dir = cfg.datasets.tf_dataset_dir

    try:
        result = subprocess.run([script_path, data_set_name, output_dir])
        return result.stdout, None
    except subprocess.CalledProcessError as e:
        return None, e.stderr


@hydra.main(version_base=None, config_path="../configs/", config_name="default")
def main(cfg: DictConfig):
    # Call the download_dataset function with the Hydra configuration
    output, error = download_dataset(cfg)

    if error:
        print("Error:", error)
    else:
        print("Output:", output)


if __name__ == "__main__":
    main()
