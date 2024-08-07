import tensorflow as tf
import functools
import json
import os
import numpy as np
import h5py
import hydra
from omegaconf import DictConfig
import tensorflow.compat.v1 as tf1

# Enable eager execution
tf1.enable_eager_execution()


def _parse(proto, meta):
    """Parses a trajectory from tf.Example."""
    feature_lists = {k: tf.io.VarLenFeature(tf.string) for k in meta["field_names"]}
    features = tf.io.parse_single_example(proto, feature_lists)
    out = {}
    for key, field in meta["features"].items():
        data = tf.io.decode_raw(features[key].values, getattr(tf, field["dtype"]))
        data = tf.reshape(data, field["shape"])
        if field["type"] == "static":
            data = tf.tile(data, [meta["trajectory_length"], 1, 1])
        elif field["type"] == "dynamic_varlen":
            length = tf.io.decode_raw(features["length_" + key].values, tf.int32)
            length = tf.reshape(length, [-1])
            data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
        elif field["type"] != "dynamic":
            raise ValueError("invalid data format")
        out[key] = data
    return out


def load_dataset(path, split):
    """Load dataset."""
    with open(os.path.join(path, "meta.json"), "r") as fp:
        meta = json.loads(fp.read())
    ds = tf.data.TFRecordDataset(os.path.join(path, split + ".tfrecord"))
    ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
    ds = ds.prefetch(1)
    return ds


def convert_tfrecord_to_h5(tf_dataset_path, save_root, data_keys):
    """Convert TFRecord dataset to H5 files."""
    os.makedirs(save_root, exist_ok=True)
    for split in ["train", "test", "valid"]:
        ds = load_dataset(tf_dataset_path, split)
        split_dir = os.path.join(save_root, split)
        os.makedirs(split_dir, exist_ok=True)
        for index, d in enumerate(ds):
            try:
                data = {key: d[key].numpy() for key in data_keys}
                save_path = os.path.join(split_dir, f"{index}.h5")
                with h5py.File(save_path, "w") as f:
                    for key, value in data.items():
                        f.create_dataset(key, data=value)
                print(f"Success in index: {index}, saved to {split_dir}/{index}.h5")
            except Exception as e:
                print(f"Skipped error in index: {index}, error: {e}")
                continue


@hydra.main(version_base=None, config_path="../configs/", config_name="default")
def main(cfg: DictConfig):
    tf_dataset_path = os.path.join(cfg.datasets.tf_dataset_dir, cfg.datasets.tf_dataset_name)
    save_root = os.path.join(cfg.datasets.root, cfg.datasets.tf_dataset_name)
    data_keys = cfg.datasets.field_names

    convert_tfrecord_to_h5(tf_dataset_path, save_root, data_keys)


if __name__ == "__main__":
    main()
