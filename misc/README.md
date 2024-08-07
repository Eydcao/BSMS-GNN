
# Download TensorFlow Dataset

To download the TensorFlow dataset from MeshGrahNet, use the `download_dataset.sh` script. Modify the paths for `tf_dataset_name` and `tf_dataset` in the corresponding config file. For example, for the airfoil dataset, modify `./configs/datasets/airfoil.yaml` to specify the dataset name and output directory. Then, run the following command to download the dataset:

```bash
python misc/download_dataset.py datasets=airfoil
```

# Parse TensorFlow Dataset into H5 Files

To convert your TensorFlow dataset (`.tfrecord`) into HDF5 files (`.h5`), first use the `create_env_convert.sh` script to create a virtual environment for this purpose. Modify the paths for `tf_dataset` and `root` in the corresponding config. For example, for the airfoil dataset, modify `./configs/datasets/airfoil.yaml` to point to your TensorFlow dataset directory and desired output directory. Then, run the following command:

```bash
python misc/parse_tfrecord.py datasets=airfoil
```