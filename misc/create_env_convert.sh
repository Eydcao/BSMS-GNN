# A no brain installation script for the environment
# for converting the MGN tensorflow record to h5

conda create -n convert_tf python=3.7
conda activate convert_tf

pip install tensorflow==1.15.5
pip install --force-reinstall protobuf==3.20
pip install h5py
pip install hydra-core