#!/bin/bash
CASE="${1-cylinder}"
CONFIG_FILE="${2-./configs/$CASE/}"
# 0: train 1: local test 2: global
MODE="${3-0}"
RESTART_EPOCH="${4--1}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "No config file for ${CASE} in configs folder"
    exit 128
fi
source $CONFIG_FILE

run(){
    python3 src/main.py \
    -case $CASE -space_dim $space_dim \
    -n_train $n_train -n_valid $n_valid -n_test $n_test -time_len $time_len\
    -noise_level $noise_level \
    -multi_mesh_layer $multi_mesh_layer -consist_mesh $consist_mesh\
    -num_epochs $num_epochs -batch $batch -lr $lr -gamma $gamma \
    -restart_epoch $RESTART_EPOCH \
    -mp_time $MP_time\
    -data_dir $data_dir -dump_dir $dump_dir -mode $MODE
}

run