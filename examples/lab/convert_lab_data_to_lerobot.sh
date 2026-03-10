#!/bin/bash

## run this in the root directory of the repo /openpi
python examples/lab/convert_lab_data_to_lerobot.py --data-dir "dummy for now" --push-to-hub

## saved at below and uses 
## RAW_DATASET_DIR_PATH = "/home/t-qimhuang/disk2/labdata_test"
# (openpi) t-qimhuang@microsoft.com@GCRAZGDL1194:~/code/openpi$ bash examples/lab/convert_lab_data_lerobot.sh 
# Converting raw data from dummy for now to LeRobot dataset at /home/t-qimhuang/.cache/huggingface/lerobot/ceilingfan456/lab_data_test