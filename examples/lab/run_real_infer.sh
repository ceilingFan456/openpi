# cd /home/showlab/VLASafety/openpi-main/examples/lab

# ----------------------------------------------------------------------------

CKPT_CFG="pi05_lab_finetune"
MODEL_NAME="$CKPT_CFG/2999"

# ----------------------------------------------------------------------------
ROOT_PATH="/home/showlab/Users/qiming/openpi/checkpoints"


# "hf://YanzheChen/VLAST/$MODEL_NAME"
# 1
# python main_pi05.py --checkpoint-name $CKPT_CFG --checkpoint-path $ROOT_PATH/$MODEL_NAME --use-mock-cameras --use-mock-robot

# 2 
# python main_pi05.py --checkpoint-name $CKPT_CFG --checkpoint-path $ROOT_PATH/$MODEL_NAME --use-mock-robot --external-camera 327122079691 --wrist-camera 218622273043 --left-camera 317222075319

# pkill -f camera_viewer.py


# 3. real-env
## using front camera
# python /home/showlab/Users/qiming/openpi/examples/lab/main_pi05.py --checkpoint-name $CKPT_CFG --checkpoint-path $ROOT_PATH/$MODEL_NAME --nuc-ip 192.168.1.112 --external-camera 317222075319 --wrist-camera 218622273043 --left-camera 336222073740 --control-mode joint_vel

## using side camera
## --checkpoint-name is my traininig config name. 
python /home/showlab/Users/qiming/openpi/examples/lab/main_pi05.py --checkpoint-name $CKPT_CFG --checkpoint-path $ROOT_PATH/$MODEL_NAME --nuc-ip 192.168.1.112 --external-camera 327122079691 --wrist-camera 218622273043 --left-camera 336222073740 --control-mode joint_vel