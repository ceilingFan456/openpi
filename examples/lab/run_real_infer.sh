# cd /home/showlab/VLASafety/openpi-main/examples/lab




############### experiments on views  #########################


# ----------------------------------------------------------------------------
## experiment 6
## comparison with experiment 2 to see whether we can train with three views.
# CKPT_CFG="pi05_lab_finetune_orange_cube_single_point_dual_external_views"
# MODEL_NAME="$CKPT_CFG/1999"
# ROOT_PATH="/home/eva-01/code/openpi/checkpoints/pi05_lab_finetune_orange_cube_single_point_dual_external_views_02_27"
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
## experiment 5
## comparison with experiment 2 to see whether we can train with three views.
# CKPT_CFG="pi05_lab_finetune_orange_cube_single_point_three_views"
# MODEL_NAME="$CKPT_CFG/1999"
# ROOT_PATH="/home/eva-01/code/openpi/checkpoints/pi05_lab_finetune_orange_cube_single_point_three_views_02_27"
# ----------------------------------------------------------------------------






############### experiments on num of demonstrations #########################


# ----------------------------------------------------------------------------
## experiment 4
## comparison with experiment 2 to see how much training demonstrations we need.  
# CKPT_CFG="pi05_lab_finetune_orange_cube_single_point_15"
# MODEL_NAME="$CKPT_CFG/1999"
# ROOT_PATH="/home/eva-01/code/openpi/checkpoints/pi05_lab_finetune_orange_cube_single_point_15_02_27"
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
## experiment 3
## comparison with experiment 2 to see how much training demonstrations we need.  
# CKPT_CFG="pi05_lab_finetune_orange_cube_single_point_10"
# MODEL_NAME="$CKPT_CFG/1999"
# ROOT_PATH="/home/eva-01/code/openpi/checkpoints/pi05_lab_finetune_orange_cube_single_point_10_02_27"
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
## experiment 2
## retrain of the original dataset with the new code for sanity chceks.
CKPT_CFG="pi05_lab_finetune_orange_cube_single_point"
MODEL_NAME="$CKPT_CFG/1999"
ROOT_PATH="/home/eva-01/code/openpi/checkpoints/pi05_lab_finetune_orange_cube_single_point_02_27"
# ----------------------------------------------------------------------------







############### some baseline experiments #########################


# ----------------------------------------------------------------------------
## experiment 1
## this one has been used as a snaity check baseline
## orange cube baseline dataset with 25 episodes, it uses front camera
## this one only has two views
# CKPT_CFG="pi05_lab_finetune_orange_cube_single_point"
# MODEL_NAME="$CKPT_CFG/1999"
# ROOT_PATH="/home/eva-01/code/openpi/checkpoints/lab_training_orange_cube_single_point"
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
## experiment 0 
## test tube moving with 40 episodes.
## it uses side camera
# CKPT_CFG="pi05_lab_finetune"
# MODEL_NAME="$CKPT_CFG/2999"
# ROOT_PATH="/home/eva-01/code/openpi/checkpoints/pi05-lab-finetune-2999"
# ----------------------------------------------------------------------------



# "hf://YanzheChen/VLAST/$MODEL_NAME"
# 1
# python main_pi05.py --checkpoint-name $CKPT_CFG --checkpoint-path $ROOT_PATH/$MODEL_NAME --use-mock-cameras --use-mock-robot

# 2 
# python main_pi05.py --checkpoint-name $CKPT_CFG --checkpoint-path $ROOT_PATH/$MODEL_NAME --use-mock-robot --external-camera 327122079691 --wrist-camera 218622273043 --left-camera 317222075319

# pkill -f camera_viewer.py


# 3. real-env
## using front camera
python /home/eva-01/code/openpi/examples/lab/main_pi05.py --checkpoint-name $CKPT_CFG --checkpoint-path $ROOT_PATH/$MODEL_NAME --nuc-ip 192.168.1.112 --external-camera 317222075319 --wrist-camera 218622273043 --left-camera 336222073740 --control-mode joint_vel

## using side camera
## --checkpoint-name is my traininig config name. 
# python /home/eva-01/code/openpi/examples/lab/main_pi05.py --checkpoint-name $CKPT_CFG --checkpoint-path $ROOT_PATH/$MODEL_NAME --nuc-ip 192.168.1.112 --external-camera 327122079691 --wrist-camera 218622273043 --left-camera 336222073740 --control-mode joint_vel