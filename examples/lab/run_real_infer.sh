# cd /home/showlab/VLASafety/openpi-main/examples/lab


# ----------------------------------------------------------------------------
## experiment ?
## ???
## observations: 
## 
# CKPT_CFG="pi05_lab_finetune_orange_cube_single_point_dual_external_views"
# MODEL_NAME="$CKPT_CFG/1999"
# ROOT_PATH="/home/eva-01/code/openpi/checkpoints/pi05_lab_finetune_orange_cube_single_point_dual_external_views_02_27"
# ----------------------------------------------------------------------------


############### experiments on views  #########################


# ----------------------------------------------------------------------------
## experiment 7
## comparison with experiment 2 to see if we can only use one view. 
## observations: similar observations as the one with three views. it just goes down? and start opening and close. why does it always go
##              down after it starts. what is it actually learning?
## 
# CKPT_CFG="pi05_lab_finetune_orange_cube_single_point_single_base_view"
# MODEL_NAME="$CKPT_CFG/1999"
# ROOT_PATH="/home/eva-01/code/openpi/checkpoints/pi05_lab_finetune_orange_cube_single_point_single_base_view_02_27"
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
## experiment 6
## comparison with experiment 2 to see whether we can train with two external views.
## observations: this one totally fails as well. it is ignoring the cube. but the actions are regressed somewhat correctly. 
##              this is interesting because it sort of shows, the action expert is overfitting really quickly? it is not learning the 
##              the connections between images and actions correctly. it is just memorising the actions. 
## 
# CKPT_CFG="pi05_lab_finetune_orange_cube_single_point_dual_external_views"
# MODEL_NAME="$CKPT_CFG/1999"
# ROOT_PATH="/home/eva-01/code/openpi/checkpoints/pi05_lab_finetune_orange_cube_single_point_dual_external_views_02_27"
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
## experiment 5
## comparison with experiment 2 to see whether we can train with three views.
## observations: this one totally fails worse than the previous ones. the gripper just goes down vertically without any other movement. 
##              this is a bit strange. since it doesnt seem to be overfitting. the actions are totally wrong. 
##              i think this is mainly because we have trained on a new view so need more training. 
## 
# CKPT_CFG="pi05_lab_finetune_orange_cube_single_point_three_views"
# MODEL_NAME="$CKPT_CFG/1999"
# ROOT_PATH="/home/eva-01/code/openpi/checkpoints/pi05_lab_finetune_orange_cube_single_point_three_views_02_27"
# ----------------------------------------------------------------------------






############### experiments on num of demonstrations #########################


# ----------------------------------------------------------------------------
## experiment 4
## comparison with experiment 2 to see how much training demonstrations we need.  
## observations: can grasp correctly. but placement still have some issue. not so extreme to the point where it places at the corner. 
##              but doesnt place it onto the centre of the coaster still. 
##
# CKPT_CFG="pi05_lab_finetune_orange_cube_single_point_15"
# MODEL_NAME="$CKPT_CFG/1999"
# ROOT_PATH="/home/eva-01/code/openpi/checkpoints/pi05_lab_finetune_orange_cube_single_point_15_02_27"
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
## experiment 3
## comparison with experiment 2 to see how much training demonstrations we need.  
## observations: can grasp not so correctly. will miss on the first try and sometimes succeed on the second try. 
##              the placement is completely bad. can will place the cube at the far corner of the table near the robot side. 
## 
# CKPT_CFG="pi05_lab_finetune_orange_cube_single_point_10"
# MODEL_NAME="$CKPT_CFG/1999"
# ROOT_PATH="/home/eva-01/code/openpi/checkpoints/pi05_lab_finetune_orange_cube_single_point_10_02_27"
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
## experiment 2
## retrain of the original dataset with the new code for sanity chceks.
# CKPT_CFG="pi05_lab_finetune_orange_cube_single_point"
# MODEL_NAME="$CKPT_CFG/1999"
# ROOT_PATH="/home/eva-01/code/openpi/checkpoints/pi05_lab_finetune_orange_cube_single_point_02_27"
# ----------------------------------------------------------------------------







############### some baseline experiments #########################


# ----------------------------------------------------------------------------
## experiment 1
## this one has been used as a snaity check baseline
## orange cube baseline dataset with 25 episodes, it uses front camera
## this one only has two views
CKPT_CFG="pi05_lab_finetune_orange_cube_single_point"
MODEL_NAME="$CKPT_CFG/1999"
ROOT_PATH="/home/eva-01/code/openpi/checkpoints/baselines/lab_training_orange_cube_single_point"
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
## experiment 0 
## test tube moving with 40 episodes.
## it uses side camera
# CKPT_CFG="pi05_lab_finetune"
# MODEL_NAME="$CKPT_CFG/2999"
# ROOT_PATH="/home/eva-01/code/openpi/checkpoints/baselines/pi05-lab-finetune-2999"
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