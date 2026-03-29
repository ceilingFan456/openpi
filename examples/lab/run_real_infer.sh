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











############### experiments on with three views using yanzhe build-block dataset  #########################


# ----------------------------------------------------------------------------
## experiment 19
## experiment on 3k -30k steps. but the inference environment is different from training setup. training is white background and white platform. testing if the new full black env. 
## observations:  3k doesnt work. it is grabbing in front. 12k is really bad, it will stuck and stops moving. 18k is working for one corner.  21k is not working well. it has an offset to the left. 24k has the correct trend but it doenst
##              know how to close the gripper. this is really interesting. 
## 
CKPT_CFG="pi05_yanzhe_grid_5_three_views_30k_steps"
MODEL_NAME="$CKPT_CFG/21000"
ROOT_PATH="/home/eva-01/code/openpi/checkpoints/pi05_yanzhe_grid_5_three_views_30k_steps"
# ----------------------------------------------------------------------------

























############### experiments on co-training with 2d auxilliary loss  #########################





# ----------------------------------------------------------------------------
## experiment 18
## 25-25 training on with no weight on aux-loss, this is the trained on the original dataset 
## observations: it is good. can do the job well other than gripper lose connection at the end. meaning the training pipeline has no issue. 
## 
# CKPT_CFG="pi05_aux2d_co_training_baseline_orange_cube_paired_25"
# MODEL_NAME="$CKPT_CFG/12000"
# ROOT_PATH="/home/eva-01/code/openpi/checkpoints/pi05_aux2d_co_training_baseline_orange_cube_paired_25_12000"
# ----------------------------------------------------------------------------





# ----------------------------------------------------------------------------
## experiment 17
## 25-25 training on with no weight on aux-loss, this is the trained on the original dataset 
## observations: it is good. can do the job well other than gripper lose connection at the end. meaning the training pipeline has no issue. 
## 
# CKPT_CFG="pi05_aux2d_co_training_orange_cube_paired_25"
# MODEL_NAME="$CKPT_CFG/9000"
# ROOT_PATH="/home/eva-01/code/openpi/checkpoints/pi05_aux2d_co_training_orange_cube_paired_25_9000"
# ----------------------------------------------------------------------------





# ----------------------------------------------------------------------------
## experiment 17
## 25-25 training on with no weight on aux-loss, this is the trained on the original dataset 
## observations: it is good. can do the job well other than gripper lose connection at the end. meaning the training pipeline has no issue. 
## 
# CKPT_CFG="pi05_aux2d_co_training_baseline_orange_cube_paired_25"
# MODEL_NAME="$CKPT_CFG/9000"
# ROOT_PATH="/home/eva-01/code/openpi/checkpoints/pi05_aux2d_co_training_baseline_orange_cube_paired_25_9000"
# ----------------------------------------------------------------------------





# ----------------------------------------------------------------------------
## experiment 16
## 30-30 training on with no weight on aux-loss
## observations: 
## 
# CKPT_CFG="pi05_aux2d_co_training_baseline"
# MODEL_NAME="$CKPT_CFG/18000"
# ROOT_PATH="/home/eva-01/code/openpi/checkpoints/pi05_aux2d_co_training_baseline_18000"
# ----------------------------------------------------------------------------






# ----------------------------------------------------------------------------
## experiment 15
## 30-30 euqal weight training on aux-loss and policy-loss
## observations: not working very well. hope new baseline works
## 
# CKPT_CFG="pi05_aux2d_co_training"
# MODEL_NAME="$CKPT_CFG/7000"
# ROOT_PATH="/home/eva-01/code/openpi/checkpoints/pi05_aux2d_co_training_7000"
# ----------------------------------------------------------------------------






############### experiments on longer training  #########################



# ----------------------------------------------------------------------------
## experiment 14
## try running single views for longer time.
## observations: 9k worked for one time, in general it cannot grasp correctly because of the bad depth estimation. since only trained on front view. 
##              export HYDRA_FULL_ERROR=1 somehow made the performance worse by clamping in midair. need to double check the reason. 
## 
# CKPT_CFG="pi05_lab_finetune_orange_cube_single_point_single_base_view_15k_steps"
# MODEL_NAME="$CKPT_CFG/9000"
# ROOT_PATH="/home/eva-01/code/openpi/checkpoints/pi05_lab_finetune_orange_cube_single_point_single_base_view_15k_steps_02"
# ----------------------------------------------------------------------------





# ----------------------------------------------------------------------------
## experiment 13
## try running single views for longer time.
## observations: until 6k, it doesnt work. i will train longer to see the result. 
## 
# CKPT_CFG="pi05_lab_finetune_orange_cube_single_point_single_base_view_15k_steps"
# MODEL_NAME="$CKPT_CFG/7000"
# ROOT_PATH="/home/eva-01/code/openpi/checkpoints/pi05_lab_finetune_orange_cube_single_point_single_base_view_15k_steps"
# ----------------------------------------------------------------------------





# ----------------------------------------------------------------------------
## experiment 12
## try running two external views for longer time.
## observations: i think 3k steps are already working. this is really interesting since we can use two external views to train.
##              6k and 9k are working fine too. this is pretty interesting. since we dont have wrist review. 
## 
# CKPT_CFG="pi05_lab_finetune_orange_cube_single_point_dual_external_views_15k_steps"
# MODEL_NAME="$CKPT_CFG/9000"
# ROOT_PATH="/home/eva-01/code/openpi/checkpoints/pi05_lab_finetune_orange_cube_single_point_dual_external_views_15k_steps"
# ----------------------------------------------------------------------------




# ----------------------------------------------------------------------------
## experiment 11
## try running two external views for longer time.
## observations: i think 3k steps are already working. this is really interesting since we can use two external views to train.
##              6k and 9k are working fine too. this is pretty interesting. since we dont have wrist review. 
## 
# CKPT_CFG="pi05_lab_finetune_orange_cube_single_point_dual_external_views_15k_steps"
# MODEL_NAME="$CKPT_CFG/9000"
# ROOT_PATH="/home/eva-01/code/openpi/checkpoints/pi05_lab_finetune_orange_cube_single_point_dual_external_views_15k_steps"
# ----------------------------------------------------------------------------




# ----------------------------------------------------------------------------
## experiment 10
## try running three views for longer time.
## observations: using the 1,2 setting for cameras inside main_pi0.5, 3k steps is aleady working. the 2k version is not working. this is really interesting. 
##              3k is really good. 9k is already a bit trash. cannot finish the task correctly quite frequently. 
##              it seems the reason is the black colour patch on the cube?
## 
# CKPT_CFG="pi05_lab_finetune_orange_cube_single_point_three_views_10k_steps"
# MODEL_NAME="$CKPT_CFG/11999"
# ROOT_PATH="/home/eva-01/code/openpi/checkpoints/pi05_lab_finetune_orange_cube_single_point_three_views_10k_steps"
# ----------------------------------------------------------------------------



############### experiments on 0 shot  #########################


# ----------------------------------------------------------------------------
## experiment 8
## trying to zero shot on pi0.5_base
## observations: really bad and without any trend to do the task.
## 
# CKPT_CFG="pi05_lab_finetune_orange_cube_single_point_dual_external_views"
# MODEL_NAME="$CKPT_CFG/1999"
# ROOT_PATH="/home/eva-01/code/openpi/checkpoints/pi05_lab_finetune_orange_cube_single_point_dual_external_views_02_27"

# python /home/eva-01/code/openpi/examples/lab/main_pi05.py --checkpoint-name "pi05_lab_finetune_orange_cube_single_point" --checkpoint-path "/home/eva-01/code/openpi/checkpoints/pi05_base" --nuc-ip 192.168.1.112 --external-camera 317222075319 --wrist-camera 218622273043 --left-camera 336222073740 --control-mode joint_vel
# ----------------------------------------------------------------------------



# ----------------------------------------------------------------------------
## experiment 9
## trying to zero shot on pi0.5_base
## observations: really bad but with trend to do the correct task.
## 
# CKPT_CFG="pi05_lab_finetune_orange_cube_single_point_dual_external_views"
# MODEL_NAME="$CKPT_CFG/1999"
# ROOT_PATH="/home/eva-01/code/openpi/checkpoints/pi05_lab_finetune_orange_cube_single_point_dual_external_views_02_27"

# python /home/eva-01/code/openpi/examples/lab/main_pi05.py --checkpoint-name "pi05_lab_finetune_orange_cube_single_point" --checkpoint-path "/home/eva-01/code/openpi/checkpoints/pi05_droid" --nuc-ip 192.168.1.112 --external-camera 317222075319 --wrist-camera 218622273043 --left-camera 336222073740 --control-mode joint_vel
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
# CKPT_CFG="pi05_lab_finetune_orange_cube_single_point"
# MODEL_NAME="$CKPT_CFG/1999"
# ROOT_PATH="/home/eva-01/code/openpi/checkpoints/baselines/lab_training_orange_cube_single_point"
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