#!/bin/bash

# -----------------------------
# 设置 sudo 密码和 Python 环境
# -----------------------------
PASSWD=Showlab123
export CONDA_PREFIX=/home/franka/miniforge3/envs/polymetis-local

# -----------------------------
# kill all existing run_server processes
# -----------------------------
echo "$PASSWD" | sudo pkill -9 run_server

# -----------------------------
# 用于保存后台进程 PID
# -----------------------------
PIDS=()

# -----------------------------
# 启动后台程序
# -----------------------------
echo "$PASSWD" | sudo -S -E nohup $CONDA_PREFIX/bin/python3 /home/franka/fairo/polymetis/polymetis/python/scripts/launch_robot.py robot_client=franka_hardware > follower.log 2>&1 &
PIDS+=($!)

echo "$PASSWD" | sudo -S -E nohup $CONDA_PREFIX/bin/python3 /home/franka/fairo/polymetis/polymetis/python/scripts/launch_robot.py robot_client=franka_hardware_leader port=50054 > leader.log 2>&1 &
PIDS+=($!)

echo "$PASSWD" | sudo -S -E nohup $CONDA_PREFIX/bin/python3 /home/franka/fairo/polymetis/polymetis/python/scripts/launch_gripper.py gripper=franka_hand > gripper.log 2>&1 &
PIDS+=($!)

# -----------------------------
# 等待后台服务启动
# -----------------------------
echo "Waiting 1s for servers to start..."
sleep 1s

# -----------------------------
# 前台启动 bimanual_mirror_server.py
# -----------------------------
cd /home/franka/franka_server
$CONDA_PREFIX/bin/python3 bimanual_mirror_server.py
