#!/bin/bash -l


# basic config

#$ -P ecog-eeg     
#$ -N run_single_unit_opt_test_gpu     # 作业名称
#$ -cwd                       # 在当前目录运行
#$ -o outputs.log             # 标准输出（相当于 > outputs.log）
#$ -e errors.log              # 错误输出
#$ -l h_rt=02:00:00           # 最长运行时间（视任务大小调整）
#$ -m e                      # 在开始和结尾发送电子邮件

# GPU config

#$ -l gpus=1
#$ -l gpu_memory=20G



# 激活 conda 环境
module load miniconda
module load cuda/12.8
conda activate neuralflow

# 执行你的 Python 脚本
python run_optimization_single_unit.py --device GPU --max_epochs 300