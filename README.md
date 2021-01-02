# python版本要求
python3
# 环境需求
gym

pip install gym #可以在命令行中通过此命令安装pip

numpy

pip install numpy #可以在命令行中通过此命令安装pip
# Q-learning
针对最经典的表格型Q learning算法进行了复现，能够支持gym中大多数的离散动作和状态空间的环境，譬如CliffWalking-v0。
以悬崖寻路（CliffWalking-v0）为例，测试结果为

epoch: 998, avg_return: -13.0

o  o  o  o  o  o  o  o  o  o  o  o

o  o  o  o  o  o  o  o  o  o  o  o

o  o  o  o  o  o  o  o  o  o  o  o

o  C  C  C  C  C  C  C  C  C  C  x
