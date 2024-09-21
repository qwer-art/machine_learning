### 配置一个拥有torch的conda环境
1. conda create -n cpu_torch python=3.9
2.  conda activate cpu_torch
3.  conda env list
4.  conda install pytorch torchvision torchaudio cpuonly -c pytorch
5. conda list 
   1. ubuntu: grep torch 
   2. window: findstr torch
### 在pycharm中使用上conda环境
1. Python Interpreter
2. D:\ProgramData\Anaconda3\envs\cpu_torch\python.exe
### torch重点需要学习的部分
#### meshgrid
#### view
#### reshape
#### 升维降维
#### torch.gather(input, dim, index)
### git
1. window配置
   1. ssh-keygen -t rsa -C "1062428799@qq.com"
   2. C:\Users\10624/.ssh/id_rsa
   3. C: 
   4. dir
2. git remote
   1. git remote -v
   2. git remote set-url origin git@github.com:qwer-art/machine_learning.git