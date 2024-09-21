

### torch
#### 从0实现torch的layer
##### 卷积层
##### batch_norm
##### layer_norm
##### softmax层
##### 节点图
1. 一分多
2. 多分一

#### 关键函数
##### meshgrid
1. 和矩阵的轴是一样的
   1. 纵轴为x 
   2. 横轴为y
2. vec_aa(n,),vec_bb(m,)
   1. xx,yy 都为 (n,m)
      1. xx: vec_aa横向扩展m个
      2. yy: vec_bb纵向扩展n个
##### view
##### reshape
##### 升维降维
##### torch.gather(input, dim, index)

### 环境
#### conda: 配置一个拥有torch的conda环境
1. conda create -n cpu_torch python=3.9
2.  conda activate cpu_torch
3.  conda env list
4.  conda install pytorch torchvision torchaudio cpuonly -c pytorch
5. conda list 
   1. ubuntu: grep torch 
   2. window: findstr torch
#### pycharm: 在pycharm中使用上conda环境
1. Python Interpreter
2. D:\ProgramData\Anaconda3\envs\cpu_torch\python.exe
#### git
1. window配置
   1. ssh-keygen -t rsa -C "1062428799@qq.com"
   2. C:\Users\10624/.ssh/id_rsa
   3. C: 
   4. dir
2. git remote
   1. git remote -v
   2. git remote set-url origin git@github.com:qwer-art/machine_learning.git