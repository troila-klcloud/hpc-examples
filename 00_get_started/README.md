# 快速使用 HPC 平台训练手写数字识别模型

手写数字识别（MNIST）是一个很好的入门卷积神经网络的示例。本教程使用 pytorch 实现了 MNIST 模型，并演示如何通过 slurm HPC 平台提交训练任务。
对于 AI 训练所依赖的运行环境，本示例提供两种方式来创建：conda 和 apptainer (借助 docker image)。

## 前置条件

* 本示例通过 slurm job 的形式提交训练任务。所以 slurm 集群是必不可少的
* 本示例通过两种方式提供训练任务所需的环境 conda 和 apptainer。conda 需要用户自行安装，apptainer 需要集群管理员提前安装好。

## conda

### conda 安装

上文提到本示例会使用 conda 和 apptainer 两种方式来创建作业运行环境。但是为保障顺利运行本仓库中的各个示例，**强烈建议** 不管你喜欢使用那种方式，在使用新用户第一次登录 HPC 平台时，优先安装 conda。

安装参考文档：https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
推荐使用：[Anaconda Distribution](https://www.anaconda.com/download)

### 创建训练环境

通过 conda 我们可以灵活创建专属的训练环境：

    ```
    $ conda create -n pytorch python=3.12
    $ conda activate pytorch
    $ conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
    ```

#### （可选）配置国内下载源

`pytorch` 官方的下载源在国外，有时候下载会很慢，可以配置清华源下载

    ```
    $ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
    $ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
    $ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
    ```

下载：

    ```
    $ conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c nvidia
    ```

**注意：** 上面只是把 pytorch 的下载源改为了国内，CUDA 的下载源依然是 nvidia (暂无找到国内 nvidia cuda 源)

#### （可选）通过 pip 安装依赖包

创建好环境后，也可以通过 pip 安装依赖包：

    ```
    $ conda install pip
    $ pip install xxx
    ```

### 下载数据

在训练时下载数据会造成长时间的资源占用，**强烈推荐**提前下载好数据。本示例准备了训练数据的下载脚本，执行下面命令即可下载数据。

```
$ python download_data.py
```

### 提交作业，使用 CPU 运行

在没用 GPU 或者 GPU 特别反面的环境也可以使用 CPU 运行训练任务。

    ```
    $ sbatch conda.cpu.job
    ```

### 提交作业，独占 GPU 运行

    ```
    $ sbatch --gres=gpu:1 conda.gpu.job
    ```

独占 GPU 可以防止运行过程中与其他作业出现资源竞争，能极大的保障作业的稳定运行。`--gres=gpu:1` 声明使用一个 GPU 运行训练任务。

### 提交作业，共享 GPU 运行

对于计算量不大，不需要太多显存但运行时间长的作业。独占 GPU 的方式是对资源的极大浪费，尤其是资源紧张，有其他作业排队的时候。这时我们可以在提交作业时声明我们作业可以与其他作业共享 GPU 运行。
slurm 支持三种方式共享 GPU：MIG，MPS，Sharding 详情请参考[官方文档](https://slurm.schedmd.com/gres.html)。MIG 需要有专门的硬件支持，MPS 同一时间只能允许一个用户运行作业。因此我们选择
Sharding 模式共享 GPU。Sharding 模式不需要特定 GPU 支持，支持多用户同时运行多个作业在一个 GPU 上，但是 sharding 模式无法限制单个作业在单个 GPU 内使用的资源量，
但是 Sharding 模式可以声明作业使用的资源量（这个声明的资源量只在调度的时候起作用（决定作业在哪个 GPU 上运行），在作业运行期间无法对作业产生影响）。

    ```
    $ sbatch --gres=shard:16 conda.gpu.job
    ```

对于 `--gres=shard:16` 这里不理解没关系，后面会有补充说明。

## apptainer

如前文所说 apptainer 需要管理员提前安装好，安装方式参考：https://apptainer.org/docs/admin/main/installation.html

借助 apptainer 我们可以使用已有的 docker image 快速的创建运行环境。

### 提前准备

为缩短训练时间，提高训练效率。我们需要做一些提前准备工作，这些工作都是一次性的。提前做好这些工作能避免每次占用作业时间（作业在运行时会锁定资源，这些工作如果在作业脚本里完成，会导致资源长时间被一个作业锁定）。

#### 拉取镜像

    ```
    $ apptainer pull docker://registry.cn-hangzhou.aliyuncs.com/troila-klcloud/pytorch:24.01-py3
    ```

#### 转换 dokcer 镜像

    ```
    $ apptainer build pytorch-24.01.sif docker://registry.cn-hangzhou.aliyuncs.com/troila-klcloud/pytorch:24.01-py3
    ```

这一步把 docker 镜像转为 apptainer 镜像，转换完成后会发现当前目录（执行命令的目录）下多了一个 `pytorch-24.01.sif` 文件。

### 提交作业，使用 CPU 运行

    ```
    $ sbatch apptainer.cpu.job
    ```

### 提交作业，独占 GPU 运行

    ```
    $ sbatch --gres=gpu:1 apptainer.gpu.job
    ```

### 提交作业，共享 GPU 运行

    ```
    $ sbatch --gres=shard:16 apptainer.gpu.job
    ```

## sharding 补充

上文提到本示例文档，默认 slurm 时通过 sharding 模式管理 GPU 的。在这一章节，我们对 GPU 的使用做一些补充说明。`Sharding` 翻译成汉语是 `分片` 的意思，`--gres=shard:16` 表示，该作业声明它需要 16 片 GPU 分片来运行。

查看集群 GPU 信息：

```
$ sinfo --all --Node --Format=NodeList,Gres:60,GresUsed:100
NODELIST            GRES                                                        GRES_USED
control118          gpu:Quadro_RTX_4000:2,shard:Quadro_RTX_4000:128             gpu:Quadro_RTX_4000:1(IDX:0),shard:Quadro_RTX_4000:32(0/64,32/64)
control-173         gpu:Quadro_RTX_5000:4,shard:Quadro_RTX_5000:256             gpu:Quadro_RTX_5000:2(IDX:0-1),shard:Quadro_RTX_5000:0(0/64,0/64,0/64,0/64)
```

* 第一列（NODELIST）：节点名称
* 第二列（GRES）：每个节点 GPU 信息

    **解读：** 在节点 control118 上：`Quadro_RTX_4000:2` 有 2 个 RTX 4000 GPU，`shard:Quadro_RTX_4000:128` 这两个 GPU 更分为了 128 shard（片），也就是每个 GPU 分为了 64 片。
    在节点 control-173 上: `Quadro_RTX_5000:4` 有 4 个 RTX 5000 GPU，`shard:Quadro_RTX_5000:256 ` 这四个 GPU 被分为了 256 片，每个 GPU 被分为了 64 片。

*  第三列（GRES_USED）：GPU 使用情况

    **解读：** 在节点 control118 上：`Quadro_RTX_4000:1(IDX:0)` 表示 IDX 为 0 的 GPU （第一快 GPU）被独占使用，`shard:Quadro_RTX_4000:32(0/64,32/64)` 表示该节点第二个 GPU 被占用了 64 片。
    在节点 control-173 上：`Quadro_RTX_5000:2(IDX:0-1)` 表示 IDX 为 0 和 1 的 GPU 被独占使用，`shard:Quadro_RTX_5000:0(0/64,0/64,0/64,0/64)` 没用 GPU 分片被占用。
