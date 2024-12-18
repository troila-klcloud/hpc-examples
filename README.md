随着 AI 技术的发展，越来越多的学校，机构，个人加入 AI 研发大军。但是对于 AI 模型的训练存在高成本，高复杂度等问题。为了提高资源利用率，简化模型训练，通常会借助一个 HPC 平台来提交 AI 训练任务。在众多 HPC 平台的搭建方案中 [slurm] 无疑是一个最简单，高效，经典的方案。

> 但是即使有了 HPC 平台的加持，如何合理，高效，稳定的运行 AI 训练任务依然不是一个简单的话题。本仓库提供了数个示例，说明如何基于 slurm 平台，根据不同需求合理的提交训练任务。

> **提示**
> 本仓库的目标人群是 AI 技术的研究人员，需要具备一定的 AI 基础知识，能熟练使用 AI 框架（本仓库绝大部分的演示示例是基于 pytorch 编写的）。本仓库不对 slurm 的安装与使用做具体讲解。对于 slurm 的使用可参考下面文档：

* https://hpc.pku.edu.cn/ug/guide/slurm/
* https://docs.hpc.sjtu.edu.cn/job/slurm.html
* https://hpc.sicau.edu.cn/syzn/slurm.htm
* https://www.arch.jhu.edu/short-tutorial-how-to-create-a-slurm-script/

## 目录

#### [00_get_started](./00_get_started/README.md)

该示例使用 pytorch  实现了一个手写数字识别模型（mnist）,借助该模型演示了 slurm 平台的基本用法，包括两种训练环境创建方法：

* conda 灵活，强大。可更具需求快速灵活的修改环境
* docker image 简单，高效。可直接使用其他人构建好的镜像作为运行环境
  
三种作业运行方式：

* CPU 低成本运行训练任务
* 共享 GPU 多个作业运行在一个 GPU 上（作业之间可能会有竞争，导致作业运行失败）
* 独占 GPU 一个作业独占一个 GPU

#### [01_pytorch_dp](./01_pytorch_dp/README.md)

本示例展示了如何在一个节点上使用多块 GPU 上训练 AI 模型（借助 [pytorch Data Parallelism](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)）。

#### [02_pytorch_ddp](./02_pytorch_ddp/README.md)

本示例展示一种更高效的方式（[pytorch DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)（简称：pytorch ddp））使用多 GPU。本示例参考以下链接完成：

* https://pytorch.org/tutorials/beginner/dist_overview.html
* https://pytorch.org/tutorials/distributed/home.html
* https://pytorch.org/tutorials/beginner/ddp_series_theory.html
* https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
* https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series


[slurm]: https://slurm.schedmd.com/documentation.html