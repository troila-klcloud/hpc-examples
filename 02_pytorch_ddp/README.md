# Pytorch Distributed Data-Parallel（pytorch-DDP）

在[上个示例](../01_pytorch_dp/README.md)我们介绍了 [pytorch DataParallel] （简称：DP），可以在多个 GPU 上执行训练任务，加快训练速度。但是 pytorch 官方推荐使用 [Pytorch Distributed Data-Parallel] 简称 DDP 替代 [pytorch DataParallel]。

它们的主要区别有两个：

* 在同一个节点跨多个 GPU 执行任务 DP 使用多线程，DDP 使用多进程。因此 DDP 可以避免 GIL 的影响，效率更高。
* DDP 可以支持跨节点，可以使用更多的 GPU。

详情请参考以下文档：

* [Comparison between DataParallel and DistributedDataParallel]
* [Use nn.parallel.DistributedDataParallel instead of multiprocessing or nn.DataParallel]。

本示例将详细讲解如何使用 slurm HPC 平台 提交 pytorch DDP 任务。

## Distributed communication package

上文提到 DDP 是多进程的，进程间通信需要用到通信工具，pythorch 提供了 [Distributed communication package] 来完成这个工作，[Distributed communication package] 提供了三种后端：gloo，mpi，nccl。如何选择后端可以参考官方文档：[Which backend to use?]。本示例我们使用 GPU 训练模型，所以这里要选择 nccl 后端。因此在准备阶段，除了要完成以下几个步骤：

* [安装 conda](../00_get_started/README.md#conda-安装)
* [创建训练环境](../00_get_started/README.md#创建训练环境)
* [下载数据](../00_get_started/README.md#下载数据)

还需要安装 nccl：

    ```
    $ conda install conda-forge::nccl
    ```

## Pytorch Elastic

Pytroch 使用 [elastic] 来管理 pytorch 多节点作业，pytorch [elastic] 提供了容错能力（fault-tolerant）,简化了作业的提交步骤。在提交作业时我们需要用到 [elastic] 提供的 [torchrun] 工具。使用 [torchrun] 会自动设置三个环境变量 `RANK`、 `WORLD_SIZE` 和 `LOCAL_RANK`。

## 代码改动

### 进程组构建和清理

```python
def setup_ddp():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")

def cleanup_ddp():
    dist.destroy_process_group()
```

前面提到 DDP 使用的是多进程，因此需要一个高效的进程间通信机制（尤其是在多节点的情况下）。pytorch 是通过构建进程组的方式来实现的。这里 [init_process_group] 只传入了一个参数 `backend`，在官方文档，或者其它示例你可能会经常看到另外两个参数： `world_size` 和 `RANK`。没有传入这两个参数时 [init_process_group] 会从环境变量中读取（前文提到 torchrun 会自动设置这两个参数）。

### 准备 dataloader

DDP 使用多进程运行训练任务，我们希望每个进程在不同的数据上训练以加快训练。这个工作可以通过 [DistributedSampler] 来完成：

```python
def prepare_dataloader(train_batch_size, test_batch_size):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('data', train=True, download=False,
                       transform=transform)
    dataset2 = datasets.MNIST('data', train=False, download=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(
        dataset1, batch_size=train_batch_size,
        pin_memory=True,
        sampler=DistributedSampler(dataset1))
    test_loader = torch.utils.data.DataLoader(
        dataset2, batch_size=test_batch_size,
        pin_memory=True,
        sampler=DistributedSampler(dataset2))
    return train_loader, test_loader
```

在每轮训练开始前，还需要调用 `set_epoch`。

```python
    for epoch in range(1, args.epochs + 1):
        train_loader.sampler.set_epoch(epoch)
```

### 构建 DDP 模型

```python
    gpu_id = int(os.environ["LOCAL_RANK"])
    model = Net().to(gpu_id)
    model = DDP(model, device_ids=[gpu_id])
```

如果模型包括 [BatchNorm] 层，需要把他转为 [SyncBatchNorm]，pytorch 提供了一个转换工具 [convert_sync_batchnorm]。本示例中使用的 mnist 模式并没有 [BatchNorm] 层，所以不需要这一步。

### 插入同步点

如果你仔细阅读了代码，会发现在 `main()` 函数中，训练完成后会有下面这一行代码：

```python
    dist.barrier()
```

这行代码用于同步所有的分布式进程。现在执行到这一行代码的进程会暂停等待，直到所有进程都执行到这一行代码后，整体在运行接下来的代码。

## 作业提交

### 独占 GPU

```
sbatch --gres=gpu:1 conda.ddp.job
```

在作业脚本文件 `conda.ddp.job` 我们申请了两个节点 (`#SBATCH --nodes=2 `) 运行作业。`--gres=gpu:1` 表示每个节点需要一块 GPU，所以本次一共需要使用到两个节点，两个 GPU。

### 共享 GPU

```
$ sbatch --gres=shard:32 conda.ddp.job
```

[pytorch DataParallel]: https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
[Pytorch Distributed Data-Parallel]: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
[Distributed communication package]: https://pytorch.org/docs/stable/distributed.html
[Which backend to use?]: https://pytorch.org/docs/stable/distributed.html#which-backend-to-use
[elastic]: https://pytorch.org/docs/stable/distributed.elastic.html
[torchrun]: https://pytorch.org/docs/stable/elastic/run.html
[init_process_group]: https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
[DistributedSampler]: https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
[Comparison between DataParallel and DistributedDataParallel]: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#comparison-between-dataparallel-and-distributeddataparallel
[Use nn.parallel.DistributedDataParallel instead of multiprocessing or nn.DataParallel]: https://pytorch.org/docs/stable/notes/cuda.html#cuda-nn-ddp-instead
[BatchNorm]: https://pytorch.org/docs/stable/generated/torch.nn.functional.batch_norm.html
[SyncBatchNorm]: https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html
[convert_sync_batchnorm]: https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html#torch.nn.SyncBatchNorm.convert_sync_batchnorm
[barrier]: https://pytorch.org/tnt/stable/utils/generated/torchtnt.utils.distributed.barrier.html