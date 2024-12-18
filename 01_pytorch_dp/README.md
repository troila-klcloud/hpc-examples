# pytorch Data Parallelism （单节点多 GPU）

本示例展示如何使用单节点多 GPU 训练 mnist 模型，主要借助了 [pytorch Data Parallelism](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)机制。

## 代码改动

本示例中的 mnist.py 代码基于 [00_get_started 的代码](../00_get_started/mnist.py) 做了微小改动。

* 删除了下面代码

```python
    if use_cuda:
        print("using cuda")
        device = torch.device("cuda")
    elif use_mps:
        print("using mps")
        device = torch.device("mps")
    else:
        print("using cpu")
        device = torch.device("cpu")
```

在多 GPU 环境，我们只需要考虑 CUDA。

* 添加了以下代码

```python
    model = Net()
    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
```

**注意：** 与原代码的区别，原代码加载模型只需要一行：

```python
    model = Net().to(device)
```

## 作业脚本改动

本示例的作业脚本为：apptainer.db.job 和 conda.dp.job，相对于 [00_get_started 的作业脚本](../00_get_started/conda.gpu.job)，只添加了一行代码：

```bash
export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS
```

* SLURM_JOB_GPUS 参考[官方文档](https://slurm.schedmd.com/sbatch.html#OPT_SLURM_JOB_GPUS)，slurm 通过这个变量告诉作业该使用哪些 GPU。
* CUDA_VISIBLE_DEVICES 通过这个变量，告诉 pytorch 程序该使用哪些 GPU

## 准备

参考 [00_get_started](../00_get_started/README.md)：
[安装 conda](../00_get_started/README.md#conda-安装)，
[创建运行环境](../00_get_started/README.md#创建训练环境)，
[下载数据](../00_get_started/README.md#下载数据)

## 提交作业

### conda

    ```
    $ sbatch --gres=gpu:2 conda.dp.job
    ```

`--gres=gpu:2` 表示独占两个 GPU 执行训练任务。

**注意：** 在 [00_get_started](../00_get_started/README.md#提交作业共享-gpu-运行) 中提到：多个作业可以共享 GPU。但是 slurm 在同节点使用多个 GPU 时必须独占 GPU。也就是说我们没法通过`--gres=shard:128` 使作业使用多个 GPU（如果你这么做了，这个作业会一直是 pending）。

### apptiner

    ```
    $ sbatch --gres=gpu:2 apptainer.dp.job
    ```