# 目录

<!-- TOC -->

- [目录](#目录)
- [ConvMixer描述](#ConvMixer描述)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
- [训练和测试](#训练和测试)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [ImageNet-1k上的ConvMixer训练](#ImageNet-1k上的ConvMixer训练)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# [ConvMixer描述](#目录)

ViT（Vision Transformer）等视觉模型的强大性能，是来自于 Transformer，还是被忽略的 patch？有研究者提出了简单 ConvMixer 模型进行证明，直接将 patch 作为输入，实验表明，ConvMixer 性能优于 ResNet 等经典视觉模型，并且在类似的参数计数和数据集大小方面也优于 ViT、MLP-Mixer 及其一些变体。
近年来，深度学习系统中的卷积神经网络在处理计算机视觉任务中，一直占据主要地位。但最近，基于 Transformer 模型的架构，例如 ViT（Vision Transformer）架构（Dosovitskiy 等人，2020 年），在许多任务中都表现出了引人注目的性能，它们通常优于经典卷积网络，尤其是在大型数据集上表现更佳。
我们可以假设，Transformer 成为视觉领域的主导架构只是时间问题，就像它们在 NLP 领域中一样。然而，为了将 Transformer 应用于图像领域，信息的表示方法必须改变：因为如果在每像素级别上应用 Transformer 中的自注意力层，它的计算成本将与每张图像的像素数成二次方扩展，所以折衷的方法是首先将图像分成多个 patch，再将这些 patch 线性嵌入 ，最后将 transformer 直接应用于此 patch 集合。
在本文中，研究者为后者提供了一些证据：具体而言，该研究提出了 ConvMixer，这是一个极其简单的模型，在思想上与 ViT 和更基本的 MLP-Mixer 相似，这些模型直接将 patch 作为输入进行操作，分离空间和通道维度的混合，并在整个网络中保持相同的大小和分辨率。然而，相比之下，该研究提出的 ConvMixer 仅使用标准卷积来实现混合步骤。尽管它很简单，但研究表明，除了优于 ResNet 等经典视觉模型之外，ConvMixer 在类似的参数计数和数据集大小方面也优于 ViT、MLP-Mixer 及其一些变体。


# [数据集](#目录)

使用的数据集：[ImageNet2012](http://www.image-net.org/)

- 数据集大小：共1000个类、224*224彩色图像
    - 训练集：共1,281,167张图像
    - 测试集：共50,000张图像
- 数据格式：JPEG
    - 注：数据在dataset.py中处理。
- 下载数据集，目录结构如下：

 ```text
└─imagenet
    ├─train                 # 训练数据集
    └─val                   # 评估数据集
 ```

# [特性](#目录)

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.9/others/mixed_precision.html?highlight=%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6)
的训练方法，使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。

# [环境要求](#目录)

- 硬件（Ascend）
    - 使用Ascend来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# [脚本说明](#目录)        │   

## 脚本及样例代码

```text
    └── ConvMixer
        ├── eval.py
        ├── README.md
        ├── requriments.txt
        ├── scripts
        │   ├── run_distribute_train_ascend.sh
        │   ├── run_eval_ascend.sh
        │   ├── run_infer_310.sh
        │   └── run_standalone_train_ascend.sh
        ├── src
        │   ├── args.py
        │   ├── configs
        │   │   ├── parser.py
        │   │   └── convmixer_1536_20.yaml
        │   ├── data
        │   │   ├── augment
        │   │   │   ├── auto_augment.py
        │   │   │   ├── __init__.py
        │   │   │   ├── mixup.py
        │   │   │   ├── random_erasing.py
        │   │   │   └── transforms.py
        │   │   ├── data_utils
        │   │   │   ├── __init__.py
        │   │   │   └── moxing_adapter.py
        │   │   ├── imagenet.py
        │   │   └── __init__.py
        │   ├── models
        │   │   ├── __init__.py
        │   │   ├── layers
        │   │   │   ├── drop_path.py
        │   │   │   └── identity.py
        │   │   └── convmixer.py
        │   ├── tools
        │   │   ├── callback.py
        │   │   ├── cell.py
        │   │   ├── criterion.py
        │   │   ├── get_misc.py
        │   │   ├── __init__.py
        │   │   ├── optimizer.py
        │   │   ├── schedulers.py
        │   │   └── var_init.py
        │   └── trainer
        │       └── train_one_step.py
        └── train.py
```

## 脚本参数

在convmixer_1536_20.yaml中可以同时配置训练参数和评估参数。

- 配置ConvMixer和ImageNet-1k数据集。

  ```text
    # Architecture 81.37%
    arch: convmixer_1536_20
  
    # ===== Dataset ===== #
    data_url: ../data/imagenet
    set: ImageNet
    num_classes: 1000
    mix_up: 0.8
    cutmix: 1.0
    auto_augment: rand-m9-mstd0.5-inc1
    interpolation: bicubic
    re_prob: 0.0
    re_mode: pixel
    re_count: 1
    mixup_prob: 1.0
    switch_prob: 0.5
    mixup_mode: batch
    image_size: 224
    crop_pct: 0.96
  
    # ===== Learning Rate Policy ======== #
    optimizer: adamw
    base_lr: 0.0005
    eps: 0.001
    warmup_lr: 0.000001
    min_lr: 0.00001
    lr_scheduler: cosine_lr
    warmup_length: 20
  
    # ===== Network training config ===== #
    amp_level: O1
    keep_bn_fp32: True
    beta: [ 0.9, 0.999 ]
    clip_global_norm_value: 1.
    is_dynamic_loss_scale: True
    epochs: 300
    label_smoothing: 0.1
    weight_decay: 0.05
    momentum: 0.9
    batch_size: 32
    drop_path_rate: 0.2
  
    # ===== Hardware setup ===== #
    num_parallel_workers: 16
    device_target: Ascend
  ```

更多配置细节请参考脚本`convmixer_1536_20.yaml`。 通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

# [训练和测试](#目录)

- Ascend处理器环境运行

  ```bash
  # 使用python启动单卡训练
  python train.py --device_id 0 --device_target Ascend --config ./src/configs/convmixer_1536_20.yaml \
  > train.log 2>&1 &
  
  # 使用脚本启动单卡训练
  bash ./scripts/run_standalone_train_ascend.sh [DEVICE_ID] [CONFIG_PATH]
  
  # 使用脚本启动多卡训练
  bash ./scripts/run_distribute_train_ascend.sh [RANK_TABLE_FILE] [CONFIG_PATH]
  
  # 使用python启动单卡运行评估示例
  python eval.py --device_id 0 --device_target Ascend --config ./src/configs/convmixer_1536_20.yaml > ./eval.log 2>&1 &
  
  # 使用脚本启动单卡运行评估示例
  bash ./scripts/run_eval_ascend.sh [DEVICE_ID] [CONFIG_PATH] [CHECKPOINT_PATH]
  ```
  

对于分布式训练，需要提前创建JSON格式的hccl配置文件。

请遵循以下链接中的说明：

[hccl工具](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)


# [模型描述](#目录)

## 性能

### 评估性能

#### ImageNet-1k上的ConvMixer训练

| 参数                 | Ascend                          |
| -------------------------- |---------------------------------|
|模型| ConvMixer                            |
| 模型版本              | convmixer_1536_20                  |
| 资源                   | Ascend 910 8卡                   |
| 上传日期              | 2022-11-04                      |
| MindSpore版本          | 1.5.1                           |
| 数据集                    | ImageNet-1k Train，共1,281,167张图像 |
| 训练参数        | epoch=300, batch_size=256      |
| 优化器                  | AdamWeightDecay                 |
| 损失函数              | SoftTargetCrossEntropy          |
| 损失| 0.831                           |
| 输出                    | 概率                              |
| 分类准确率             | 八卡：top1:81.7% top5:95.7%      |
| 速度                      | 8卡：476.913毫秒/步                  |
| 训练耗时          | 192h30min03s（run on OpenI）       |


# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)