QTClassification
========

**A lightweight and extensible toolbox for image classification**

[![version](https://img.shields.io/badge/Version-0.8.1--dev-brightgreen)](https://github.com/horrible-dong/QTClassification)
&emsp;[![docs](https://img.shields.io/badge/Docs-Latest-orange)](https://github.com/horrible-dong/QTClassification/blob/main/README.md)
&emsp;[![license](https://img.shields.io/badge/License-Apache--2.0-blue)](https://github.com/horrible-dong/QTClassification/blob/main/LICENSE)

> Author: QIU Tian  
> Affiliation: Zhejiang University  
> <a href="#installation">🛠️ Installation</a> | <a href="#getting_started">📘
> Documentation </a> | <a href="#dataset_zoo">🌱 Dataset Zoo</a> | <a href="#model_zoo">👀 Model Zoo</a>  
> English | [简体中文](README_zh-CN.md)

## <span id="Installation">Installation</span>

The development environment of this project is `python 3.7 & pytorch 1.11.0+cu113`.

1. Create your conda environment if needed.

```bash
conda create -n qtcls python==3.7 -y
```

2. Enter your conda environment.

```bash
conda activate qtcls
```

3. Install PyTorch.

```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

or you can refer to [PyTorch](https://pytorch.org/get-started/previous-versions/) to install newer or older versions.
Please note that if pytorch ≥ 1.13, python ≥ 3.8 is required.

4. Install the necessary dependencies.

```bash
pip install -r requirements.txt
```

## <span id="getting_started">Getting Started</span>

For a quick experience, you can directly run the following commands:

**Training**

```bash
# multi-gpu (recommended, needs pytorch>=1.9.0)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 main.py \
  --data_root ./data \
  --dataset cifar10 \
  --model resnet50 \
  --batch_size 256 \
  --lr 1e-4 \
  --epochs 12 \
  --output_dir ./runs/__tmp__
  
# multi-gpu (for any pytorch version)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py \
  --data_root ./data \
  --dataset cifar10 \
  --model resnet50 \
  --batch_size 256 \
  --lr 1e-4 \
  --epochs 12 \
  --output_dir ./runs/__tmp__
  
# single-gpu
CUDA_VISIBLE_DEVICES=0 \
python main.py \
  --data_root ./data \
  --dataset cifar10 \
  --model resnet50 \
  --batch_size 256 \
  --lr 1e-4 \
  --epochs 12 \
  --output_dir ./runs/__tmp__
```

The `cifar10` dataset and `resnet50` pretrained weights will be automatically downloaded. The `cifar10` dataset will be
downloaded to `./data`. During the training, the config file, checkpoints, logs and other outputs will be stored in
`./runs/__tmp__`.

**Evaluation**

```bash
# multi-gpu (recommended, needs pytorch>=1.9.0)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 main.py \
  --data_root ./data \
  --dataset cifar10 \
  --model resnet50 \
  --batch_size 256 \
  --resume ./runs/__tmp__/checkpoint.pth \
  --eval
  
# multi-gpu (for any pytorch version)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py \
  --data_root ./data \
  --dataset cifar10 \
  --model resnet50 \
  --batch_size 256 \
  --resume ./runs/__tmp__/checkpoint.pth \
  --eval
  
# single-gpu
CUDA_VISIBLE_DEVICES=0 \
python main.py \
  --data_root ./data \
  --dataset cifar10 \
  --model resnet50 \
  --batch_size 256 \
  --resume ./runs/__tmp__/checkpoint.pth \
  --eval
```

### How to use

When using the toolbox for training and evaluation, you may run the commands we provided above *with your own
arguments*.

**Frequently-used command-line arguments**

|  Command-Line Argument   |                                                                                                                                           Description                                                                                                                                           |  Default Value   |
|:------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------:|
|      `--data_root`       |                                                                                                                            Directory where your datasets is stored.                                                                                                                             |     `./data`     |
|  `--dataset`<br />`-d`   |                                                                                    Dataset name defined in [qtcls/datasets/\_\_init\_\_.py](qtcls/datasets/__init__.py), such as `cifar10` and `imagenet1k`.                                                                                    |        /         |
|      `--model_lib`       |                                                         Model library where models come from. The toolbox's basic (default) model library is extended from `torchvision` and `timm`, and the toolbox also supports the original `timm`.                                                         |    `default`     |
|   `--model`<br />`-m`    |                                              Model name defined in [qtcls/models/\_\_init\_\_.py](qtcls/models/__init__.py), such as `resnet50` and `vit_b_16`. Currently supported model names are listed in <a href="#model_zoo">Model Zoo</a>.                                               |        /         |
|      `--criterion`       |                                                                                            Criterion name defined in [qtcls/criterions/\_\_init\_\_.py](qtcls/criterions/__init__.py), such as `ce`.                                                                                            |    `default`     |
|      `--optimizer`       |                                                                                      Optimizer name defined in [qtcls/optimizers/\_\_init\_\_.py](qtcls/optimizers/__init__.py), such as `sgd` and `adam`.                                                                                      |     `adamw`      |
|      `--scheduler`       |                                                                                          Scheduler name defined in [qtcls/schedulers/\_\_init\_\_.py](qtcls/schedulers/__init__.py), such as `cosine`.                                                                                          |     `cosine`     |
|      `--evaluator`       |                                                           Evaluator name defined in [qtcls/evaluators/\_\_init\_\_.py](qtcls/evaluators/__init__.py). The `default` evaluator computes the accuracy, recall, precision, and f1_score.                                                           |    `default`     |
|  `--pretrain`<br />`-p`  | Path to the pre-trained weights, which is of the higher priority than the path stored in [qtcls/models/\_pretrain\_.py](qtcls/models/_pretrain_.py). For long-term use of a pretrained weight path, it is preferable to write it in [qtcls/models/\_pretrain\_.py](qtcls/models/_pretrain_.py). |        /         |
|     `--no_pretrain`      |                                                                                                                            Forcibly not use the pre-trained weights.                                                                                                                            |     `False`      |
|   `--resume`<br />`-r`   |                                                                                                                                 Checkpoint path to resume from.                                                                                                                                 |        /         |
| `--output_dir`<br />`-o` |                                                                                                                       Path to save checkpoints, logs, and other outputs.                                                                                                                        | `./runs/__tmp__` |
|    `--save_interval`     |                                                                                                                                Interval for saving checkpoints.                                                                                                                                 |       `1`        |
| `--batch_size`<br />`-b` |                                                                                                                                                /                                                                                                                                                |       `8`        |
|        `--epochs`        |                                                                                                                                                /                                                                                                                                                |      `300`       |
|          `--lr`          |                                                                                                                                         Learning rate.                                                                                                                                          |      `1e-4`      |
|         `--amp`          |                                                                                                                           Enable automatic mixed precision training.                                                                                                                            |     `False`      |
|         `--eval`         |                                                                                                                                         Evaluate only.                                                                                                                                          |     `False`      |
|         `--note`         |                                                                                                    Note. The note content prints after each epoch, in case you forget what you are running.                                                                                                     |        /         |

**Using the config file (Recommended)**

Or you can write the arguments into a config file (.py) and directly use `--config` / `-c`
to import it.

`--config` / `-c`: Config file path. See [configs](configs). Arguments in the config file merge or override command-line
arguments `args`.

For example,

```bash
python main.py --config configs/_demo_.py
```

or

```bash
python main.py -c configs/_demo_.py
```

For more details, please see ["How to write and import your configs"](configs/README.md).

**Dataset placement**

Currently, `mnist`, `fashion_mnist`, `cifar10`, `cifar100`, `stl10`, `svhn`, `pets`, `flowers`, `cars` and `food`
datasets will be automatically downloaded to the `--data_root` directory. For other datasets, please refer
to ["How to put your dataset"](data/README.md).

### How to customize

The toolbox is flexible enough to be extended. Please follow the instructions below:

[How to register your datasets](qtcls/datasets/README.md)

[How to register your models](qtcls/models/README.md)

[How to register your criterions](qtcls/criterions/README.md)

[How to register your optimizers](qtcls/optimizers/README.md)

[How to register your schedulers](qtcls/schedulers/README.md)

[How to register your evaluators](qtcls/evaluators/README.md)

[How to write and import your configs](configs/README.md)

## <span id="dataset_zoo">Dataset Zoo</span>

Currently supported argument `--dataset` / `-d`:  
`mnist`, `fashion_mnist`, `cifar10`, `cifar100`, `stl10`, `svhn`, `pets`, `flowers`, `cars`, `food`,
`imagenet1k`, `imagenet21k (also called imagenet22k)`,
and all datasets in `folder` format (consistent with `imagenet` storage format,
see ["How to put your dataset - About folder format datasets"](data/README.md) for details).

## <span id="model_zoo">Model Zoo</span>

The toolbox's basic (default) model library is extended from `torchvision` and `timm`, and the toolbox also supports the
original `timm`.

### default

Set the argument `--model_lib` to `default`.

Currently supported argument `--model` / `-m`:

**AlexNet**  
`alexnet`

**CaiT**  
`cait_xxs24_224`, `cait_xxs24_384`, `cait_xxs36_224`, `cait_xxs36_384`, `cait_xs24_384`, `cait_s24_224`, `cait_s24_384`, `cait_s36_384`, `cait_m36_384`, `cait_m48_448`

**ConvNeXt**  
`convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large`

**DeiT**  
`deit_tiny_patch16_224`, `deit_small_patch16_224`, `deit_base_patch16_224`, `deit_base_patch16_384`, `deit_tiny_distilled_patch16_224`, `deit_small_distilled_patch16_224`, `deit_base_distilled_patch16_224`, `deit_base_distilled_patch16_384`, `deit3_small_patch16_224`, `deit3_small_patch16_384`, `deit3_medium_patch16_224`, `deit3_base_patch16_224`, `deit3_base_patch16_384`, `deit3_large_patch16_224`, `deit3_large_patch16_384`, `deit3_huge_patch14_224`, `deit3_small_patch16_224_in21ft1k`, `deit3_small_patch16_384_in21ft1k`, `deit3_medium_patch16_224_in21ft1k`, `deit3_base_patch16_224_in21ft1k`, `deit3_base_patch16_384_in21ft1k`, `deit3_large_patch16_224_in21ft1k`, `deit3_large_patch16_384_in21ft1k`, `deit3_huge_patch14_224_in21ft1k`

**DenseNet**  
`densenet121`, `densenet169`, `densenet201`, `densenet161`

**EfficientNet**  
`efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`, `efficientnet_b3`, `efficientnet_b4`, `efficientnet_b5`, `efficientnet_b6`, `efficientnet_b7`

**GoogLeNet**  
`googlenet`

**Inception**    
`inception_v3`

**LeViT**    
`levit_128s`, `levit_128`, `levit_192`, `levit_256`, `levit_256d`, `levit_384`

**MLP-Mixer**   
`mixer_s32_224`, `mixer_s16_224`, `mixer_b32_224`, `mixer_b16_224`, `mixer_b16_224_in21k`, `mixer_l32_224`, `mixer_l16_224`, `mixer_l16_224_in21k`, `mixer_b16_224_miil_in21k`, `mixer_b16_224_miil`, `gmixer_12_224`, `gmixer_24_224`, `resmlp_12_224`, `resmlp_24_224`, `resmlp_36_224`, `resmlp_big_24_224`, `resmlp_12_distilled_224`, `resmlp_24_distilled_224`, `resmlp_36_distilled_224`, `resmlp_big_24_distilled_224`, `resmlp_big_24_224_in22ft1k`, `resmlp_12_224_dino`, `resmlp_24_224_dino`, `gmlp_ti16_224`, `gmlp_s16_224`, `gmlp_b16_224`

**MNASNet**   
`mnasnet0_5`, `mnasnet0_75`, `mnasnet1_0`, `mnasnet1_3`

**MobileNet**  
`mobilenet_v2`, `mobilenetv3`, `mobilenet_v3_large`, `mobilenet_v3_small`

**PoolFormer**   
`poolformer_s12`, `poolformer_s24`, `poolformer_s36`, `poolformer_m36`, `poolformer_m48`

**PVT**  
`pvt_tiny`, `pvt_small`, `pvt_medium`, `pvt_large`, `pvt_huge_v2`

**RegNet**  
`regnet_y_400mf`, `regnet_y_800mf`, `regnet_y_1_6gf`, `regnet_y_3_2gf`, `regnet_y_8gf`, `regnet_y_16gf`, `regnet_y_32gf`, `regnet_y_128gf`, `regnet_x_400mf`, `regnet_x_800mf`, `regnet_x_1_6gf`, `regnet_x_3_2gf`, `regnet_x_8gf`, `regnet_x_16gf`, `regnet_x_32gf`

**ResNet**     
`resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`, `resnext50_32x4d`, `resnext101_32x8d`, `wide_resnet50_2`, `wide_resnet101_2`

**ShuffleNetV2**  
`shufflenet_v2_x0_5`, `shufflenet_v2_x1_0`, `shufflenet_v2_x1_5`, `shufflenet_v2_x2_0`

**SqueezeNet**  
`squeezenet1_0`, `squeezenet1_1`

**Swin Transformer**  
`swin_tiny_patch4_window7_224`, `swin_small_patch4_window7_224`, `swin_base_patch4_window7_224`, `swin_base_patch4_window12_384`, `swin_large_patch4_window7_224`, `swin_large_patch4_window12_384`, `swin_base_patch4_window7_224_in22k`, `swin_base_patch4_window12_384_in22k`, `swin_large_patch4_window7_224_in22k`, `swin_large_patch4_window12_384_in22k`

**Swin Transformer V2**  
`swinv2_tiny_window8_256`, `swinv2_tiny_window16_256`, `swinv2_small_window8_256`, `swinv2_small_window16_256`, `swinv2_base_window8_256`, `swinv2_base_window16_256`, `swinv2_base_window12_192_22k`, `swinv2_base_window12to16_192to256_22kft1k`, `swinv2_base_window12to24_192to384_22kft1k`, `swinv2_large_window12_192_22k`, `swinv2_large_window12to16_192to256_22kft1k`, `swinv2_large_window12to24_192to384_22kft1k`

**TNT**  
`tnt_s_patch4_32`, `tnt_s_patch16_224`, `tnt_b_patch16_224`

**Twins**  
`twins_pcpvt_small`, `twins_pcpvt_base`, `twins_pcpvt_large`, `twins_svt_small`, `twins_svt_base`, `twins_svt_large`

**VGG**  
`vgg11`, `vgg11_bn`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19`, `vgg19_bn`

**Vision Transformer (timm)**   
`vit_tiny_patch4_32`, `vit_tiny_patch16_224`, `vit_tiny_patch16_384`, `vit_small_patch32_224`, `vit_small_patch32_384`, `vit_small_patch16_224`, `vit_small_patch16_384`, `vit_small_patch8_224`, `vit_base_patch32_224`, `vit_base_patch32_384`, `vit_base_patch16_224`, `vit_base_patch16_384`, `vit_base_patch8_224`, `vit_large_patch32_224`, `vit_large_patch32_384`, `vit_large_patch16_224`, `vit_large_patch16_384`, `vit_large_patch14_224`, `vit_huge_patch14_224`, `vit_giant_patch14_224`

**Vision Transformer (torchvision)**  
`vit_b_16`, `vit_b_32`, `vit_l_16`, `vit_l_32`

### timm

Set the argument `--model_lib` to `timm`.

Currently supported argument `--model` / `-m`:  
All supported. Please refer to `timm` for the specific model name.

## LICENSE

QTClassification is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.

Copyright (c) QIU Tian. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use these files except in compliance with
the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an 
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

## Citation

If you find QTClassification Toolbox useful in your research, please consider citing:

```bibtex
@misc{2023QTClassification,
    title={QTClassification},
    author={Qiu, Tian},
    howpublished={\url{https://github.com/horrible-dong/QTClassification}},
    year={2023}
}
```
