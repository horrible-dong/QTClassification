QTClassification<sup>Dev</sup>
========

> **An elegant toolbox for image classification**   
> Author: QIU, Tian   
> Affiliate: Zhejiang University

## Installation

Coming soon ...

## Getting Started

For quick use, you can directly run the following commands:

```bash
# multi-gpu (recommend, needs pytorch>=1.9.0)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 main.py \
  --data_root ./data \
  --dataset cifar10 \
  --model resnet50 \
  --batch_size 4 \
  --epochs 50 \
  --output_dir ./runs/__tmp__
  
# multi-gpu (for any pytorch version)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node=2 main.py \
  --data_root ./data \
  --dataset cifar10 \
  --model resnet50 \
  --batch_size 4 \
  --epochs 50 \
  --output_dir ./runs/__tmp__
  
# single-gpu
python main.py \
  --data_root ./data \
  --dataset cifar10 \
  --model resnet50 \
  --batch_size 4 \
  --epochs 50 \
  --output_dir ./runs/__tmp__
  
```

Coming soon ...

## LICENSE

QTClassification is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.

Copyright (c) QIU, Tian. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use these files except in compliance with
the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "
AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

## Citation

Coming soon ...
