# Copyright (c) QIU Tian. All rights reserved.

from torch import nn

# runtime
device = 'cuda'
seed = 42
batch_size = 256
epochs = 300
clip_max_norm = 1.0
eval_interval = 1
num_workers = None  # auto
pin_memory = True
sync_bn = True
find_unused_params = False
dist_url = 'env://'
print_freq = 50
amp = True

# dataset
data_root = './data'
dataset = 'cifar10'

# data augmentation
image_size = 32

# model
model_lib = 'default'
model = 'vit_tiny_patch4_32'
model_kwargs = dict(in_chans=3, act_layer=nn.GELU, drop_path_rate=0.1)  # Do NOT set 'num_classes' in 'model_kwargs'.

# criterion
criterion = 'default'

# optimizer
optimizer = 'adamw'
lr = 0.0005 * (batch_size / 512)
weight_decay = 5e-2

# lr_scheduler
scheduler = 'cosine'
warmup_epochs = 20
warmup_lr = 1e-06
min_lr = 1e-05

# evaluator
evaluator = 'default'

# loading
no_pretrain = True

# saving
save_interval = 5
output_dir = f'./runs/{model}-{dataset}'

# remarks
note = f"Using the demo config in 'configs/_demo_.py'. | dataset: {dataset} | model: {model} | output_dir: {output_dir}"
