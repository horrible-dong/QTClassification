from torch import nn

device = 'cuda'
seed = 42
epochs = 300
eval_interval = 1
num_workers = None  # auto
pin_memory = True
sync_bn = True
find_unused_params = False
dist_url = 'env://'
need_targets = False
model_lib = 'default'
criterion = 'ce'
optimizer = 'adamw'
weight_decay = 5e-2
scheduler = 'cosine'
warmup_epochs = 20
warmup_lr = 1e-06
min_lr = 1e-05
evaluator = 'default'
no_pretrain = True
save_interval = 5
clip_max_norm = 1.0
amp = True

image_size = 32
batch_size = 256
lr = 0.0005 * (batch_size / 512)
data_root = './data'
dataset = 'cifar10'
model = 'vit_tiny_patch4_32'
model_kwargs = dict(in_chans=3, act_layer=nn.GELU, drop_path_rate=0.1)  # Do NOT set 'num_classes' in 'model_kwargs'.
output_dir = f'./runs/{model}-{dataset}'
note = f"Using the demo config in 'configs/_demo_.py'. | dataset: {dataset} | model: {model} | output_dir: {output_dir}"
print_freq = 20
