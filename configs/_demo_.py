device = 'cuda'
seed = 42
batch_size = 256
epochs = 12
eval_interval = 1
num_workers = 2
pin_memory = True
sync_bn = True
data_root = './data'
dataset = 'mnist'
no_pretrain = True
model_lib = 'torchvision-ex'
model = 'resnet18'
optimizer = 'sgd'
lr = 2e-3
momentum = 0.9
scheduler = 'cosine'
save_interval = 1

output_dir = './runs/resnet18_baseline-cifar10'
note = "using the demo config in configs/_demo_.py"

print_freq = 10
clip_max_norm = 5.0

warmup_epochs = 2
min_lr = 1e-6
amp = True

model_kwargs = dict(in_chans=1, groups=1, width_per_group=64)  # Do NOT set 'num_classes' in 'model_kwargs'.
