device = 'cuda'
seed = 42
batch_size = 128
epochs = 12
eval_interval = 1
num_workers = 2
pin_memory = True
sync_bn = True
data_root = './data'
dataset = 'cifar10'
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
amp = True

model_kwargs = dict(groups=1, width_per_group=64)  # do NOT set 'num_classes' in 'model_kwargs'
