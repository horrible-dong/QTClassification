device = 'cuda'
seed = 42
batch_size = 128
epochs = 300
start_epoch = 0
eval = False
eval_interval = 1
num_workers = 2
pin_memory = True
distributed = False
sync_bn = True
find_unused_params = False
need_targets = False
data_root = './data'
dataset = 'cifar10'
model_lib = 'torchvision-ex'
model = 'resnet18'
optimizer = 'sgd'
lr = 0.0001
momentum = 0.9
weight_decay = 0.05
scheduler = 'cosine'
warmup_epochs = 0

output_dir = './runs/resnet18_baseline-cifar10'
save_interval = 1

note = "using demo config"

print_freq = 10
clip_max_norm = 5.0

amp = True
