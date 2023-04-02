# Copyright (c) QIU, Tian. All rights reserved.

import argparse
import datetime
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.utils.data as Data

from engine import evaluate, train_one_epoch
from qtcls import build_criterion, build_dataset, build_model, build_optimizer, build_scheduler
from qtcls.utils.io import checkpoint_saver, checkpoint_loader, variables_loader, variables_saver
from qtcls.utils.misc import makedirs, init_distributed_mode, init_seeds, is_main_process


def get_args_parser():
    parser = argparse.ArgumentParser('QTClassification', add_help=False)

    parser.add_argument('--config', '-c', type=str)

    # runtime
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', '-b', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--clip_max_norm', default=0.0, type=float, help='gradient clipping max norm')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--sync_bn', type=bool, default=False)
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--need_targets', action='store_true')
    parser.add_argument('--drop_lr_now', action='store_true')
    parser.add_argument('--drop_last', action='store_true')
    parser.add_argument('--amp', action='store_true', help='mixed precision training')

    # dataset
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--dataset', '-d', type=str, default='cifar10')

    # model
    parser.add_argument('--model_lib', default='torchvision-ex', type=str, choices=['torchvision-ex', 'timm'],
                        help='model library')
    parser.add_argument('--model', '-m', default='resnet50', type=str, help='model name')

    # criterion
    parser.add_argument('--criterion', default='default', type=str, help='criterion name')

    # optimizer
    parser.add_argument('--optimizer', default='adamw', type=str, help='optimizer name')
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--lr_drop', default=-1, type=int)
    parser.add_argument('--momentum', default=0.9, type=float, help='sgd momentum')
    parser.add_argument('--weight_decay', default=5e-2, type=float)

    # lr_scheduler
    parser.add_argument('--scheduler', default='cosine', type=str, help='scheduler name')
    parser.add_argument('--warmup_epochs', default=0, type=int, help='for warmup')
    parser.add_argument('--warmup_lr', default=1e-6, type=float, help='for warmup')
    parser.add_argument('--min_lr', default=1e-5, type=float, help='for Cosine')
    parser.add_argument('--step_size', type=int, help='for StepLR')
    parser.add_argument('--milestones', type=int, nargs='*', help='for MultiStepLR')
    parser.add_argument('--gamma', default=0.1, type=float, help='for StepLR and MultiStepLR')

    # evaluator
    parser.add_argument('--evaluator', default='default', type=str, help='evaluator name')

    # loading weights
    parser.add_argument('--no_pretrain', action='store_true')
    parser.add_argument('--resume', '-r', type=str, default='')
    parser.add_argument('--load_pos', type=str)

    # saving weights
    parser.add_argument('--output_dir', '-o', type=str, default='./runs/__tmp__')
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--save_pos', type=str)

    # remarks
    parser.add_argument('--note', type=str)

    return parser


def main(args):
    init_seeds(args.seed)
    init_distributed_mode(args)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        args.amp = False
    if args.num_workers is None:
        args.num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    if args.note is None:
        args.note = args.output_dir
    output_dir = Path(args.output_dir)

    print(args)
    variables_saver(vars(args), os.path.join(args.output_dir, 'config.py'))

    # ** model **
    model = build_model(args)
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'number of params: {n_parameters}')

    # ** optimizer & scheduler **
    param_dicts = [
        {'params': [p for n, p in model_without_ddp.named_parameters() if p.requires_grad]},
    ]

    optimizer = build_optimizer(args, param_dicts)
    lr_scheduler = build_scheduler(args, optimizer)

    # ** criterion **
    criterion = build_criterion(args)

    # ** dataset **
    dataset_train = build_dataset(args, split='train')
    dataset_val = build_dataset(args, split='val')

    if args.distributed:
        sampler_train = Data.distributed.DistributedSampler(dataset=dataset_train, shuffle=True)
        sampler_val = Data.distributed.DistributedSampler(dataset=dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = Data.DataLoader(dataset=dataset_train,
                                        sampler=sampler_train,
                                        batch_size=args.batch_size,
                                        drop_last=args.drop_last,
                                        pin_memory=args.pin_memory,
                                        num_workers=args.num_workers,
                                        collate_fn=dataset_train.collate_fn)

    data_loader_val = Data.DataLoader(dataset=dataset_val,
                                      sampler=sampler_val,
                                      batch_size=args.batch_size,
                                      pin_memory=args.pin_memory,
                                      num_workers=args.num_workers,
                                      collate_fn=dataset_val.collate_fn)

    # ** scaler **
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        checkpoint_loader(model_without_ddp, checkpoint['model'], delete_keys=())
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            checkpoint_loader(optimizer, checkpoint['optimizer'], verbose=False)
            checkpoint_loader(lr_scheduler, checkpoint['lr_scheduler'], verbose=False)
            args.start_epoch = checkpoint['epoch'] + 1
            if args.drop_lr_now:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1
        if scaler and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.eval:
        test_stats, evaluator = evaluate(
            model, data_loader_val, criterion, device, args, args.print_freq, args.need_targets
        )
        return

    print('\n' + 'Start training:')
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm, scaler, args.print_freq,
            args.need_targets
        )
        lr_scheduler.step(epoch)
        if args.output_dir and (epoch + 1) % args.save_interval == 0:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 1 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                checkpoint = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                if scaler:
                    checkpoint['scaler'] = scaler.state_dict()
                checkpoint_saver(checkpoint, checkpoint_path)

        test_stats, evaluator = evaluate(
            model, data_loader_val, criterion, device, args, args.print_freq, args.need_targets
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and is_main_process():
            log_path = output_dir / 'log.txt'
            log_exists = True if log_path.exists() else False
            with log_path.open('a') as f:
                f.write(json.dumps(log_stats) + '\n')
            if not log_exists:
                log_path.chmod(mode=0o777)

        if args.note:
            print(f'Note: {args.note}\n')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('QTClassification', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.config:
        args_vars = vars(args)
        cfg = variables_loader(args.config)
        for k, v in cfg.items():
            setattr(args, k, v)

    if args.data_root:
        makedirs(args.data_root, exist_ok=True)
    if args.output_dir:
        makedirs(args.output_dir, exist_ok=True)

    main(args)
