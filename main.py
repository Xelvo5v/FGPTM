from util import get_dataloader, train_one_epoch, evaluate
from util.result_logging import (
    build_epoch_record,
    is_better_metrics,
    reset_metric_files,
    write_best_metrics,
    write_epoch_metrics,
)
from models import PTM_MoE
from pathlib import Path
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer_v2
import torch
import torch.nn as nn
from timm.utils import NativeScaler
import util.utils as utils
import numpy as np

import os

import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(
        'FGPTM training script', add_help=False)
    parser.add_argument('--ptm_type', default='Nitrosylation', type=str,
                        help='PTM type name under data/<ptm_type>')
    parser.add_argument('--data_root', default='./data', type=str,
                        help='root directory for PTM data')
    parser.add_argument('--dssp_path', default='./data/mkdssp', type=str,
                        help='path to mkdssp executable')
    parser.add_argument('--repeat', default=None, type=int,
                        help='optional Musite split repeat index under data/<ptm_type>/splits/repeat_<repeat>')
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--cache_workers', default=16, type=int,
                        help='parallel workers used only when building missing graph caches')
    parser.add_argument('--width', default=128, type=int)
    parser.add_argument('--seq_embed_dim', default=128, type=int,
                        help='amino-acid token embedding dimension used by PTM_MoE')
    parser.add_argument('--num_heads',default=4,type=int)
    parser.add_argument('--moe_num_experts', default=16, type=int,
                        help='number of experts used by Soft-MoE layers')
    
    #optimizer
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--alpha', default=0.85, type=float)
    
    parser.add_argument('--output_dir', default='', type=str,
                        help='path where to save, empty for no saving')
    parser.add_argument('--no_save_checkpoint', action='store_true',
                        help='write logs/metrics but do not save checkpoint weights')
    parser.add_argument('--resume', default=None, type=str,
                        help='resume path')
    parser.add_argument('--pre_train', default=None, type=str,
                        help='pre_train path')
    
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=5e-4, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=15, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=0, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument("--local_rank", default=0, type=int)
    
    parser.add_argument('--seed', default=88, type=int)
    
    return parser


def main(args):
    print(args)
    utils.init_distributed_mode(args)
    
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0
        device = torch.device("cpu")
    
    start_epoch = args.start_epoch
    train_loader, test_loader = get_dataloader(
        args.ptm_type,
        args.batch_size,
        args.num_workers,
        data_root=args.data_root,
        dssp_path=args.dssp_path,
        repeat=args.repeat,
        cache_workers=args.cache_workers,
    )
    
    print("Creating model")
    
    model = PTM_MoE(
        args.seq_embed_dim,
        args.width,
        args.num_heads,
        moe_num_experts=args.moe_num_experts,
    )
    print(model)
    if args.resume:
        checkpoint = torch.load(args.resume,map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'],strict=False)
        print(msg)
        start_epoch = checkpoint['epoch'] + 1
    if args.pre_train:
        checkpoint = torch.load(args.pre_train)
        msg = model.load_state_dict(checkpoint['model'],strict=False)
        print(msg)
    model = model.to(device)
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    if torch.distributed.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[local_rank] if device.type == "cuda" else None,
                                                          output_device=local_rank if device.type == "cuda" else None)
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    optimizer = create_optimizer_v2(model_without_ddp, opt='adamw', lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    lr_scheduler, _ = create_scheduler(args, optimizer)
    loss_scaler = NativeScaler()
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss_scaler.load_state_dict(checkpoint['scaler'])

    # loss_fn = utils.Focal_Loss(alpha=args.alpha,reduction="mean")
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    
    print("start training")
    if args.output_dir:
        os.makedirs(args.output_dir,exist_ok=True)
        if start_epoch == 0:
            reset_metric_files(args.output_dir)
    test_stats = {}
    best_record = None
    for epoch in range(start_epoch, args.epochs):
        if torch.distributed.is_initialized() and train_loader.sampler is not None:
            train_loader.sampler.set_epoch(epoch)
        lr_scheduler.step(epoch)
        print("epoch:{}".format(epoch+1))
        train_one_epoch(model,loss_fn,train_loader,optimizer,loss_scaler)
        test_stats = evaluate(model,loss_fn,test_loader,epoch,args)
        epoch_record = build_epoch_record(args, epoch, test_stats, n_parameters)
        write_epoch_metrics(args, epoch, test_stats, n_parameters)
        if is_better_metrics(epoch_record, best_record):
            best_record = epoch_record
        if args.output_dir and epoch==args.epochs-1 and not args.no_save_checkpoint:
            checkpoint_path = os.path.join(args.output_dir,"checkpoint{}.pth".format(epoch))
            utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)
    write_best_metrics(args, best_record)
    return test_stats
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'FGPTM training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
