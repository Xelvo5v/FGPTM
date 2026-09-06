import argparse

import torch
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from models import PTM_MoE
from util import evaluate, get_dataset
import util.utils as utils


def get_args_parser():
    parser = argparse.ArgumentParser("FGPTM evaluation script", add_help=False)
    parser.add_argument("--ptm_type", default="Nitrosylation", type=str,
                        help="PTM type name under data/<ptm_type>")
    parser.add_argument("--data_root", default="./data", type=str,
                        help="root directory for PTM data")
    parser.add_argument("--dssp_path", default="./data/mkdssp", type=str,
                        help="path to mkdssp executable")
    parser.add_argument("--repeat", default=None, type=int,
                        help="optional Musite split repeat index under data/<ptm_type>/splits/repeat_<repeat>")
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--cache_workers", default=1, type=int,
                        help="parallel workers used only when building missing graph caches")
    parser.add_argument("--width", default=420, type=int)
    parser.add_argument("--seq_embed_dim", default=128, type=int,
                        help="amino-acid token embedding dimension used by PTM_MoE")
    parser.add_argument("--num_heads", default=12, type=int)
    parser.add_argument("--moe_num_experts", default=None, type=int,
                        help="number of experts used by Soft-MoE layers; defaults to checkpoint args or 16")
    parser.add_argument("--ckpt", required=True, type=str,
                        help="checkpoint path")
    parser.add_argument("--output_dir", default="", type=str,
                        help="optional path for appending evaluation log")

    parser.add_argument("--world_size", default=1, type=int,
                        help="number of distributed processes")
    parser.add_argument("--dist_url", default="env://",
                        help="url used to set up distributed training")
    parser.add_argument("--local_rank", default=0, type=int)
    return parser


def build_test_loader(args):
    test_dataset = get_dataset(
        args.ptm_type,
        "test",
        data_root=args.data_root,
        dssp_path=args.dssp_path,
        build_cache=True,
        repeat=args.repeat,
        cache_workers=args.cache_workers,
    )
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if torch.distributed.is_initialized() else None
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "sampler": test_sampler,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(test_dataset, **loader_kwargs)


def main(args):
    utils.init_distributed_mode(args)
    if torch.cuda.is_available():
        local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0
        device = torch.device("cpu")

    test_loader = build_test_loader(args)

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = checkpoint.get("args") if isinstance(checkpoint, dict) else None
    moe_num_experts = args.moe_num_experts
    if moe_num_experts is None and ckpt_args is not None:
        moe_num_experts = getattr(ckpt_args, "moe_num_experts", None)
    if moe_num_experts is None:
        moe_num_experts = 16

    model = PTM_MoE(
        args.seq_embed_dim,
        args.width,
        args.num_heads,
        moe_num_experts=moe_num_experts,
    )
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    model = model.to(device)
    if torch.distributed.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
        )

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")
    evaluate(model, loss_fn, test_loader, epoch=0, args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("FGPTM evaluation script", parents=[get_args_parser()])
    main(parser.parse_args())
