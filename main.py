import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import wandb
import random
import os
import numpy as np

from train import train
from test import test
from models.simclr import SimCLR
from data import KADIS700Dataset
from utils.utils import PROJECT_ROOT, parse_config, parse_command_line_args, merge_configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    args, unknown = parser.parse_known_args()
    config = parse_config(args.config)
    # Parse the command-line arguments and merge with the config
    args = parse_command_line_args(config)
    args = merge_configs(config, args)  # Command-line arguments take precedence over config file
    print(args)

    # Set the device
    if args.device != -1 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")

    # Set seed
    SEED = args.seed
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    np.random.seed(SEED)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    args.data_base_path = Path(args.data_base_path)
    args.checkpoint_base_path = PROJECT_ROOT / "experiments"

    # Initialize the training dataset and dataloader
    train_dataset = KADIS700Dataset(root=args.data_base_path / "KADIS700",
                                    patch_size=args.training.data.patch_size,
                                    max_distortions=args.training.data.max_distortions,
                                    num_levels=args.training.data.num_levels,
                                    pristine_prob=args.training.data.pristine_prob)
    train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size, num_workers=args.training.num_workers,
                                  shuffle=True, pin_memory=True, drop_last=True)

    # Initialize the model
    model = SimCLR(encoder_params=args.model.encoder, temperature=args.model.temperature)
    model = model.to(device)

    # Initialize the optimizer
    if args.training.optimizer.name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.training.lr,
                                     weight_decay=args.training.optimizer.weight_decay,
                                     betas=args.training.optimizer.betas, eps=args.training.optimizer.eps)
    elif args.training.optimizer.name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.training.lr,
                                      weight_decay=args.training.optimizer.weight_decay,
                                      betas=args.training.optimizer.betas, eps=args.training.optimizer.eps)
    elif args.training.optimizer.name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.training.lr, momentum=args.training.optimizer.momentum,
                                    weight_decay=args.training.optimizer.weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {args.training.optimizer.name} not implemented")

    # Initialize the scheduler
    if "lr_scheduler" in args.training and args.training.lr_scheduler.name == "CosineAnnealingWarmRestarts":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                            T_0=args.training.lr_scheduler.T_0,
                                                                            T_mult=args.training.lr_scheduler.T_mult,
                                                                            eta_min=args.training.lr_scheduler.eta_min,
                                                                            verbose=False)
    else:
        lr_scheduler = None

    scaler = torch.cuda.amp.GradScaler()  # Automatic mixed precision scaler

    run_id = None
    if args.training.resume_training:
        try:
            checkpoint_path = args.checkpoint_base_path / args.experiment_name / "pretrain"
            checkpoint_path = [el for el in checkpoint_path.glob("*.pth") if "last" in el.name][0]
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            epoch = checkpoint["epoch"]
            args.training.start_epoch = epoch + 1
            run_id = checkpoint["config"]["logging"]["wandb"].get("run_id", None)
            args.best_srocc = checkpoint["config"]["best_srocc"]
            print(f"--- Resuming training after epoch {epoch + 1} ---")
        except Exception:
            print("ERROR: Could not resume training. Starting from scratch.")

    # Initialize logger
    if args.logging.use_wandb:
        logger = wandb.init(project=args.logging.wandb.project,
                            entity=args.logging.wandb.entity,
                            name=args.experiment_name if not args.training.resume_training else None,
                            config=args,
                            mode="online" if args.logging.wandb.online else "offline",
                            resume=args.training.resume_training,
                            id=run_id)
        args.logging.wandb.run_id = logger.id
    else:
        logger = None

    train(args, model, train_dataloader, optimizer, lr_scheduler, scaler, logger, device)
    print("--- Training finished ---")

    checkpoint_path = args.checkpoint_base_path / args.experiment_name / "pretrain"
    checkpoint_path = [ckpt_path for ckpt_path in checkpoint_path.glob("*.pth") if "best" in ckpt_path.name][0]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    print(f"Starting testing with best checkpoint...")

    test(args, model, logger, device)
    print("--- Testing finished ---")


if __name__ == '__main__':
    main()
