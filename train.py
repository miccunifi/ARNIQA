import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import yaml
import wandb
from wandb.wandb_run import Run
from PIL import ImageFile
from dotmap import DotMap
from typing import Optional, Tuple

from data import KADID10KDataset
from test import get_results, synthetic_datasets, authentic_datasets
from utils.visualization import visualize_tsne_umap_mos

ImageFile.LOAD_TRUNCATED_IMAGES = True


def train(args: DotMap,
          model: torch.nn.Module,
          train_dataloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          lr_scheduler: Optional[LRScheduler],
          scaler: torch.cuda.amp.GradScaler,
          logger: Optional[Run],
          device: torch.device) -> None:
    """
    Train the given model with the strategy proposed in the paper https://arxiv.org/abs/2310.14918.

    Args:
        args (dotmap.DotMap): the training arguments
        model (torch.nn.Module): the model to train
        train_dataloader (torch.utils.data.Dataloader): the training data loader
        optimizer (torch.optim.Optimizer): the optimizer to use
        lr_scheduler (Optional[torch.optim.lr_scheduler]): the learning rate scheduler to use
        scaler (torch.cuda.amp.GradScaler): the scaler to use for mixed precision training
        logger (Optional[wandb.wandb_run.Run]): the logger to use
        device (torch.device): the device to use for training
    """
    checkpoint_path = args.checkpoint_base_path / args.experiment_name / "pretrain"
    checkpoint_path.mkdir(parents=True, exist_ok=False)
    print("Saving checkpoints in folder: ", checkpoint_path)
    with open(args.checkpoint_base_path / args.experiment_name / "config.yaml", "w") as f:
        dumpable_args = args.copy()
        for key, value in dumpable_args.items():  # Convert PosixPath values of args to string
            if isinstance(value, Path):
                dumpable_args[key] = str(value)
        yaml.dump(dumpable_args.toDict(), f)

    # Initialize training parameters
    if args.training.resume_training:
        start_epoch = args.training.start_epoch
        max_epochs = args.training.epochs
        best_srocc = args.best_srocc
    else:
        start_epoch = 0
        max_epochs = args.training.epochs
        best_srocc = 0

    last_srocc = 0
    last_plcc = 0
    last_model_filename = ""
    best_model_filename = ""

    # Training loop
    for epoch in range(start_epoch, max_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{max_epochs}]")

        for i, batch in enumerate(tqdm(train_dataloader)):
            num_logging_steps = i * args.training.batch_size + len(train_dataloader) * args.training.batch_size * epoch

            # Initialize inputs
            inputs_A_orig = batch["img_A_orig"].to(device=device, non_blocking=True)
            inputs_A_ds = batch["img_A_ds"].to(device=device, non_blocking=True)
            inputs_A = torch.cat((inputs_A_orig, inputs_A_ds), dim=0)
            inputs_B_orig = batch["img_B_orig"].to(device=device, non_blocking=True)
            inputs_B_ds = batch["img_B_ds"].to(device=device, non_blocking=True)
            inputs_B = torch.cat((inputs_B_orig, inputs_B_ds), dim=0)
            img_A_name = batch["img_A_name"]
            img_B_name = batch["img_B_name"]

            distortion_functions = np.array(batch["distortion_functions"]).T  # Handle PyTorch's indexing of lists
            distortion_functions = [list(filter(None, el)) for el in distortion_functions]  # Remove padding
            distortion_values = torch.stack(batch["distortion_values"]).T  # Handle PyTorch's indexing of lists
            distortion_values = [el[el != torch.inf] for el in distortion_values]  # Remove padding

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            with torch.cuda.amp.autocast():  # For fp16 training
                loss = model(inputs_A, inputs_B)

            if torch.isnan(loss):
                raise ValueError("Loss is NaN")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if lr_scheduler and lr_scheduler.__class__.__name__ == "CosineAnnealingWarmRestarts":
                lr_scheduler.step(epoch + i / len(train_dataloader))

            cur_loss = loss.item()
            running_loss += cur_loss
            progress_bar.set_postfix(loss=running_loss / (i + 1), SROCC=last_srocc, PLCC=last_plcc)

            # Logging
            if logger:
                logger.log({"loss": cur_loss, "lr": optimizer.param_groups[0]["lr"]}, step=num_logging_steps)

                # Log images
                if i % args.training.log_images_frequency == 0:
                    log_input = []
                    for j, (img_A, name_A, img_B, name_B, dist_funcs, dist_values) in enumerate(
                            zip(inputs_A_orig, img_A_name, inputs_B_orig, img_B_name, distortion_functions, distortion_values)):
                        caption = "A_orig_" + name_A + "_" + "_".join(
                            [f"{value:.2f}{dist}" for dist, value in zip(dist_funcs, dist_values)])
                        log_img = wandb.Image(torch.clip(img_A, 0, 1), caption=caption)
                        log_input.append(log_img)
                        caption = "B_orig_" + name_B + "_" + "_".join(
                            [f"{value:.2f}{dist}" for dist, value in zip(dist_funcs, dist_values)])
                        log_img = wandb.Image(torch.clip(img_B, 0, 1), caption=caption)
                        log_input.append(log_img)
                        caption = "A_ds_" + name_A + "_" + "_".join(
                            [f"{value:.2f}{dist}" for dist, value in zip(dist_funcs, dist_values)])
                        log_img = wandb.Image(torch.clip(inputs_A_ds[j], 0, 1), caption=caption)
                        log_input.append(log_img)
                        caption = "B_ds_" + name_B + "_" + "_".join(
                            [f"{value:.2f}{dist}" for dist, value in zip(dist_funcs, dist_values)])
                        log_img = wandb.Image(torch.clip(inputs_B_ds[j], 0, 1), caption=caption)
                        log_input.append(log_img)
                    logger.log({"input": log_input}, step=num_logging_steps)

        if lr_scheduler and lr_scheduler.__class__.__name__ != "CosineAnnealingWarmRestarts":
            lr_scheduler.step()

        # Validation
        if epoch % args.validation.frequency == 0:
            print("Starting validation...")
            last_srocc, last_plcc = validate(args, model, logger, num_logging_steps, device)

            # Log embeddings visualizations
            if args.validation.visualize and logger:
                kadid10k_val = KADID10KDataset(args.data_base_path / "KADID10K", phase="val")
                val_dataloader = DataLoader(kadid10k_val, batch_size=args.test.batch_size, shuffle=False,
                                            num_workers=args.test.num_workers)
                figures = visualize_tsne_umap_mos(model, val_dataloader,
                                                  tsne_args=args.validation.visualization.tsne,
                                                  umap_args=args.validation.visualization.umap,
                                                  device=device)
                logger.log(figures, step=num_logging_steps)

            progress_bar.set_postfix(loss=running_loss / (i + 1), SROCC=last_srocc, PLCC=last_plcc)

        # Save checkpoints
        print("Saving checkpoint")

        # Save best checkpoint weights
        if last_srocc > best_srocc:
            best_srocc = last_srocc
            best_plcc = last_plcc
            # Save best metrics in arguments for resuming training
            args.best_srocc = best_srocc
            args.best_plcc = best_plcc
            if best_model_filename:
                os.remove(checkpoint_path / best_model_filename)  # Remove previous best model
            best_model_filename = f"best_epoch_{epoch}_srocc_{best_srocc:.3f}_plcc_{best_plcc:.3f}.pth"
            torch.save(model.state_dict(), checkpoint_path / best_model_filename)

        # Save last checkpoint
        if last_model_filename:
            os.remove(checkpoint_path / last_model_filename)  # Remove previous last model
        last_model_filename = f"last_epoch_{epoch}_srocc_{last_srocc:.3f}_plcc_{last_plcc:.3f}.pth"
        args.last_srocc = last_srocc
        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "epoch": epoch,
                    "config": args,
                    }, checkpoint_path / last_model_filename)

    print('Finished training')


def validate(args: DotMap,
             model: torch.nn.Module,
             logger: Optional[Run],
             num_logging_steps: int,
             device: torch.device) -> Tuple[float, float]:
    """
    Validate the given model on the validation datasets.

    Args:
        args (dotmap.DotMap): the training arguments
        model (torch.nn.Module): the model to validate
        logger (Optional[wandb.wandb_run.Run]): the logger to use
        num_logging_steps (int): the number of logging steps
        device (torch.device): the device to use
    """
    model.eval()

    srocc_all, plcc_all, _, _, _ = get_results(model=model, data_base_path=args.data_base_path,
                                               datasets=args.validation.datasets, num_splits=args.validation.num_splits,
                                               phase="val", alpha=args.validation.alpha, grid_search=False,
                                               crop_size=args.test.crop_size, batch_size=args.test.batch_size,
                                               num_workers=args.test.num_workers, device=device)

    # Compute the median for each list in srocc_all and plcc_all
    srocc_all_median = {key: np.median(value["global"]) for key, value in srocc_all.items()}
    plcc_all_median = {key: np.median(value["global"]) for key, value in plcc_all.items()}

    # Compute the synthetic and authentic averages
    srocc_synthetic_avg = np.mean(
        [srocc_all_median[key] for key in srocc_all_median.keys() if key in synthetic_datasets])
    plcc_synthetic_avg = np.mean([plcc_all_median[key] for key in plcc_all_median.keys() if key in synthetic_datasets])
    srocc_authentic_avg = np.mean(
        [srocc_all_median[key] for key in srocc_all_median.keys() if key in authentic_datasets])
    plcc_authentic_avg = np.mean([plcc_all_median[key] for key in plcc_all_median.keys() if key in authentic_datasets])

    # Compute the global average
    srocc_avg = np.mean(list(srocc_all_median.values()))
    plcc_avg = np.mean(list(plcc_all_median.values()))

    if logger:
        logger.log({f"val_srocc_{key}": srocc_all_median[key] for key in srocc_all_median.keys()}, step=num_logging_steps)
        logger.log({f"val_plcc_{key}": plcc_all_median[key] for key in plcc_all_median.keys()}, step=num_logging_steps)
        logger.log({"val_srocc_synthetic_avg": srocc_synthetic_avg, "val_plcc_synthetic_avg": plcc_synthetic_avg,
                    "val_srocc_authentic_avg": srocc_authentic_avg, "val_plcc_authentic_avg": plcc_authentic_avg,
                    "val_srocc_avg": srocc_avg, "val_plcc_avg": plcc_avg}, step=num_logging_steps)

    return srocc_avg, plcc_avg
