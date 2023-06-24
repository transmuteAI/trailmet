import torch
import torch.optim as optim
import torch.nn as nn
from trailmet.algorithms.prune.utils import (
    update_bn_grad,
    summary_model,
)
from trailmet.algorithms.prune.pns import SlimPruner
from trailmet.algorithms.prune.prune import BasePruning

import logging
from datetime import datetime
from tqdm import tqdm
import wandb
import pandas as pd
import numpy as np
import os
import time

from trailmet.utils import AverageMeter, accuracy, save_checkpoint, adjust_learning_rate
from trailmet.models import get_resnet_model

logger = logging.getLogger(__name__)


class Network_Slimming(BasePruning):
    """
    Base Algorithm class that defines the structure of each model compression algorithm implemented in this library.
    Every new algorithm is expected to directly use or overwrite the template functions defined below.
    The root command to invoke the compression of any model is .compress_model(). Thus, it is required that all
    algorithms complete this template function and use it as the first point of invoking the model compression process.
    For methods that require to perform pretraining and fine-tuning, the implementation of base_train() method can
    directly be used for both the tasks. In case of modifications, overwrite this function based on the needs.
    """

    def __init__(self, dataloaders, **kwargs):
        self.device = "cuda"
        self.num_classes = kwargs.get("num_classes", 100)
        self.dataloaders = dataloaders
        self.pr = kwargs.get("pr", None)
        self.ft_only = kwargs.get("ft_only", False)
        self.scheduler_type = kwargs.get("scheduler_type", 1)
        self.weight_decay = kwargs.get("weight_decay", 5e-4)
        self.net = kwargs.get("net")
        self.dataset = kwargs.get("dataset")
        self.epochs = kwargs.get("epochs", 200)
        self.s = kwargs.get("s", 1e-3)
        self.lr = kwargs.get("learning_rate", 2e-3)
        self.prune_schema = os.path.join(
            kwargs.get("schema_root"), f"schema/{self.net}.json"
        )
        self.sparsity_train = kwargs.get("sparsity_train", True)
        self.fine_tune_epochs = kwargs.get("fine_tune_epochs", 165)
        self.fine_tune_lr = kwargs.get("fine_tune_learning_rate", 1e-4)
        self.prune_ratio = kwargs.get("prune_ratio", 0.5)

        self.wandb_monitor = kwargs.get("wandb", "False")
        self.kwargs = kwargs
        self.dataset_name = dataloaders["train"].dataset.__class__.__name__
        self.save = "./checkpoints/"

        self.name = "_".join(
            [
                self.dataset_name,
                f"{self.epochs}",
                f"{self.lr}",
                datetime.now().strftime("%b-%d_%H:%M:%S"),
            ]
        )

        os.makedirs(f"{os.getcwd()}/logs/Network_Slimming", exist_ok=True)
        os.makedirs(self.save, exist_ok=True)
        self.logger_file = f"{os.getcwd()}/logs/Network_Slimming/{self.name}.log"

        logging.basicConfig(
            filename=self.logger_file,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

        logger.info(f"Experiment Arguments: {self.kwargs}")

        if self.wandb_monitor:
            wandb.init(project="Trailmet Network_Slimming", name=self.name)
            wandb.config.update(self.kwargs)

    def compress_model(self) -> None:
        """Template function to be overwritten for each model compression method"""
        if self.ft_only:
            print("Error")
            return 0
        model = get_resnet_model(
            self.net, self.num_classes, insize=32, pretrained=False
        )
        model = self.base_train(model, self.dataloaders, fine_tune=False)
        self.lr = self.fine_tune_lr
        pruner = SlimPruner(model, self.prune_schema)
        pruning_result = pruner.run(self.prune_ratio)
        summary_model(pruner.pruned_model)
        pruned_model = pruner.pruned_model
        self.pr = pruning_result
        pruned_model.is_pruned = True
        del model
        pruned_model = self.base_train(pruned_model, self.dataloaders, fine_tune=True)

    def base_train(self, model, dataloaders, fine_tune=False):
        """
        This function is used to perform standard model training and can be used for various purposes, such as model
        pretraining, fine-tuning of compressed models, among others. For cases, where base_train is not directly
        applicable, feel free to overwrite wherever this parent class is inherited.
        """
        num_epochs = self.fine_tune_epochs if fine_tune else self.epochs
        best_top1_acc = 0  # setting to lowest possible value
        lr = self.lr if fine_tune else self.fine_tune_lr
        scheduler_type = self.scheduler_type
        weight_decay = self.weight_decay

        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        ###########################
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        epochs_list = []
        val_top1_acc_list = []
        val_top5_acc_list = []

        for epoch in range(num_epochs):
            adjust_learning_rate(optimizer, epoch, num_epochs, scheduler_type, lr)

            t_loss = self.train_one_epoch(
                model,
                dataloaders["train"],
                criterion,
                optimizer,
                epoch,
            )

            if self.sparsity_train:
                update_bn_grad(model)

            valid_loss, valid_top1_acc, valid_top5_acc = self.test(
                model, dataloaders["val"], criterion, epoch
            )

            is_best = False
            if valid_top1_acc > best_top1_acc:
                best_top1_acc = valid_top1_acc
                is_best = True

            if self.pr is not None:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "best_top1_acc": best_top1_acc,
                        "optimizer": optimizer.state_dict(),
                        "pruning_key": self.pr,
                    },
                    is_best,
                    self.save,
                )
            else:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "best_top1_acc": best_top1_acc,
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best,
                    self.save,
                )

            if self.wandb_monitor:
                wandb.log({"best_top1_acc": best_top1_acc})

            epochs_list.append(epoch)
            val_top1_acc_list.append(valid_top1_acc.cpu().numpy())
            val_top5_acc_list.append(valid_top5_acc.cpu().numpy())

            df_data = np.array(
                [
                    epochs_list,
                    val_top1_acc_list,
                    val_top5_acc_list,
                ]
            ).T
            df = pd.DataFrame(
                df_data,
                columns=[
                    "Epochs",
                    "Validation Top1",
                    "Validation Top5",
                ],
            )
            df.to_csv(
                f"{os.getcwd()}/logs/Network_Slimming/{self.name}.csv",
                index=False,
            )

        state = torch.load(f"{self.save}/model_best.pth.tar")
        model.load_state_dict(state["state_dict"], strict=True)
        return model

    def train_one_epoch(
        self, model, dataloader, loss_fn, optimizer, epoch, extra_functionality=None
    ):
        """standard training loop which can be used for various purposes with an extra functionality function to add to its working at the end of the loop."""
        model.train()
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")

        end = time.time()

        epoch_iterator = tqdm(
            dataloader,
            desc="Training Epoch [X] (X / X Steps) (batch time=X.Xs) (data time=X.Xs) (loss=X.X)",
            bar_format="{l_bar}{r_bar}",
            dynamic_ncols=True,
            disable=False,
        )

        for i, (images, labels) in enumerate(epoch_iterator):
            data_time.update(time.time() - end)
            images = images.to(device=self.device)
            labels = labels.to(device=self.device)
            scores = model(images)

            loss = loss_fn(scores, labels)

            n = images.size(0)
            losses.update(loss.item(), n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            epoch_iterator.set_description(
                "Training Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (data time=%2.5fs) (loss=%2.5f)"
                % (
                    epoch,
                    (i + 1),
                    len(dataloader),
                    batch_time.val,
                    data_time.val,
                    losses.val,
                )
            )

            logger.info(
                "Training Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (data time=%2.5fs) (loss=%2.5f)"
                % (
                    epoch,
                    (i + 1),
                    len(dataloader),
                    batch_time.val,
                    data_time.val,
                    losses.val,
                )
            )

            if self.wandb_monitor:
                wandb.log(
                    {
                        "train_loss": losses.val,
                    }
                )

            if extra_functionality is not None:
                extra_functionality()

        return losses.avg

    def test(self, model, dataloader, loss_fn, epoch):
        """This method is used to test the performance of the trained model."""
        model.eval()

        batch_time = AverageMeter("Time", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")
        top1 = AverageMeter("Acc@1", ":6.2f")
        top5 = AverageMeter("Acc@5", ":6.2f")

        epoch_iterator = tqdm(
            dataloader,
            desc="Validating Epoch [X] (X / X Steps) (batch time=X.Xs) (loss=X.X) (top1=X.X) (top5=X.X)",
            bar_format="{l_bar}{r_bar}",
            dynamic_ncols=True,
            disable=False,
        )
        with torch.no_grad():
            end = time.time()

            for i, (images, labels) in enumerate(epoch_iterator):
                images = images.to(device=self.device)
                labels = labels.to(device=self.device)
                scores = model(images)
                loss = loss_fn(scores, labels)

                pred1, pred5 = accuracy(scores, labels, topk=(1, 5))

                n = images.size(0)
                losses.update(loss.item(), n)
                top1.update(pred1[0], n)
                top5.update(pred5[0], n)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                epoch_iterator.set_description(
                    "Validating Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (loss=%2.5f) (top1=%2.5f) (top5=%2.5f)"
                    % (
                        epoch,
                        (i + 1),
                        len(dataloader),
                        batch_time.val,
                        losses.val,
                        top1.val,
                        top5.val,
                    )
                )

                logger.info(
                    "Validating Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (loss=%2.5f) (top1=%2.5f) (top5=%2.5f)"
                    % (
                        epoch,
                        (i + 1),
                        len(dataloader),
                        batch_time.val,
                        losses.val,
                        top1.val,
                        top5.val,
                    )
                )

                if self.wandb_monitor:
                    wandb.log(
                        {
                            "val_loss": losses.val,
                            "val_top1_acc": top1.val,
                            "val_top5_acc": top5.val,
                        }
                    )

            print(
                " * acc@1 {top1.avg:.3f} acc@5 {top5.avg:.3f}".format(
                    top1=top1, top5=top5
                )
            )

        return losses.avg, top1.avg, top5.avg
