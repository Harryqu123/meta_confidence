from multiprocessing import set_forkserver_preload
import os
from collections import OrderedDict
from re import L
from statistics import mode

import numpy as np
import torch
from tqdm import tqdm

from confidnet.learners.learner import AbstractLeaner
from confidnet.utils import misc
from confidnet.utils.logger import get_logger
from confidnet.utils.metrics import Metrics

import ipdb
import learn2learn as l2l
import glob
import shutil
from torch.utils.data import DataLoader
import time

class CIFAR10Meta(torch.utils.data.Dataset):
    def __init__(self, root):
        self.pths = glob.glob(root + '/*.pth')

    def __getitem__(self, index):
        data = torch.load(self.pths[index])
        label = int(self.pths[index].split('_')[-1].split('.')[0])
        return data, label

    def __len__(self):
        return len(self.pths)

LOGGER = get_logger(__name__, level="DEBUG")

def is_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

class SelfConfidLearnerMeta(AbstractLeaner):
    def __init__(self, config_args, train_loader, val_loader, test_loader, start_epoch, device):
        super().__init__(config_args, train_loader, val_loader, test_loader, start_epoch, device)
        self.freeze_layers()
        self.disable_bn(verbose=True)
        if self.config_args["model"].get("uncertainty", None):
            self.disable_dropout(verbose=True)

        self.lr_meta = config_args['training']['optimizer']['lr']
        self.model = l2l.algorithms.MAML(self.model, lr=self.lr_meta, first_order=True, allow_unused=None, allow_nograd=True)

        # ipdb.set_trace()

        self.meta_data_dir_lt = './data/' + str(config_args['training']['output_folder']).split('/')[-1]


        self.meta_loader_lt = None
        self.iterator_lt = None


    # In this project, we construct sets for correctness label C through the function below
    # This way of construction is slightly different from the way described in paper
    # We made this change during further exploration because, this way is more implementation-friendly while can still achieve competitive results during our testing.
    def generate_meta_lt(self):

        is_path(self.meta_data_dir_lt)
        shutil.rmtree(self.meta_data_dir_lt)
        is_path(self.meta_data_dir_lt)

        criterion_meta_lt = torch.nn.CrossEntropyLoss(reduction='none')
        meta_raw_lt = []
        for batch_id, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            with torch.no_grad():
                output = self.model(data)
                loss_meta_lt = criterion_meta_lt(output[0], target)
            for data_item, target_item, loss_meta_lt_item in zip(data, target, loss_meta_lt):
                meta_raw_lt.append([data_item.cpu(), target_item.cpu(), loss_meta_lt_item.cpu()])
        meta_raw_lt = sorted(meta_raw_lt, key=lambda x: x[2], reverse=True)
        for i, item in enumerate(meta_raw_lt[:int(len(meta_raw_lt) * 0.1)]):
            name_lt = '{:0>6d}_{}.pth'.format(i, item[1])
            torch.save(item[0], os.path.join(self.meta_data_dir_lt, name_lt))

        self.meta_loader_lt = DataLoader(CIFAR10Meta(self.meta_data_dir_lt), batch_size=self.train_loader.batch_size, shuffle=True)
        self.iterator_lt = iter(self.meta_loader_lt)

    def train(self, epoch):
        self.model.train()
        self.disable_bn()
        if self.config_args["model"].get("uncertainty", None):
            self.disable_dropout()
        
        metrics = Metrics(
            self.metrics, self.prod_train_len, self.num_classes
        )
        loss, confid_loss = 0, 0
        len_steps, len_data = 0, 0
        
        # Training loop
        loop = tqdm(self.train_loader)
        for batch_id, (data, target) in enumerate(loop):

            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)

            # Potential temperature scaling
            if self.temperature:
                output = list(output)
                output[0] = output[0] / self.temperature
                output = tuple(output)

            if self.task == "classification":
                current_loss = self.criterion(output, target)
            elif self.task == "segmentation":
                current_loss = self.criterion(output, target.squeeze(dim=1))
            
            # We perform the meta learning process every several iterations.
            # This is because, during our experiments, we find that, incorporating meta learning in every iteration can instablize the optimization process
            # This finding is also somehow consistent with others' finding (e.g., Sharp-MAML: Sharpness-Aware Model-Agnostic Meta Learning published in ICML 2022)
            if batch_id % 5 == 0:
                model_clone = self.model.clone()
                model_clone.adapt(current_loss)
                self.optimizer.zero_grad()
            
                try:
                    inner_batch_data, inner_batch_target = next(self.iterator_lt)
                except:
                    del self.iterator_lt
                    self.iterator_lt = iter(self.meta_loader_lt)
                    inner_batch_data, inner_batch_target = next(self.iterator_lt)

                inner_batch_data = inner_batch_data.to(self.device)
                inner_batch_target = inner_batch_target.to(self.device)

                output_clone = model_clone(inner_batch_data)

                inner_loss_oracle_lt = self.criterion(output_clone, inner_batch_target)

                del model_clone

                weight = np.random.dirichlet(np.ones(2),size=1)[0]
                loss_total = current_loss * weight[0] + inner_loss_oracle_lt * weight[1]            
                loss_total.backward()

            else:
                current_loss.backward()

            loss += current_loss
            self.optimizer.step()
            if self.task == "classification":
                len_steps += len(data)
                len_data = len_steps
            elif self.task == "segmentation":
                len_steps += len(data) * np.prod(data.shape[-2:])
                len_data += len(data)

            # Update metrics
            pred = output[0].argmax(dim=1, keepdim=True)
            confidence = torch.sigmoid(output[1])
            metrics.update(pred, target, confidence)

            # Update the average loss
            loop.set_description(f"Epoch {epoch}/{self.nb_epochs}")
            loop.set_postfix(
                OrderedDict(
                    {
                        "loss_confid": f"{(loss / len_data):05.3e}",
                        "acc": f"{(metrics.accuracy / len_steps):05.2%}",
                    }
                )
            )
            loop.update()

        scores = metrics.get_scores(split="train")
        logs_dict = OrderedDict(
            {
                "epoch": {"value": epoch, "string": f"{epoch:03}"},
                "lr": {
                    "value": self.optimizer.param_groups[0]["lr"],
                    "string": f"{self.optimizer.param_groups[0]['lr']:05.1e}",
                },
                "train/loss_confid": {
                    "value": loss / len_data,
                    "string": f"{(loss / len_data):05.4e}",
                },
            }
        )
        for s in scores:
            logs_dict[s] = scores[s]

        # Val scores
        val_losses, scores_val = self.evaluate(self.val_loader, self.prod_val_len, split="val")
        logs_dict["val/loss_confid"] = {
            "value": val_losses["loss_confid"].item() / self.nsamples_val,
            "string": f"{(val_losses['loss_confid'].item() / self.nsamples_val):05.4e}",
        }
        for sv in scores_val:
            logs_dict[sv] = scores_val[sv]

        # Test scores
        test_losses, scores_test = self.evaluate(self.test_loader, self.prod_test_len, split="test")
        logs_dict["test/loss_confid"] = {
            "value": test_losses["loss_confid"].item() / self.nsamples_test,
            "string": f"{(test_losses['loss_confid'].item() / self.nsamples_test):05.4e}",
        }
        for st in scores_test:
            logs_dict[st] = scores_test[st]

        # Print metrics
        misc.print_dict(logs_dict)

        # Save the model checkpoint
        self.save_checkpoint(epoch)

        # CSV logging
        misc.csv_writter(path=self.output_folder / "logs.csv", dic=OrderedDict(logs_dict))

        # Tensorboard logging
        self.save_tb(logs_dict)

        # Scheduler step
        if self.scheduler:
            self.scheduler.step()
        # ipdb.set_trace()

    def evaluate(self, dloader, len_dataset, split="test", verbose=False, **args):
        # ipdb.set_trace()
        self.model.eval()
        metrics = Metrics(self.metrics, len_dataset, self.num_classes)
        loss = 0

        # Evaluation loop
        loop = tqdm(dloader, disable=not verbose)
        for batch_id, (data, target) in enumerate(loop):
            data, target = data.to(self.device), target.to(self.device)

            with torch.no_grad():
                output = self.model(data)
                if self.task == "classification":
                    loss += self.criterion(output, target)
                elif self.task == "segmentation":
                    loss += self.criterion(output, target.squeeze(dim=1))
                # Update metrics
                pred = output[0].argmax(dim=1, keepdim=True)
                confidence = torch.sigmoid(output[1])
                metrics.update(pred, target, confidence)

        scores = metrics.get_scores(split=split)
        losses = {"loss_confid": loss}
        return losses, scores

    def load_checkpoint(self, state_dict_bkp, uncertainty_state_dict=None, strict=True):
        
        state_dict = OrderedDict()
        for k, v in state_dict_bkp.items():
            k = 'module.' + k
            state_dict[k] = v
        # ipdb.set_trace()

        if not uncertainty_state_dict:
            self.model.load_state_dict(state_dict, strict=strict)
        else:
            self.model.pred_network.load_state_dict(state_dict_bkp, strict=strict)

            # 1. filter out unnecessary keys
            if self.task == "classification":
                state_dict = {
                    k: v
                    for k, v in uncertainty_state_dict.items()
                    if k not in ["fc2.weight", "fc2.bias"]
                }
            if self.task == "segmentation":
                state_dict = {
                    k: v
                    for k, v in uncertainty_state_dict.items()
                    if k
                    not in [
                        "up1.conv2.cbr_unit.0.weight",
                        "up1.conv2.cbr_unit.0.bias",
                        "up1.conv2.cbr_unit.1.weight",
                        "up1.conv2.cbr_unit.1.bias",
                        "up1.conv2.cbr_unit.1.running_mean",
                        "up1.conv2.cbr_unit.1.running_var",
                    ]
                }
            
            # 2. overwrite entries in the existing state dict
            self.model.uncertainty_network.state_dict().update(state_dict)
            # 3. load the new state dict
            self.model.uncertainty_network.load_state_dict(state_dict, strict=False)

    def load_checkpoint_bkp(self, state_dict, uncertainty_state_dict=None, strict=True):
        if not uncertainty_state_dict:
            self.model.load_state_dict(state_dict, strict=strict)
        else:
            self.model.pred_network.load_state_dict(state_dict, strict=strict)

            # 1. filter out unnecessary keys
            if self.task == "classification":
                state_dict = {
                    k: v
                    for k, v in uncertainty_state_dict.items()
                    if k not in ["fc2.weight", "fc2.bias"]
                }
            if self.task == "segmentation":
                state_dict = {
                    k: v
                    for k, v in uncertainty_state_dict.items()
                    if k
                    not in [
                        "up1.conv2.cbr_unit.0.weight",
                        "up1.conv2.cbr_unit.0.bias",
                        "up1.conv2.cbr_unit.1.weight",
                        "up1.conv2.cbr_unit.1.bias",
                        "up1.conv2.cbr_unit.1.running_mean",
                        "up1.conv2.cbr_unit.1.running_var",
                    ]
                }
            # 2. overwrite entries in the existing state dict
            self.model.uncertainty_network.state_dict().update(state_dict)
            # 3. load the new state dict
            self.model.uncertainty_network.load_state_dict(state_dict, strict=False)

    def freeze_layers(self):
        # Eventual fine-tuning for self-confid
        LOGGER.info("Freezing every layer except uncertainty")
        for param in self.model.named_parameters():
            if "uncertainty" in param[0]:
                print(param[0], "kept to training")
                continue
            param[1].requires_grad = False

    def disable_bn(self, verbose=False):
        # Freeze also BN running average parameters
        if verbose:
            LOGGER.info("Keeping original BN parameters")
        for layer in self.model.named_modules():
            if "bn" in layer[0] or "cbr_unit.1" in layer[0]:
                if verbose:
                    print(layer[0], "original BN setting")
                layer[1].momentum = 0
                layer[1].eval()

    def disable_dropout(self, verbose=False):
        # Freeze also BN running average parameters
        if verbose:
            LOGGER.info("Disable dropout layers to reduce stochasticity")
        for layer in self.model.named_modules():
            if "dropout" in layer[0]:
                if verbose:
                    print(layer[0], "set to eval mode")
                layer[1].eval()
