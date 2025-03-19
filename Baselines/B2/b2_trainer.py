import torch
from torch import nn
from torchvision import models
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from Baselines.B2.b2_checkpoint import B2Checkpoint
from Baselines.B2.b2_history import B2History, B2HistoryItem
from Models.base_trainer import _BaseTrainer
from Models.base_model import BaseModel
from Models.player_dataset import PlayerDataset
from Enums.classification_level import ClassificationLevel
from Utils.cuda import get_device


class B2Trainer(_BaseTrainer):
    """
    B2 baseline trainer for single player classification using cropped images.
    """

    def __init__(self, checkpoint_path: str = None, history_path: str = None):
        super().__init__(checkpoint_path, history_path)

    def _init_values(self):
        self.train_loss, self.train_correct, self.train_total = 0.0, 0, 0
        self.val_loss, self.val_correct, self.val_total = 0.0, 0, 0
        self.test_loss, self.test_correct, self.test_total = 0.0, 0, 0

    def _get_dataset_type(self):
        return PlayerDataset

    def _prepare_model(self):
        self.model = BaseModel(
            backbone=models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
            level=ClassificationLevel.PLAYER
        ).set_backbone_requires_grad(False) \
         .set_backbone_layer_requires_grad('layer4', True) \
         .set_backbone_layer_requires_grad('fc', True)

    def _prepare_optimizer(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=self.get_bl_cf().training.learning_rate
        )
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=2
        )

    def _to_available_device(self):
        self.model.to_available_device()
        self.criterion.to(get_device())
        for state in self.optimizer.state.values():
            if isinstance(state, torch.Tensor):
                state.data = state.data.to(get_device())

    def _get_checkpoint(self, checkpoint_path: str = None) -> B2Checkpoint:
        return B2Checkpoint(
            input_path=checkpoint_path,
            epoch=0,
            model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict()
        )

    def _get_history(self, history_path: str = None) -> B2History:
        return B2History(history_path)

    def _get_train_loader(self):
        return self.train_loader

    def _get_val_loader(self):
        return self.val_loader

    def _get_test_loader(self):
        return self.test_loader

    def _train_mode(self):
        self.model.train()

    def _eval_mode(self):
        self.model.eval()

    def _on_checkpoint_load(self) -> int:
        checkpoint: B2Checkpoint = self._checkpoint
        self.model.load_state_dict(checkpoint.model_state)
        self.optimizer.load_state_dict(checkpoint.optimizer_state)
        self.scheduler.load_state_dict(checkpoint.scheduler_state)

    def _train_batch_step(self, inputs, labels):
        self.optimizer.zero_grad()
        logits, _ = self.model(inputs)
        logits = logits.squeeze(1)

        loss = self.criterion(logits, labels.long())
        loss.backward()
        self.optimizer.step()

        self.train_loss += loss.item()
        _, predicted = logits.max(1)
        self.train_correct += (predicted == labels).sum().item()
        self.train_total += labels.size(0)

    def _eval_batch_step(self, inputs, labels):
        logits, _ = self.model(inputs)
        logits = logits.squeeze(1)

        loss = self.criterion(logits, labels.long())
        self.val_loss += loss.item()

        _, predicted = logits.max(1)
        self.val_correct += (predicted == labels).sum().item()
        self.val_total += labels.size(0)

    def _test_batch_step(self, inputs, labels):
        logits, _ = self.model(inputs)
        logits = logits.squeeze(1)

        loss = self.criterion(logits, labels.long())
        self.test_loss += loss.item()

        _, predicted = logits.max(1)
        self.test_correct += (predicted == labels).sum().item()
        self.test_total += labels.size(0)

    def _on_epoch_step(self, epoch: int):
        self.scheduler.step(self.val_loss)
        self._checkpoint.update_state(
            epoch=epoch,
            model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict()
        )
        return B2HistoryItem(
            epoch,
            self.train_loss / self.train_total if self.train_total > 0 else 0,
            100 * self.train_correct / self.train_total if self.train_total > 0 else 0,
            self.val_loss / self.val_total if self.val_total > 0 else 0,
            100 * self.val_correct / self.val_total if self.val_total > 0 else 0,
        )

    def _on_test_step(self):
        print(
            f"Test Results:\nLoss: {self.test_loss / self.test_total:.4f}, "
            f"Acc: {100 * self.test_correct / self.test_total:.2f}%"
        )

    def _save_trained_model(self):
        torch.save(self.model, self._model_path)