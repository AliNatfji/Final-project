import torch
from Baselines.B1.b1_checkpoint import B1Checkpoint
from Baselines.B1.b1_history import B1History, B1HistoryItem
from Models.base_trainer import _BaseTrainer
from Enums.classification_level import ClassificationLevel
from Enums.dataset_type import DatasetType
from Models.base_model import BaseModel
from Models.image_dataset import ImageDataset
from torch import nn
from torchvision import models
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from Utils.cuda import get_device


class B1Trainer(_BaseTrainer):
    """
    Training pipeline for the B1 baseline model.
    """

    def __init__(self, checkpoint_path: str = None, history_path: str = None):
        super().__init__(checkpoint_path, history_path)

    def _init_values(self):
        """ Initialize loss and accuracy tracking variables """
        self.train_loss, self.train_correct, self.train_total = 0.0, 0, 0
        self.val_loss, self.val_correct, self.val_total = 0.0, 0, 0
        self.test_loss, self.test_correct, self.test_total = 0.0, 0, 0

    def _prepare_loaders(self):
        """ Prepare data loaders for train, validation, and test sets """

        train_dataset = ImageDataset(type=DatasetType.TRAIN)
        val_dataset = ImageDataset(type=DatasetType.VAL)
        test_dataset = ImageDataset(type=DatasetType.TEST)

        # Store dataset sizes
        self.train_size = len(train_dataset)
        self.val_size = len(val_dataset)
        self.test_size = len(test_dataset)

        batch_size = self.get_bl_cf().training.batch_size

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False) if self.val_size > 0 else None
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False) if self.test_size > 0 else None

        # Print dataset sizes for debugging
        print(f"[INFO] Train dataset size: {self.train_size}")
        print(f"[INFO] Validation dataset size: {self.val_size}")
        print(f"[INFO] Test dataset size: {self.test_size}")

        # Warning if validation/test set is empty
        if self.val_size == 0:
            print("[WARNING] Validation dataset is empty! Model won't be evaluated on validation.")
        if self.test_size == 0:
            print("[WARNING] Test dataset is empty! Model won't be evaluated on test.")

    def _prepare_model(self):
        """ Initialize and configure the model """
        self.model = BaseModel(
            backbone=models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
            level=ClassificationLevel.IMAGE
        ) \
            .set_backbone_requires_grad(False) \
            .set_backbone_layer_requires_grad('layer4', True) \
            .set_backbone_layer_requires_grad('fc', True)

    def _prepare_optimizer(self):
        """ Configure optimizer and loss function """
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=self.get_bl_cf().training.learning_rate
        )
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=2
        )

    def _to_available_device(self):
        """ Move model and tensors to available CUDA device """
        self.model.to_available_device()
        self.criterion.to(get_device())
        for state in self.optimizer.state.values():
            if isinstance(state, torch.Tensor):
                state.data = state.data.to(get_device())

    def _get_checkpoint(self, checkpoint_path: str = None) -> B1Checkpoint:
        return B1Checkpoint(
            input_path=checkpoint_path,
            epoch=0,
            model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict()
        )

    def _get_history(self, history_path: str = None) -> B1History:
        return B1History(history_path)

    def _get_train_loader(self):
        return self.train_loader

    def _get_val_loader(self):
        return self.val_loader if self.val_loader else None

    def _get_test_loader(self):
        return self.test_loader if self.test_loader else None

    def _train_mode(self):
        self.model.train()

    def _eval_mode(self):
        self.model.eval()

    def _on_checkpoint_load(self) -> int:
        checkpoint: B1Checkpoint = self._checkpoint
        self.model.load_state_dict(checkpoint.model_state)
        self.optimizer.load_state_dict(checkpoint.optimizer_state)
        self.scheduler.load_state_dict(checkpoint.scheduler_state)

    def _on_epoch_step(self, epoch: int):
        """ Handle the logic of what happens after an epoch """

        self.scheduler.step(self.val_loss)

        self._checkpoint.update_state(
            epoch=epoch,
            model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict()
        )

        # Handling ZeroDivisionError in case val_size or val_total is zero
        return B1HistoryItem(
            epoch,
            self.train_loss / self.train_size if self.train_size > 0 else 0,
            100 * self.train_correct / self.train_total if self.train_total > 0 else 0,
            self.val_loss / self.val_size if self.val_size > 0 else 0,
            100 * self.val_correct / self.val_total if self.val_total > 0 else 0,
        )

    def _save_trained_model(self):
        """ Save the trained model to disk """
        torch.save(self.model, self._model_path)

    def _train_batch_step(self, inputs, labels):
        """ Training step for each batch """

        self.optimizer.zero_grad()
        outputs = self.__map_outputs(self.model(inputs))
        loss = self.criterion(outputs, labels)

        loss.backward()
        self.optimizer.step()

        self.train_loss += loss.item()

        _, predicted = outputs.max(1)
        self.train_correct += (predicted == labels).sum().item()
        self.train_total += labels.size(0)

    def _eval_batch_step(self, inputs, labels):
        """ Validation step for each batch """

        if self.val_loader is None:
            return  # Skip if no validation dataset

        outputs = self.__map_outputs(self.model(inputs))
        loss = self.criterion(outputs, labels)

        self.val_loss += loss.item()

        _, predicted = outputs.max(1)
        self.val_correct += (predicted == labels).sum().item()
        self.val_total += labels.size(0)

    def _test_batch_step(self, inputs, labels):
        """ Testing step for each batch """

        if self.test_loader is None:
            return  # Skip if no test dataset

        outputs = self.__map_outputs(self.model(inputs))
        loss = self.criterion(outputs, labels)

        self.test_loss += loss.item()

        _, predicted = outputs.max(1)
        self.test_correct += (predicted == labels).sum().item()
        self.test_total += labels.size(0)

    def _on_test_step(self):
        """ Print final test results """

        if self.test_size > 0:
            print(
                f"Test Results:\nLoss: {self.test_loss/self.test_size:.4f}, "
                f"Acc: {100 * self.test_correct/self.test_total:.2f}%"
            )
        else:
            print("[INFO] No test data available. Skipping test phase.")

    def __map_outputs(self, outputs):
        """ Reshape model outputs """
        batch_size, _, __ = outputs.shape
        return outputs.view(batch_size, -1)