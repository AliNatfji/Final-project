from abc import ABC, abstractmethod
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from Models.base_checkpoint import _BaseCheckpoint
from Models.config_mixin import _ConfigMixin
from Models.base_history import _BaseHistory, _BaseHistoryItem
from Utils.cuda import get_device


class _BaseTrainer(_ConfigMixin, ABC):
    """
    Abstract base class for training pipelines.
    It provides a consistent workflow for training, evaluation, and testing.
    """

    def __init__(self, checkpoint_path: str = None, history_path: str = None):
        """Initializes the training pipeline by preparing loaders, model, and optimizer."""
        self._init_values()
        self._prepare_loaders()
        self._prepare_model()
        self._prepare_optimizer()
        self._to_available_device()

        self._checkpoint = self._get_checkpoint(checkpoint_path)
        self._history = self._get_history(history_path)
        self._model_path = os.path.join(
            self.get_bl_cf().output_dir, 'model.pth')

    @abstractmethod
    def _init_values(self) -> None:
        pass

    @abstractmethod
    def _prepare_loaders(self) -> None:
        """
        Prepares DataLoaders for training, validation, and testing.

        - Loads ImageDataset for train, val, and test sets.
        - Initializes DataLoaders with the configured batch size."""
        pass

    @abstractmethod
    def _prepare_model(self) -> None:
        """
        Prepares model for training."""
        pass

    @abstractmethod
    def _prepare_optimizer(self) -> None:
        """
        Prepares optimizer and configure it."""
        pass

    @abstractmethod
    def _to_available_device(self) -> None:
        pass

    @abstractmethod
    def _get_checkpoint(self, checkpoint_path: str = None) -> _BaseCheckpoint:
        pass

    @abstractmethod
    def _get_history(self, history_path: str = None) -> _BaseHistory:
        pass

    @abstractmethod
    def _get_train_loader(self) -> DataLoader:
        """
        Retrieves the DataLoader for training data.

        Returns:
            DataLoader: DataLoader for training data.
        """
        pass

    @abstractmethod
    def _get_val_loader(self) -> DataLoader:
        """
        Retrieves the DataLoader for validation data.

        Returns:
            DataLoader: DataLoader for validation data.
        """
        pass

    @abstractmethod
    def _get_test_loader(self) -> DataLoader:
        """
        Retrieves the DataLoader for testing data.

        Returns:
            DataLoader: DataLoader for testing data.
        """
        pass

    @abstractmethod
    def _train_mode(self) -> None:
        """Sets baseline model to training mode."""
        pass

    @abstractmethod
    def _eval_mode(self) -> None:
        """Sets baseline model to evaluation mode."""
        pass

    @abstractmethod
    def _train_batch_step(self, inputs, labels) -> None:
        """
        Defines a single training step.

        - Clears the gradients.
        - Performs a forward pass to get predictions.
        - Computes the loss.
        - Backpropagates the loss and updates model weights.
        - Calculates training accuracy.

        Args:
            inputs (torch.Tensor): Input images.
            labels (torch.Tensor): Ground truth labels.
        """
        pass

    @abstractmethod
    def _eval_batch_step(self, inputs, labels) -> None:
        """
        Defines a single evaluation step.

        - Performs a forward pass to get predictions.
        - Computes the loss.
        - Calculates validation accuracy.

        Args:
            inputs (torch.Tensor): Input images.
            labels (torch.Tensor): Ground truth labels.
        """
        pass

    @abstractmethod
    def _on_epoch_step(self, epoch: int) -> _BaseHistoryItem:
        """A callback function emitted after each epoch."""
        pass

    @abstractmethod
    def _test_batch_step(self, inputs, labels) -> None:
        """
        Defines a single testing step.

        - Performs a forward pass to get predictions.
        - Computes the loss.
        - Calculates test accuracy.

        Args:
            inputs (torch.Tensor): Input images.
            labels (torch.Tensor): Ground truth labels.
        """
        pass

    @abstractmethod
    def _on_test_step(self) -> None:
        pass

    @abstractmethod
    def _on_checkpoint_load(self) -> int:
        pass

    @abstractmethod
    def _save_trained_model(self):
        pass

    def train(self, override=False):
        self.get_bl_cf().create_baseline_dir()
        self.clear_output()

        if override:
            self._history.reset()
            self._checkpoint.reset()
        else:
            try:
                self._history.load(from_input=True)
                self._checkpoint.load(from_input=True)
                self._on_checkpoint_load()
            except:
                print('Loading Checkpoint failed, Training started from begining')
                self._history.reset()
                self._checkpoint.reset()

        self._to_available_device()

        epochs = self.get_bl_cf().training.epochs
        for epoch in range(self._checkpoint.epoch, epochs):
            self._train_mode()

            progress_bar = tqdm(self._get_train_loader(),
                                desc=f"Epoch {epoch+1}/{epochs}", leave=True)

            for batch_idx, (inputs, labels) in enumerate(progress_bar):
                inputs, labels = inputs.to(
                    get_device()), labels.to(get_device())

                self._train_batch_step(inputs, labels)

            self.evaluate()
            history_item = self._on_epoch_step(epoch)
            self._history.add(history_item)
            self._checkpoint.save()
            self._init_values()

        self._save_trained_model()



    def evaluate(self):
        if self._get_val_loader() is None:
            print("[WARNING] Skipping evaluation because validation dataset is empty!")
            return  # Exit function early if no validation data

        self._eval_mode()
        with torch.no_grad():
            for inputs, labels in self._get_val_loader():  # This used to fail
                self._eval_batch_step(inputs, labels)          

    def test(self):
        if self._get_test_loader() is None:
            print("[WARNING] Skipping testing because test dataset is empty!")
            return  # Exit function early if no test data
        
        self._eval_mode()
        self._init_values()
        self._to_available_device()

        with torch.no_grad():
            for inputs, labels in self._get_test_loader():
                inputs, labels = inputs.to(
                    get_device()), labels.to(get_device())
                self._test_batch_step(inputs, labels)
            self._on_test_step()
