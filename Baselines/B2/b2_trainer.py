import torch
from Baselines.B2.b2_checkpoint import B2Checkpoint
from Baselines.B2.b2_history import B2History, B2HistoryItem
from Models.base_trainer import _BaseTrainer
from Enums.classification_level import ClassificationLevel
from Enums.dataset_type import DatasetType
from Models.base_model import BaseModel
from Models.dataset import PlayerDataset
from torch import nn
from torchvision import models
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from Utils.cuda import get_device

class B2Trainer(_BaseTrainer):
    """
    Training pipeline for the B2 baseline model.
    This class extends _BaseTrainer and implements specific logic for:
    - Data loading
    - Model preparation
    - Optimizer and scheduler configuration
    - Training loop
    - Evaluation and testing
    """

    def __init__(self, checkpoint_path: str = None, history_path: str = None):
        print("Initializing B2Trainer...")  
        super().__init__(checkpoint_path, history_path)
        print("B2Trainer initialized successfully.")  

    def _init_values(self):
        self.train_loss, self.train_correct, self.train_total = 0.0, 0, 0
        self.val_loss, self.val_correct, self.val_total = 0.0, 0, 0
        self.test_loss, self.test_correct, self.test_total = 0.0, 0, 0

    def _prepare_loaders(self):
        print("Preparing data loaders...")  
        train_dataset = PlayerDataset(type=DatasetType.TRAIN)
        val_dataset = PlayerDataset(type=DatasetType.VAL)
        test_dataset = PlayerDataset(type=DatasetType.TEST)

        batch_size = self.get_bl_cf().training.batch_size

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)
        print("Data loaders prepared successfully.")  

    def _prepare_model(self):
        print("Initializing model...")  
        self.model = BaseModel(
            backbone=models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
            level=ClassificationLevel.PLAYER
        ) \
            .set_backbone_requires_grad(False) \
            .set_backbone_layer_requires_grad('layer4', True) \
            .set_backbone_layer_requires_grad('fc', True)
        print("Model initialized successfully.")  

    def _prepare_optimizer(self):
        print("Preparing optimizer and loss function...")  
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=self.get_bl_cf().training.learning_rate
        )
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=2
        )
        print("Optimizer and scheduler prepared successfully.") 

    def _to_available_device(self):
        print("Moving model and optimizer to available device...")  
        self.model.to_available_device()
        self.criterion.to(get_device())
        for state in self.optimizer.state.values():
            if isinstance(state, torch.Tensor):
                state.data = state.data.to(get_device())
        print("Model moved to device successfully.")  

    def _get_checkpoint(self, checkpoint_path: str = None) -> B2Checkpoint:
        print("Initializing checkpoint...")  
        return B2Checkpoint(
            input_path=checkpoint_path,
            epoch=0,
            model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict()
        )

    def _get_history(self, history_path: str = None) -> B2History:
        print("Initializing training history...")  
        return B2History(history_path)

    def _train_mode(self):
        self.model.train()

    def _eval_mode(self):
        self.model.eval()

    def _on_checkpoint_load(self) -> int:
        checkpoint: B2Checkpoint = self._checkpoint
        self.model.load_state_dict(checkpoint.model_state)
        self.optimizer.load_state_dict(checkpoint.optimizer_state)
        self.scheduler.load_state_dict(checkpoint.scheduler_state)

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
            self.train_loss / len(self.train_loader),
            100 * self.train_correct / self.train_total,
            self.val_loss / len(self.val_loader),
            100 * self.val_correct / self.val_total,
        )

    def _save_trained_model(self):
        print("Saving trained model...")  
        torch.save(self.model, self._model_path)
        print("Model saved successfully.")  
