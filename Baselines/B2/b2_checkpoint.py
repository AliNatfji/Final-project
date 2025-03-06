from typing import Any
from Models.base_checkpoint import _BaseCheckpoint


class B2Checkpoint(_BaseCheckpoint):
    def __init__(
        self,
        input_path: str = None,
        epoch=0,
        model_state: dict[str, Any] = {},
        optimizer_state: dict[str, Any] = {},
        scheduler_state: dict[str, Any] = {},
        feature_extractor_state: dict[str, Any] = {},
    ):
        """
        Initializes the checkpoint for B2 model training.
        Stores model state, optimizer state, scheduler state, and feature extractor state.
        """
        print("Initializing B2Checkpoint...")  
        super().__init__(
            input_path=input_path,
            epoch=epoch,
            model_state=model_state,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state
        )
        self.feature_extractor_state = feature_extractor_state
        print(f"Checkpoint initialized with epoch {epoch}") 

    def update_state(
        self,
        epoch=0,
        model_state: dict[str, Any] = {},
        optimizer_state: dict[str, Any] = {},
        scheduler_state: dict[str, Any] = {},
        feature_extractor_state: dict[str, Any] = {},
    ):
        """
        Updates the state of the checkpoint.
        This includes the model's weights, optimizer, scheduler, and feature extractor state.
        """
        print(f"Updating checkpoint state at epoch {epoch}...")  
        self.epoch = epoch
        self.model_state = model_state
        self.optimizer_state = optimizer_state
        self.scheduler_state = scheduler_state
        self.feature_extractor_state = feature_extractor_state
        print("Checkpoint state updated successfully.")

    def _get_state_dict(self):
        """
        Returns a dictionary containing all the stored state information.
        This is used for saving the checkpoint.
        """
        print("Fetching checkpoint state dictionary...")  
        return {
            'epoch': self.epoch,
            'model_state': self.model_state,
            'optimizer_state': self.optimizer_state,
            'scheduler_state': self.scheduler_state,
            'feature_extractor_state': self.feature_extractor_state,
        }
