from Models.base_history import _BaseHistory, _BaseHistoryItem


class B2History(_BaseHistory):
    def __init__(self, input_path: str = None):
        """
        Initializes the B2 training history.

        Args:
            input_path (str): Path to a saved history file, if any.
        """
        super().__init__(input_path)

    def plot_history(self):
        """
        Optional: Implement this if you want to visualize training/validation curves.
        Currently left empty for compatibility.
        """
        pass


class B2HistoryItem(_BaseHistoryItem):
    def __init__(
            self,
            epoch: int,
            train_loss: float,
            train_acc: float,
            val_loss: float,
            val_acc: float
    ):
        super().__init__(epoch)
        self.train_loss = train_loss
        self.train_acc = train_acc
        self.val_loss = val_loss
        self.val_acc = val_acc

    def to_dict(self) -> dict[str, object]:
        """
        Converts the item to a dictionary for logging or saving.

        Returns:
            dict: Dictionary of metrics.
        """
        return {
            'epoch': self.epoch,
            'train-loss': self.train_loss,
            'train-acc': self.train_acc,
            'val-loss': self.val_loss,
            'val-acc': self.val_acc,
        }

    def __str__(self):

        return f"\nTrain Loss: {self.train_loss:.3f} - Train Acc: {self.train_acc:.2f}% - Val Loss: {self.val_loss:.3f} - Val Acc: {self.val_acc:.2f}%\n"