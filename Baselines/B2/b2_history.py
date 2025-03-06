from Models.base_history import _BaseHistory, _BaseHistoryItem

class B2History(_BaseHistory):
    def __init__(self, input_path: str = None):
        """
        Initializes the history tracking for B2 model training.
        """
        print("Initializing B2History...") 
        super().__init__(input_path)
        print("B2History initialized successfully.")  

    def plot_history(self):
        """
        Placeholder for history visualization.
        """
        print("Plotting training history...")  
        pass  


class B2HistoryItem(_BaseHistoryItem):
    def __init__(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
    ):
        """
        Represents an individual training history record for B2 model.
        """
        super().__init__(epoch)
        self.train_loss = train_loss
        self.train_acc = train_acc
        self.val_loss = val_loss
        self.val_acc = val_acc
        print(f"History item created for epoch {epoch}.")  

    def to_dict(self) -> dict[str, object]:
        """
        Converts the history item into a dictionary format.
        """
        print("Converting history item to dictionary...") 
        return {
            'epoch': self.epoch,
            'train-loss': self.train_loss,
            'train-acc': self.train_acc,
            'val-loss': self.val_loss,
            'val-acc': self.val_acc,
        }

    def __str__(self):
        """
        Returns a string representation of the history item.
        """
        return f"\nTrain Loss: {self.train_loss:.3f} - Train Acc: {self.train_acc:.2f}% - Val Loss: {self.val_loss:.3f} - Val Acc: {self.val_acc:.2f}%\n"
