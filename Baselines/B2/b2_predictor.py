import torch
from Models.base_predictor import _BasePredictor
from Models.base_model import BaseModel

class B2Predictor(_BasePredictor):
    def __init__(self, model_path: str = None):
        """
        Initializes the predictor for B2 model inference.
        Loads the trained model for performing predictions.
        """
        print("Initializing B2Predictor...")  
        super().__init__(model_path)
        print("B2Predictor initialized successfully.")  

    def _load_model(self):
        """
        Loads the trained model from the specified path.
        """
        print(f"Loading model from {self._model_path}...")  
        self.model: BaseModel = torch.load(self._model_path)
        print("Model loaded successfully.")  
