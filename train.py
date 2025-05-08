from abc import ABC, abstractmethod


class Trainer(ABC):
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.args = None 
    
    
    def train_loop(self):
        """implements the training loop here."""
        raise NotImplementedError("train_loop must be implemented by the user.")

    def setup(self):
        """
        (Re)build all components that depend on hyperparameters in `self.args`.

        This includes:
          1. Downloading/preparing datasets and wrapping them in DataLoaders
          2. Instantiating the model architecture on the correct device
          3. Creating the optimizer (and any learning‐rate scheduler)
          4. Setting up the loss function and any other training utilities
        """
        pass
   
    def evaluate(self) -> dict:
        """ implements the evaluation here and it must return dictionary."""
        pass
    
    
   

     

        





