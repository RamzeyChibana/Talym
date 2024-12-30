import inspect
import torch
from utils import get_dict_keys , check_return_dict
from abc import ABC, abstractmethod



class Trainer(ABC):
    

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        
        # Check if the subclass implements `evaluate`
        if 'evaluate' in cls.__dict__:
            original_evaluate = cls.__dict__['evaluate']
            cls.evaluate = check_return_dict(original_evaluate)
        else:
            raise NotImplementedError(
                f"Subclass '{cls.__name__}' must implement the 'evaluate' method."
            )
    def setupd_data(self):
        """User implements or imports their model and optimizer here."""
        raise NotImplementedError("setup_data must be implemented by the user.")
    
    def setup_model(self):
        """User implements or imports their model here."""
        raise NotImplementedError("setup_model must be implemented by the user.")
    
    def train_loop(self):
        """User implements the training loop here."""
        raise NotImplementedError("train_loop must be implemented by the user.")
    
    @check_return_dict
    def evaluate(self) -> dict:
        """User implements the evaluation here and it must return dictionary."""
        raise NotImplementedError("evaluate must be implemented by the user.")
    
    
   
class test(Trainer):
    def evaluate(self, x):
        return {"accuracy": 0.95, "loss": 0.05}
     

        

ss = test()
print(ss.evaluate("ramzey"))
print(ss.evaluation_keys)

