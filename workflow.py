import torch 
from train import Trainer
import os 
from utils import save_args,load_args,get_dict_keys,write_to_csv
import csv
from time import time 
import argparse




class Workflow:
    def __init__(self,trainer:Trainer,path,epochs,name=None):
        self.trainer = trainer 
        self.name = name 
        self.epochs = epochs
        self.path = path

        

       
    def _verify_setup(self):
        """Ensure setup() actually built model, optimizer and args."""
        if self.trainer.model is None:
            raise NotImplementedError(
                f"model must be implemented in {self.trainer.__class__}.setup()"
            )
        if self.trainer.optimizer is None:
            raise NotImplementedError(
                f"optimizer must be implemented in {self.trainer.__class__}.setup()"
            )
        if self.trainer.args is None:
            raise NotImplementedError(
                f"args must be implemented in {self.trainer.__class__}.setup()"
            )
        # optional: more type checks...
        if not isinstance(self.trainer.model, torch.nn.Module):
            raise TypeError("Model is not a torch.nn.Module")
        if not isinstance(self.trainer.optimizer, torch.optim.Optimizer):
            raise TypeError("Optimizer is not a torch.optim.Optimizer")
        

    def run_training(self,epoch):
        print(epoch)
        for i in range(epoch+1,epoch+self.epochs):

            t_start_epoch = time()
            train_history = self.trainer.train_loop()
            t_end_epoch = time()

            checkpoint = {
                "epoch":i,
                "optimizer_state":self.trainer.optimizer.state_dict(),
            }
            
            torch.save(checkpoint,f"{self.path}/{self.name}/last_checkpoint.pth")
            torch.save(self.trainer.model.state_dict(),f"{self.path}/{self.name}/last_weights.pt")
            
            t_start_testing = time()
            test_history = self.trainer.evaluate()
            t_end_testing = time()

            # Check if train_loop() and evaluate() have history to return 
            if train_history is not None and not isinstance(train_history, dict):
                raise TypeError(f"Expected train_loop() to return dict or have no return, but got {type(train_history)}")
            if test_history is not None and not isinstance(test_history,dict):
                raise TypeError(f"Expected evaluate() to return dict or have no return, but got {type(train_history)}")
            
            
            data_csv = dict() 
            if isinstance(train_history, dict):
                # Store train history and time 
                data_csv.update({"train_time":t_end_epoch-t_start_epoch,**train_history})
            if isinstance(train_history, dict):
                # Store evaluation history and time 
                data_csv.update({"evaluation_time":t_end_testing-t_start_testing,**test_history})

            if data_csv:# Check if there is data stored in data_csv
                # Create history csv file to save the history of training and evaluation
                data_csv = {f"{i}":epoch,**data_csv}  
                write_to_csv(self.file_csv,data_csv)
            



    def run(self):
        """Execute the workflow."""
        expirement_names = os.listdir(self.path)
        # Check if the experience name exists 

        if self.name in expirement_names:
            # If name exist in amoung the experiments in the given path the training will continue
            print(f"Continue Trainingc [{self.name}]..")
            # Save the hyperparamters and informations about the model and experiment from json file 
            loaded_args = load_args(f"{self.path}/{self.name}/args.json")
            # Load checkpoints of the experiment
            checkpoint = torch.load(f"{self.path}/{self.name}/last_checkpoint.pth",weights_only=False)
            epoch = checkpoint["epoch"]
            self.trainer.args = loaded_args
            self.trainer.setup()
            self._verify_setup()
            state_dict = torch.load(
                f"{self.path}/{self.name}/last_weights.pt",
                weights_only=True
            )
            self.trainer.model.load_state_dict(state_dict)
            self.trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
           
            
        else :
            # If name doesn't exist in amoung the experiments in the given path experiment will start with new training 
            print(f"Start new Training [{self.name}]..")

            # Assign default name for the experiment if no name is given 
            if self.name is None :
                correct_name=0
                while f"exp {correct_name}" in expirement_names:
                    correct_name +=1
                self.name = f"exp {correct_name}"

            # Make directory for the experiment
            os.mkdir(os.path.join(f"{self.path}",self.name))
            # Save the hyperparamters and informations about the model and experiment in json file    
            save_args(self.trainer.args,f"{self.path}/{self.name}/args.json")
            self.trainer.setup()
            self._verify_setup()
            epoch = 0
        print(self.trainer.args)
        self.file_csv = f'{self.path}/{self.name}/history.csv'
        
        
        self.run_training(epoch)
        
 

        