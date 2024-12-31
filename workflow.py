import torch 
from train import Trainer
import os 
from utils import save_args,load_args,get_dict_keys,write_to_csv
import csv
from time import time 





class Workflow:
    def __init__(self,trainer:Trainer,path,name,epochs,args):
        self.trainer = trainer 
        self.name = name 
        self.epochs = epochs
        self.path = path
        self.args = args
        evaluation_keys = trainer.evaluation_keys
        self.file_csv = f'{self.path}/{self.name}/history.csv'

    def run_training(self,epoch):
        for i in range(epoch+1,self.epochs):
            t_start_epoch = time()
            train_history = self.trainer.train_loop()
            t_end_epoch = time()

            checkpoint = {
                "epoch":epoch,
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
            
            if data_csv:
                # Create history csv file to save the history of training and evaluation
                data_csv = {"epoch":epoch,**data_csv}  
                write_to_csv(self.file_csv)
            



    def run(self):
        """Execute the workflow."""
        expirement_names = os.listdir(self.path)
        # Check if the experience name exists 

        if self.name in expirement_names:
            # If name exist in amoung the experiments in the given path the training will continue
            print(f"Continue Trainingc [{self.name}]..")
            # Save the hyperparamters and informations about the model and experiment from json file 
            args = load_args(f"{self.path}/{self.name}/args.json")
            # Load checkpoints of the experiment
            checkpoint = torch.load(f"{self.path}/{self.name}/last_checkpoint.pth")
            epoch = checkpoint["epoch"]
            self.trainer.model.load_state_dict(torch.load(f"{self.path}/{self.name}/last_weights.pt"))
            self.trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
           
            
        else :
            # If name doesn't exist in amoung the experiments in the given path experiment will start with new training 
            print(f"Start new Training [{self.name}]..")

            # Assign default name for the experiment if no name is given 
            if self.name is None :
                correct_name=0
                while f"exp ({correct_name})" not in expirement_names:
                    correct_name +=1
                self.name = f"exp ({correct_name})"

            # Save the hyperparamters and informations about the model and experiment in json file    
            save_args(args,f"{self.path}/{self.name}/args.json")
            # Make directory for the experiment
            os.mkdir(os.path.join("{self.path}",self.name))

     
            epoch = 0
            




 

        print("Preparing data...")
        self.dataset.prepare_data()

        print("Setting up the model...")
        self.model.setup_model()

        print("Training the model...")
        self.model.train_model()

        print("Evaluating the model...")
        self.model.evaluate_model()