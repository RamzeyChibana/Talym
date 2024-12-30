import torch 
from train import Trainer
import os 
from utils import save_args,load_args,get_dict_keys
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

    def run_training(self,epoch):
        for i in range(epoch+1,self.epochs):
            t_start_epoch = time()
            self.trainer.train_loop()
            t_end_epoch = time()

            checkpoint = {
                "epoch":epoch,
                "optimizer_state":self.trainer.optimizer.state_dict(),
            }
            
            torch.save(checkpoint,f"Experiments/{self.name}/last_checkpoint.pth")
            torch.save(self.trainer.model.state_dict(),f"Experiments/{self.name}/last_weights.pt")



    def run(self):
        """Execute the workflow."""
        expirement_names = os.listdir(self.path)

        if self.name in expirement_names:
            print(f"Continue Training [{self.name}]..")
            args = load_args(f"Experiments/{self.name}/args.json")
            checkpoint = torch.load(f"Experiments/{self.name}/last_checkpoint.pth")
            file_csv = open(f'Experiments/{self.name}/history.csv', 'a', newline='')
            writer = csv.writer(file_csv)
            epoch = checkpoint["epoch"]
            self.trainer.model.load_state_dict(torch.load(f"Experiments/{self.name}/last_weights.pt"))
            self.trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
           
            
        else :
            print("Start new Training")
            if self.name is None :
                correct_name=0
                while f"exp ({correct_name})" not in expirement_names:
                    correct_name +=1
                self.name = f"exp ({correct_name})"
            save_args(args,f"Experiments/{self.name}/args.json")
            os.mkdir(os.path.join("Experiments",self.name))
            file_csv = open(f'Experiments/{self.name}/history.csv', 'w', newline='')
            writer = csv.writer(file_csv)
            epoch = 0
            csv_columns = ["epoch","epoch_time"]+self.evaluation_keys
            writer.writerow(csv_columns)




 

        print("Preparing data...")
        self.dataset.prepare_data()

        print("Setting up the model...")
        self.model.setup_model()

        print("Training the model...")
        self.model.train_model()

        print("Evaluating the model...")
        self.model.evaluate_model()