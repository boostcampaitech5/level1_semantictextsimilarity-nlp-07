import json
import os
from collections import defaultdict

def set_model_name(args):
    if args.config:
        with open(args.config) as json_data:
            data = json.load(json_data)
        return data["model"]
    else:
        return args.model_name

def set_hyperparameter_config(args):
    hyperparameter_config = defaultdict()
    if args.config:
        with open(args.config) as json_data:
            data = json.load(json_data)
        hyperparameter_config["batch_size"] = data["hyperparameter"]["batch_size"]
        hyperparameter_config["max_epoch"] = data["hyperparameter"]["max_epoch"]
        hyperparameter_config["learning_rate"] = data["hyperparameter"]["learning_rate"]
        hyperparameter_config["loss"] = data["hyperparameter"]["loss"]
        hyperparameter_config["shuffle"] = data["hyperparameter"]["shuffle"]
        hyperparameter_config["oversampling"] = data["hyperparameter"]["oversampling"]
    else:
        hyperparameter_config["batch_size"] = args.batch_size
        hyperparameter_config["max_epoch"] = args.max_epoch
        hyperparameter_config["learning_rate"] = args.loss
        hyperparameter_config["loss"] = args.wandb_project
        hyperparameter_config["shuffle"] = args.shuffle
        hyperparameter_config["oversampling"] = args.oversampling
    
    return hyperparameter_config

def set_wandb_config(args):
    wandb_config = defaultdict()
    if args.config:
        with open(args.config) as json_data:
            data = json.load(json_data)
        wandb_config["username"] = data["wandb"]["username"]
        wandb_config["entity"] = data["wandb"]["entity"]
        wandb_config["key"] = data["wandb"]["key"]
        wandb_config["project"] = data["wandb"]["project"]
    else:
        wandb_config["username"] = args.wandb_username
        wandb_config["entity"] = args.wandb_entity
        wandb_config["key"] = args.wandb_key
        wandb_config["project"] = args.wandb_project
    
    return wandb_config

def set_checkpoint_config(args):
    checkpoint_config = defaultdict()
    if args.config:
        with open(args.config) as json_data:
            data = json.load(json_data)
        checkpoint_config["checkpoint_name"] = data["checkpoint"]["checkpoint_name"]
    else:
        checkpoint_config["checkpoint_name"] = args.checkpoint_name
    
    return checkpoint_config

def extract_val_pearson(file_name):
    # Extract the val_pearson value from the file name
    try:
        val_pearson = float(file_name.split("val_pearson=")[1].split(".ckpt")[0])
    except:
        val_pearson = -1
    return val_pearson

def working_directory_match(word):
    current_working_directory = os.getcwd()
    if word not in current_working_directory:
      os.chdir('./' + word)