import json
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
    else:
        hyperparameter_config["batch_size"] = args.batch_size
        hyperparameter_config["max_epoch"] = args.max_epoch
        hyperparameter_config["learning_rate"] = args.loss
        hyperparameter_config["loss"] = args.wandb_project
        hyperparameter_config["shuffle"] = args.shuffle
    
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
        checkpoint_config["checkpoint"] = data["checkpoint"]["checkpoint"]
        checkpoint_config["new_or_best"] = data["checkpoint"]["new_or_best"]
    else:
        checkpoint_config["checkpoint"] = args.checkpoint
        checkpoint_config["new_or_best"] = args.new_or_best
    
    return checkpoint_config

def extract_val_pearson(file_name):
    # Extract the val_pearson value from the file name
    val_pearson = float(file_name.split("val_pearson=")[1].split(".ckpt")[0])
    return val_pearson