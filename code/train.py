import argparse
import datetime
import os
import json
from collections import defaultdict

import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl
# 2023-04-10 모듈 로딩 추가, callback, wandb
import wandb
import numpy as np
import random

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from utils import extract_val_pearson
import glob

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)


class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=160)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']
        self.tokenizer.add_special_tokens({'additional_special_tokens' : ['<PERSON>']})

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
        return data

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            # 학습데이터 준비
            train_inputs, train_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=args.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)


class Model(pl.LightningModule):
    def __init__(self, model_name, lr, vocab_size, loss="L1"):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        
        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1)
        self.plm.resize_token_embeddings(vocab_size)
        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        # L1 loss가 아닌 MSE loss (=L2 loss)도 사용해봅시다. 
        if loss == "MSE":
            self.loss_func = torch.nn.MSELoss()
        else:
            self.loss_func = torch.nn.L1Loss()

    def forward(self, x):
        x = self.plm(x)['logits']

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)

        val_pearson = torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze())
        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))
#        self.log("val_spearman", torchmetrics.functional.spearman_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))
#        self.log("test_spearman", torchmetrics.functional.spearman_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

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

if __name__ == '__main__':    
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='klue/roberta-small', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--checkpoint', default="True", type=str, help="True/False")
    parser.add_argument('--new_or_best', default='new', help="new/best")
    parser.add_argument('--max_epoch', default=5, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--loss', default='L1', type=str)

    parser.add_argument('--data_path', default='./data/', type=str)
    parser.add_argument('--train_path', default='./data/train.csv')
    parser.add_argument('--dev_path', default='./data/dev.csv')
    parser.add_argument('--test_path', default='./data/dev.csv')
    parser.add_argument('--predict_path', default='./data/test.csv')

    parser.add_argument('--wandb_username', default='username')
    parser.add_argument('--wandb_entity', default='username')
    parser.add_argument('--random_seed', default=False, type=bool)
    parser.add_argument('--wandb_key', default='key')
    parser.add_argument('--wandb_project', default='STS')
    parser.add_argument('--config', default="../config.json", type=str, help='config file')
       
    date = datetime.datetime.now().strftime('%Y-%m-%d')
    args = parser.parse_args()
    
    train_path = args.data_path + 'train.csv'
    dev_path = args.data_path + 'dev.csv'
    test_path = args.data_path + 'dev.csv'
    predict_path = args.data_path + 'test.csv'
        
    if args.random_seed:
        global_seed = 777
        print("="*50,"\nNOTICE: Fixing random seed to", global_seed, "\n" + "="*50, "\n")
        torch.manual_seed(global_seed)
        torch.cuda.manual_seed(global_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(global_seed)
        random.seed(global_seed)
    
    model_name = set_model_name(args)
    hyperparameter_config = set_hyperparameter_config(args)
    wandb_config = set_wandb_config(args)

    # 2023-04-10: 모델에 대한 Callback을 추가합니다.
    # Pytorch Lightning에서 지원하는 Model Checkpoint 저장 및 EarlyStopping을 추가해줍니다.
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename=f'{args.model_name.replace("/","-")}-' + 'sts-{epoch}-{val_pearson:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='val_pearson',
        mode='max'
    )

    early_stop_callback = EarlyStopping(monitor='val_pearson', 
                                        patience=5,         # 2번 이상 validation 성능이 안좋아지면 early stop
                                        mode='max'          # 'max' : monitor metric은 최대화되어야 함.
                                        )
    
    # dataloader와 model을 생성합니다.
    # dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
    #                         args.test_path, args.predict_path)
    # model = Model(args.model_name, args.learning_rate)

    wandb.login(key=wandb_config["key"])
    model_name = model_name
    wandb_logger = WandbLogger(
        log_model="all",
        name=f'{model_name.replace("/","-")}_{hyperparameter_config["batch_size"]}_{hyperparameter_config["learning_rate"]:.3e}_{hyperparameter_config["loss"]}_{date}',
        project=wandb_config["project"]+'_'+hyperparameter_config["loss"], 
        entity=wandb_config["entity"]
    )
    dataloader = Dataloader(model_name, hyperparameter_config["batch_size"], hyperparameter_config["shuffle"], train_path, dev_path, 
                                    test_path, predict_path)
    vocab_size = len(dataloader.tokenizer)
    print("LL", vocab_size)

    checkpoint_file = False
    if args.checkpoint=="True":
        checkpoint_pattern = f"./checkpoints/*.ckpt"
        checkpoint_files = glob.glob(checkpoint_pattern)
        # Sort the list of checkpoint files by val_pearson in descending order
        if args.new_or_best.lower() == "best":
            checkpoint_files = sorted(checkpoint_files, key=extract_val_pearson, reverse=True)
        else:
            checkpoint_files = sorted(checkpoint_files, key=os.path.getctime, reverse=True)
        checkpoint_file = checkpoint_files[0]
    
    if not checkpoint_file:
        model = Model(args.model_name, args.learning_rate, vocab_size, args.loss)
        trainer = pl.Trainer(gpus=1, max_epochs=hyperparameter_config["max_epoch"], log_every_n_steps=1, 
                             callbacks=[checkpoint_callback, early_stop_callback], 
                             logger=wandb_logger)
    else:
        # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
        model = Model.load_from_checkpoint(checkpoint_file)
        trainer = pl.Trainer(gpus=1, max_epochs=hyperparameter_config["max_epoch"], resume_from_checkpoint=checkpoint_file, 
                             log_every_n_steps=1, callbacks=[checkpoint_callback, early_stop_callback],
                               logger=wandb_logger)

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)