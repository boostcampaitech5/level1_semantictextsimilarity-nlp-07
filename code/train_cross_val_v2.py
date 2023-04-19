import argparse
import datetime
import os

import pandas as pd

from tqdm.auto import tqdm
from collections import Counter

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl
# 2023-04-10 모듈 로딩 추가, callback, wandb
import wandb
import numpy as np
import random

from sklearn.model_selection import KFold
from torch.utils.data.sampler import Sampler

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from utils import extract_val_pearson
from utils import set_model_name
from utils import set_hyperparameter_config
from utils import set_checkpoint_config
from utils import set_wandb_config
import glob

class OverSampler(Sampler):
    """Over Sampling Sampler
    providing uniform distribution of target labels in each batch
    """
    def __init__(self, targets):
        """
        Arguments
        ---------
        targets
            a list of class labels
        """
        self.targets = targets
        self.num_samples = len(targets)
        self.indices = list(range(len(targets)))
        target_list = targets
        target_bin = np.floor(np.array(targets) * 2.0) / 2.0
        bin_count = Counter(target_bin.reshape(-1))
        bin_count[4.5] += bin_count[5.0]
        bin_count[5.0] = bin_count[4.5]
        weights = [1.0 / bin_count[np.floor(2.0*label[0])/2.0] for label in targets]

        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        count = 0
        index = [self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True
        )]
        while count < self.num_samples:
            yield index[count] 
            count += 1
            
        # return (self.indices[i] for i in torch.multinomial(
        #     self.weights, self.batch_size, replacement=True))

    def __len__(self):
        return self.num_samples

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


class KfoldDataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path, 
                 oversampling=False, k=1, split_seed = 777, num_splits = 10):
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
        
        self.oversampling = oversampling
        if oversampling:
            print("NOTICE: Oversampling activated")

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=160)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']
        self.tokenizer.add_special_tokens({'additional_special_tokens' : ['<PERSON>']})
        
        self.k = k
        self.split_seed = split_seed
        self.num_splits = num_splits

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
            # K-fold로 분할할 학습 데이터를 호출합니다.
            # train.csv만을 가지고 k-fold cross validation을 진행.
            # dev.csv는 test stage에서만 사용합니다. 
            total_data = pd.read_csv(self.train_path)
            kf = KFold(n_splits=self.num_splits, shuffle=self.shuffle, random_state=self.split_seed)
            all_splits = [k for k in kf.split(total_data)] # List of tuples : (array([...]), array([...])) <- 각각 training, validation에 대한 index 리스트
            
            train_indexes, val_indexes = all_splits[self.k] # k 번째 set이 validation으로 들어가게 됩니다.
            train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()
            
            # iloc를 사용하여 두 개의 dataset으로 쪼갭니다. 
            train_data = total_data.iloc[train_indexes]
            val_data = total_data.iloc[val_indexes]
            
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
        if self.oversampling:
            sampler = OverSampler(self.train_dataset.targets)
            return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, 
                                               sampler=sampler, 
                                               drop_last=False)
        else:
            return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

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
            
        self.validation_step_outputs = []

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
        self.validation_step_outputs.append(val_pearson)
        return loss

    def on_validation_epoch_end(self):
        val_pearson = torch.stack(self.validation_step_outputs)
        print("\nval_pearson: ", val_pearson.mean().item(), "\n")
        self.validation_step_outputs.clear()
        

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

if __name__ == '__main__':    
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='klue/roberta-small', type=str)

    parser.add_argument('--checkpoint_use', default="True", type=str, help="True/False")
    parser.add_argument('--checkpoint_name', default=None, type=str)
    parser.add_argument('--checkpoint_new_or_best', default='new', help="input new or best")

    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=5, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--loss', default='L1', type=str)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--oversampling', default=True, type=bool)

    parser.add_argument('--data_path', default='./data/', type=str)
    parser.add_argument('--train_path', default='./data/train.csv')
    parser.add_argument('--dev_path', default='./data/dev.csv')
    parser.add_argument('--test_path', default='./data/dev.csv')
    parser.add_argument('--predict_path', default='./data/test.csv')
    parser.add_argument('--random_seed', default=False, type=bool)

    parser.add_argument('--wandb_username', default='username')
    parser.add_argument('--wandb_entity', default='username')
    parser.add_argument('--wandb_key', default='key')
    parser.add_argument('--wandb_project', default='STS')
    parser.add_argument('--config', default=False, type=str, help='config file')

    date = datetime.datetime.now().strftime('%Y-%m-%d')
    args = parser.parse_args()
    
    train_path = args.data_path + 'train.csv'
    dev_path = args.data_path + 'dev.csv'
    test_path = args.data_path + 'dev.csv'
    predict_path = args.data_path + 'test.csv'
    print(train_path)
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
    checkpoint_config = set_checkpoint_config(args)
    wandb_config = set_wandb_config(args)

    # 2023-04-10: 모델에 대한 Callback을 추가합니다.
    # Pytorch Lightning에서 지원하는 Model Checkpoint 저장 및 EarlyStopping을 추가해줍니다.
    cp_callback = ModelCheckpoint(monitor='val_pearson',    # Pearson coefficient를 기준으로 저장
                                  verbose=True,            # 중간 출력문을 출력할지 여부. False 시, 없음.
                                  save_last=False,           # last.ckpt 로 저장됨
                                  save_top_k=1,             # k개의 최고 성능 체크 포인트를 저장하겠다.
                                  save_weights_only=True,   # Weight만 저장할지, 학습 관련 정보도 저장할지 여부.
                                  mode='max',                # 'max' : monitor metric이 증가하면 저장.
                                  dirpath='./checkpoints',
                                  filename=f'{model_name.replace("/","-")}-' + 'sts-{epoch}-{val_pearson:.3f}',
                                  )
    #cp_callback.CHECKPOINT_NAME_LAST=f'{model_name.replace("/","-")}-' + 'sts-last-{val_pearson:.3f}'
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

    # dummy dataloader for vocab_size
    dataloader = KfoldDataloader(model_name, hyperparameter_config["batch_size"], hyperparameter_config["shuffle"], train_path, dev_path, 
                                    test_path, predict_path, oversampling=hyperparameter_config["oversampling"])
    vocab_size = len(dataloader.tokenizer)
    print("LL", vocab_size)

    num_folds = 10
    split_seed = 777

    # max_epoch * num_folds 만큼 ckpt 파일이 생성됩니다. 
    # 매번 Trainer, logger가 초기화되기 때문에 최고 성능의 ckpt를 자동 저장할 수 없는 문제가 있어, 임시조치로서 이렇게 했습니다.
    # torch.save를 이용해 매번 최고성능인지를 체크한 후 저장 여부를 결정하게끔 바꿀 수 있을 것 같습니다. 
    
    val_loss_list = []
    val_pearson_list = []
    test_pearson_list = []
    best_ckpt_name = None
    for k in range(num_folds):
        # K-fold cross validation : train.csv 샘플이 num_folds 개의 set으로 쪼개집니다. 
        # k : k-fold cross validation에서 k번째 set이 validation dataset으로 사용됩니다. 
        # split_seed를 고정해야 합니다. 
        
        model = Model(model_name, hyperparameter_config['learning_rate'], vocab_size, hyperparameter_config['loss'])
        cp_callback = ModelCheckpoint(monitor='val_pearson',    
                                    verbose=True,           
                                    save_last=False,           
                                    save_top_k=1,            
                                    save_weights_only=True,   
                                    mode='max',                
                                    dirpath='./checkpoints',
                                    filename=f'{model_name.replace("/","-")}-' + 'sts-{val_pearson:.3f}' + f'-k={k}',
                                    )
        wandb_logger = WandbLogger(
            log_model=False,
            name=f'{model_name.replace("/","-")}_{hyperparameter_config["batch_size"]}_{hyperparameter_config["learning_rate"]:.3e}_{hyperparameter_config["loss"]}_'+
            f'data' + args.data_path[-2] + f'_{date}',
            project=wandb_config["project"]+'_'+hyperparameter_config["loss"], 
            entity=wandb_config["entity"]
        )
        kfdataloader = KfoldDataloader(model_name, hyperparameter_config["batch_size"], 
                                hyperparameter_config["shuffle"], train_path, dev_path, test_path, predict_path, 
                                oversampling=hyperparameter_config["oversampling"], k=k, split_seed=split_seed, # k번째 set이 validation set으로 사용됨. 
                                num_splits=num_folds)
        trainer = pl.Trainer(accelerator='gpu', max_epochs=1,
                                log_every_n_steps=1, callbacks=[cp_callback, early_stop_callback],
                                logger=wandb_logger)
        trainer.fit(model=model, datamodule=kfdataloader)
        val_pearson_k = trainer.logged_metrics['val_pearson'].item()
        val_loss_k = trainer.logged_metrics['val_loss'].item()
        val_pearson_list.append(val_pearson_k)
        val_loss_list.append(val_loss_k)
        trainer.test(model=model, datamodule=kfdataloader)
        test_pearson_list.append(trainer.logged_metrics['test_pearson'].item())
        wandb.finish()
        if not val_pearson_list or max(val_pearson_list) == val_pearson_k:
            best_ckpt_name = f'{model_name.replace("/","-")}-sts-val_pearson={val_pearson_k:.3f}_{k}.ckpt'
    
    print("NOTICE: K-fold cross validation has ended")
    print("="*75,"\n")
    print("  avg_val_loss =", np.mean(val_loss_list))
    print("  avg_val_pearson =", np.mean(val_pearson_list))
    print("  avg_test_pearson =", np.mean(test_pearson_list), "\n")
    print("  Best model checkpoints:", best_ckpt_name, "\n")
    print("="*75)
    

