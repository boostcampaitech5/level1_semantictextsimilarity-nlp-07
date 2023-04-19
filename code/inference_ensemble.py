import argparse
import os
import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl
import numpy as np
from torch.utils.data.sampler import Sampler

from utils import extract_val_pearson
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


class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path, oversampling=False):
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

# class Ensemble(pl.LightningModule):
#     def __init__(self, model_names, scores):
#         """
#         Args: 
#             models : List of models
#             scores : List of scores of each model
#         """
#         super(Ensemble, self).__init__()
#         self.scores = scores
#         models = []
#         for model_name in model_names:
#             models.append(Model.load_from_checkpoint('./checkpoints/'+model_name))
#         self.models = torch.nn.ModuleList(models)
    
#     def forward(self, x):
#         outputs = []
#         for model in self.models:
#             output = model(x)
#             outputs.append(output)

#         return outputs

#     def predict_step(self, batch, batch_idx):
#         x = batch
#         outputs = self(x)
        
#         score_sum = sum(self.scores)
        
#         res = outputs[0].squeeze() * self.scores[0] / score_sum 
        
#         for i in range(1, len(outputs)):
#             res += outputs[i].squeeze() * self.scores[i] / score_sum
#         return res

if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='klue/roberta-small', type=str)

    parser.add_argument('--batch_size', default=16, type=int)

    parser.add_argument('--checkpoint_name', default="", type=str)
    parser.add_argument('--checkpoint_new_or_best', default='new', help="input new or best")

    parser.add_argument('--max_epoch', default=1, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--loss', default='L1', type=str)
    parser.add_argument('--shuffle', default=True)
    
    parser.add_argument('--data_path', default='./data/', type=str)
    args = parser.parse_args()

    train_path = args.data_path + 'train.csv'
    dev_path = args.data_path + 'dev.csv'
    test_path = args.data_path + 'test.csv'
    predict_path = args.data_path + 'test.csv'

    # Inference part
    # 저장된 모델로 예측을 진행합니다.
    # Ensemble을 적용하기 위해 불러올 모델 ckpt 이름을 아래 리스트에 넣으면 됩니다. 
    # 아래는 예시로 남겨놓았습니다. 
    ckpt_names = [
         'monologg-koelectra-base-v3-discriminator-sts-epoch=16-val_pearson=0.926.ckpt',
         'snunlp-KR-ELECTRA-discriminator-sts-epoch=12-val_pearson=0.930.ckpt',
         'snunlp-KR-ELECTRA-discriminator-sts-epoch=19-val_pearson=0.931.ckpt',
         'snunlp-KR-ELECTRA-discriminator-sts-epoch=4-val_pearson=0.932.ckpt',
         'klue-roberta-large-sts-epoch=12-val_pearson=0.923.ckpt',
         'jhgan-ko-sroberta-multitask-sts-epoch=9-val_pearson=0.919.ckpt'
    ]
    # 각 모델의 val_pearson을 저장합니다. 
    scores = []
    for ckpt_name in ckpt_names:
        # models.append(Model.load_from_checkpoint('./checkpoints/'+model_name))
        temp = ckpt_name.split('val_pearson=')[-1]
        score = float(temp.split('ckpt')[0][:-1])
        scores.append(score)
        
    
    predictions = None
    cnt = 0
    score_sum = sum(scores) # Weighted sum을 하기 위해 val_pearson을 다 더합니다.
    
    # 각 ckpt 모델에 대해 inference 수행 후 weighted sum.
    for ckpt_name in ckpt_names:
        temp = ckpt_name.split('-sts')[0].split('-')
        model_name = temp[0] + '/' + '-'.join(temp[1:])
        dataloader = Dataloader(model_name, args.batch_size, args.shuffle, train_path, dev_path,
                            test_path, predict_path)
        model = Model.load_from_checkpoint('./checkpoints/'+ckpt_name)
        trainer = pl.Trainer(accelerator='gpu', max_epochs=args.max_epoch, log_every_n_steps=1)
        res = torch.concat(trainer.predict(model=model, datamodule=dataloader))
        if predictions != None:
            predictions += res * scores[cnt] / score_sum
        else:
            predictions = res * scores[cnt] / score_sum
        cnt += 1

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    predictions = list(round(float(i), 1) for i in predictions)

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = pd.read_csv('./output/sample_submission.csv')
    output['target'] = predictions
    outputname = './output/output_' + 'ensemble' + '.csv'
    output.to_csv(outputname, index=False)
