from train import Dataset
import pandas as pd
import torch
import transformers
from tqdm import tqdm 
from torch.utils.data import Sampler
from collections import Counter
import numpy as np

train_df = pd.read_csv("../data/train.csv")

tokenizer = transformers.AutoTokenizer.from_pretrained("snunlp/KR-ELECTRA-discriminator", max_length=150)

def tokenizing(dataframe):
    data = []
    for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
        # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
        text = '[SEP]'.join([item[text_column] for text_column in ['sentence_1','sentence_2']])
        outputs = tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
        data.append(outputs['input_ids'])
    return data

def preprocessing(data):
    data = data.drop(columns=['id'])
    targets = data['label'].values.tolist()
    inputs = tokenizing(data)

    return inputs, targets

train_inputs, train_targets = preprocessing(train_df)

train_dataset = Dataset(train_inputs, train_targets)

class OverSampler(Sampler):
    """Over Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, targets, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
#        self.n_splits = int(class_vector.size(0) / batch_size)
        self.targets = targets
        self.data_len = len(targets)
        self.indices = list(range(len(targets)))
        self.batch_size = batch_size
        target_list = targets
        target_bin = np.floor(np.array(targets) * 2.0) / 2.0
        bin_count = Counter(target_bin)
        bin_count[4.5] += bin_count[5.0]
        weights = [1.0 / bin_count[np.floor(2.0*label)/2.0] for label in targets]

        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.batch_size, replacement=True))

    def __len__(self):
        return self.data_len
print(train_dataset.targets)

sampler = OverSampler(train_targets, 8)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, 
        sampler=sampler, 
        drop_last=True)

for x in next(iter(dataloader)):
    print(x)

