# Semantic Text Similarity

## Directory
checkpoints
└── checkpoint_name.ckpt
code   
├── inference.py   
├── train.py
└── utils.py
data   
├── dev.csv   
├── test.csv   
└── train.csv   
model   
└── model.pt   
output   
├── output.csv   
└── sample_submission.csv   
.gitignore   
config.json   
DA.ipynb   
Readme.md   
requirements.txt   

## Usage
1. setting(config.json 파일 사용 시)
- config.json.tab -> config.json으로 만든 후 사용   
- config.json 파일을 열어 필요한 사항을 확인
    - model
    - hyperparameter
        - 변경할 수 있는 값: batch_size, max_epoch, learning_rate, loss, shuffle
    - checkpoint
        - 변경할 수 있는 값: checkpoint_use, checkpoint_name, checkpoint_new_or_best
    - wandb
        - 변경할 수 있는 값: username, entity, key, project
        - username, entity, key는 빈 문자열로 되어 있으며 개인 정보를 wandb에서 확인 후 입력

2. train   
- config.json 파일 사용
    > python ./code/train.py --config config.json
- config.json 파일 미사용
    > python ./code/train.py [-h] [--model_name MODEL_NAME] 
                [--checkpoint_use CHECKPOINT_USE] [--checkpoint_name CHECKPOINT_NAME][--checkpoint_new_or_best CHECKPOINT_NEW_OR_BEST] 
                [--batch_size BATCH_SIZE] [--max_epoch MAX_EPOCH] [--shuffle SHUFFLE] [--learning_rate LEARNING_RATE] [--data_path DATA_PATH] [--train_path TRAIN_PATH] [--dev_path DEV_PATH]
                [--test_path TEST_PATH] [--predict_path PREDICT_PATH] [--loss LOSS] [--random_seed RANDOM_SEED] [--wandb_username WANDB_USERNAME] [--wandb_entity WANDB_ENTITY] [--wandb_key WANDB_KEY]
                [--wandb_project WANDB_PROJECT]

3. inference   
    > inference.py [-h] [--model_name MODEL_NAME] 
                    [--checkpoint_name CHECKPOINT_NAME] [--checkpoint_new_or_best CHECKPOINT_NEW_OR_BEST] 
                    [--batch_size BATCH_SIZE] [--max_epoch MAX_EPOCH] [--shuffle SHUFFLE] [--learning_rate LEARNING_RATE] [--data_path DATA_PATH]

4. checkpoints
- 기존의 model.pt -> checkpoint.ckpt로 대체
- args 설명
    -   checkpoint_use: checkpoint를 사용할 것인지의 여부 
    -   checkpoint_name: checkpoint의 경로가 아닌 이름.ckpt
    -   checkpoint_new_or_best: checkpoint_name 미지정시 최근 또는 최고의 모델 자동 선택 input: new, best