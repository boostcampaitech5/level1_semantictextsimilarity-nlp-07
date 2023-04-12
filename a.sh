#!/bin/bash
MODELS=("klue/roberta-base" "snunlp/KR-ELECTRA-discriminator" "beomi/KcELECTRA-base")

MODELS_LEN=${#MODELS[@]}

echo $MODELS_LEN

python3 code/train.py --model_name "klue/roberta-large" --batch_size 8 --learning_rate 1e-6 --max_epoch 10 --loss MSE

for (( i=0; i<${MODELS_LEN}; i++ ));
do
    echo ${MODELS[$i]}
    python3 code/train.py --model_name ${MODELS[$i]} --max_epoch 10 --loss MSE
done


python3 code/train.py --model_name "klue/roberta-large" --batch_size 8 --learning_rate 1e-6 --max_epoch 10 

for (( i=0; i<${MODELS_LEN}; i++ ));
do
    echo ${MODELS[$i]}
    python3 code/train.py --model_name ${MODELS[$i]} --max_epoch 10
done

