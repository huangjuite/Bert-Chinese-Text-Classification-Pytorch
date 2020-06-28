# Bert-Chinese-Text-Classification-Pytorch
forked from 649453932/Bert-Chinese-Text-Classification-Pytorch

## Requirements

    pip install -r requirements.txt

## Download pretrained model
https://drive.google.com/open?id=1eHM3l4fMo6DsQYGmey7UZGiTmQquHw25

put all files underchinese_robert_wwm_ext_pytorch
    
    chinese_roberta_wwm_ext_pytorch/
    ├── bert_config.json
    ├── pytorch_model.bin
    └── vocab.txt
    

## train
    python3 run.py --model bert

result will be at

    dl_with_key/
    └── chinese_roberta_wwm_ext_pytorch/
        ├── final_loss_0.0008_acc_1.0000.csv
        ├── iter00000_loss_2.3661_acc_0.0781.csv
        ├── iter00100_loss_0.7410_acc_0.9453.csv
        ├── iter00200_loss_0.4056_acc_0.9297.csv
        ├── iter00300_loss_0.3632_acc_0.9062.csv
        ├── iter00400_loss_0.2733_acc_0.9297.csv
        ├── iter00500_loss_0.2267_acc_0.9453.csv
        └── ...

## implementation detail

- model input
    - input the news title and append the keyworks to the end of each title.
    - see dl_with_key/data/train.txt

- pretrained model
    - using chinese robert as pretrained model.
    - pretrain model is trained using whole word masking.
    - see https://github.com/huangjuite/Chinese-BERT-wwm

- classification model
    - using 2 layers of fully connected network with 128 neurons at each layer.
    - see model/bert.py