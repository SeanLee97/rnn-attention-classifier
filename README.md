# rnn-attention-classifier
tensorflow 实现RNN+Attention文本分类

[注！]：此项目为临时项目

## Usage
```
usage: main.py [-h] [--mode MODE] [--use_attention] [--batch_size BATCH_SIZE]
               [--epoch EPOCH] [--log_step LOG_STEP] [--rnn_type RNN_TYPE]
               [--bi_rnn] [--num_layers NUM_LAYERS] [--embed_size EMBED_SIZE]
               [--hidden_size HIDDEN_SIZE] [--attn_size ATTN_SIZE]
               [--dropout DROPOUT] [--optim_type OPTIM_TYPE]
               [--learning_rate LEARNING_RATE] [--lr_decay LR_DECAY]
               [--l2_reg L2_REG] [--max_grad_norm MAX_GRAD_NORM]
               [--runtime_dir RUNTIME_DIR] [--ckpt_dir CKPT_DIR] [--resume]
               [--valid]
```
## how to train
```
python3 main.py --mode train --valid
```
