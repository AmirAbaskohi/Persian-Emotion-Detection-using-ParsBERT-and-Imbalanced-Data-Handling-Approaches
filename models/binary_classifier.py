from transformers import BertConfig, BertTokenizer
from tqdm import tqdm
from metrics import acc_and_f1
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperparamteres import *
from emotion_dataset import create_data_loader
from data import read_data
from emotion_model import EmotionModel
import numpy as np
import argparse
from operations import train_op, eval_op, eval_callback
import collections
from sklearn.metrics import classification_report, f1_score
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="Target Emotion")
    parser.add_argument("data_address", help="Data address")
    parser.add_argument("model_name", help="Model name or path to load from transformers")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')

    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    train, valid, test, x_train, y_train, x_valid, y_valid, x_test, y_test = read_data(args.data_address)  
    idx = np.random.randint(0, len(train))
    sample_text = train.iloc[idx]['text']
    sample_label = train.iloc[idx][args.target]

    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    config = BertConfig.from_pretrained(args.model_name)
    config.num_labels = 2

    print(config.to_json_string())

    encoding = tokenizer.encode_plus(
        sample_text,
        max_length=32,
        truncation=True,
        add_special_tokens=True, # Add '[CLS]' and '[SEP]'
        return_token_type_ids=True,
        return_attention_mask=True,
        padding='max_length',
        return_tensors='pt',  # Return PyTorch tensors
    )

    print(f'Keys: {encoding.keys()}\n')
    for k in encoding.keys():
        print(f'{k}:\n{encoding[k]}')
    
    label_list = [0, 1]
    train_data_loader = create_data_loader(train['text'].to_numpy(), train[args.target].to_numpy(), tokenizer, MAX_LEN, TRAIN_BATCH_SIZE, label_list)
    valid_data_loader = create_data_loader(valid['text'].to_numpy(), valid[args.target].to_numpy(), tokenizer, MAX_LEN, VALID_BATCH_SIZE, label_list)
    test_data_loader = create_data_loader(test['text'].to_numpy(), None, tokenizer, MAX_LEN, TEST_BATCH_SIZE, label_list)

    sample_data = next(iter(train_data_loader))

    print(sample_data.keys())

    print(sample_data['text'])
    print(sample_data['input_ids'].shape)
    print(sample_data['input_ids'][0, :])
    print(sample_data['attention_mask'].shape)
    print(sample_data['attention_mask'][0, :])
    print(sample_data['token_type_ids'].shape)
    print(sample_data['token_type_ids'][0, :])
    print(sample_data['targets'].shape)
    print(sample_data['targets'][0])

    sample_test = next(iter(test_data_loader))
    print(sample_test.keys())

    pt_model = EmotionModel(config=config, model_name_or_path=args.model_name)
    pt_model = pt_model.to(device)

    print('pt_model', type(pt_model))

    sample_data_input_ids = sample_data['input_ids']
    sample_data_attention_mask = sample_data['attention_mask']
    sample_data_token_type_ids = sample_data['token_type_ids']
    sample_data_targets = sample_data['targets']

    sample_data_input_ids = sample_data_input_ids.to(device)
    sample_data_attention_mask = sample_data_attention_mask.to(device)
    sample_data_token_type_ids = sample_data_token_type_ids.to(device)
    sample_data_targets = sample_data_targets.to(device)

    outputs = pt_model(sample_data_input_ids, sample_data_attention_mask, sample_data_token_type_ids)
    _, preds = torch.max(outputs, dim=1)

    print(outputs[:5, :])
    print(preds[:5])

    optimizer = AdamW(pt_model.parameters(), lr=LEARNING_RATE, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss()

    step = 0
    eval_loss_min = np.Inf
    history = collections.defaultdict(list)

    for epoch in tqdm(range(1, EPOCHS + 1), desc="Epochs... "):
        train_y, train_loss, step, eval_loss_min = train_op(
            model=pt_model, 
            data_loader=train_data_loader, 
            loss_fn=loss_fn, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            step=step, 
            print_every_step=EEVERY_EPOCH, 
            device=device,
            eval=True,
            eval_cb=eval_callback(epoch, EPOCHS, OUTPUT_PATH),
            eval_loss_min=eval_loss_min,
            eval_data_loader=valid_data_loader, 
            clip=CLIP)
        
        train_score = acc_and_f1(train_y[0], train_y[1], average='weighted')
        
        eval_y, eval_loss = eval_op(
            model=pt_model, 
            data_loader=valid_data_loader, 
            loss_fn=loss_fn)
        
        eval_score = acc_and_f1(eval_y[0], eval_y[1], average='weighted')
        
        history['train_acc'].append(train_score['acc'])
        history['train_loss'].append(train_loss)
        history['val_acc'].append(eval_score['acc'])
        history['val_loss'].append(eval_loss)

        y_test, y_pred = [label_list.index(label) for label in test[args.target].values], preds

        print(f'F1: {f1_score(y_test, y_pred, average="weighted")}')
        print()
        print(classification_report(y_test, y_pred))