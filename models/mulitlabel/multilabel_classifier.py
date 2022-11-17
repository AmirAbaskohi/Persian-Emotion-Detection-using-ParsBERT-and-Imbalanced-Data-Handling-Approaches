import argparse
import pandas as pd
from tqdm import tqdm
from torch import cuda
from hyperparameters import *
from preprocess import cleaning
from transformers import BertTokenizer
import warnings
warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import torch
from multilabel_dataset import MultiLabelDataset
from torch.utils.data import DataLoader
import logging
from utils import *
from multilabel_model import MultilabelModel
from operations import train, validation
logging.basicConfig(level=logging.ERROR)
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_address", help="Data address")
    parser.add_argument("model_name", help="Model name or path to load from transformers")
    parser.add_argument("thresh", help="Threshold for assigning label to each emotion", default=0.5)
    args = parser.parse_args()

    df = pd.read_csv(args.data_address)
    print(df.head())

    df['text'] = df['text'].apply(cleaning)
    df = df.sample(frac=1).reset_index(drop=True)
    print(df.head())

    rowsLabels = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        rowLabels = [row["Anger"], row["Fear"], row["Happiness"], row["Hatred"], row["Sadness"], row["Wonder"]]
        rowsLabels.append(rowLabels)
    df['labels'] = rowsLabels
    df = df.drop(columns=["Anger", "Fear", "Happiness", "Hatred", "Sadness", "Wonder"])
    print(df.head())

    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f"Device is {device}")

    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    train_size = 0.8
    train_data=df.sample(frac=train_size,random_state=200)
    test_data=df.drop(train_data.index).reset_index(drop=True)
    train_data = train_data.reset_index(drop=True)


    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_data.shape))
    print("TEST Dataset: {}".format(test_data.shape))

    training_set = MultiLabelDataset(train_data, tokenizer, MAX_LEN)
    testing_set = MultiLabelDataset(test_data, tokenizer, MAX_LEN)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    model = MultilabelModel(args.model_name)
    model.to(device)

    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        train(model, training_loader, optimizer, device, epoch)

    outputs, targets = validation(model, device, testing_loader)
    final_outputs = np.array(outputs) >= args.thresh

    val_hamming_loss = metrics.hamming_loss(targets, final_outputs)
    val_hamming_score = hamming_score(np.array(targets), np.array(final_outputs))

    print(f"Hamming Score = {val_hamming_score}")
    print(f"Hamming Loss = {val_hamming_loss}")

    anger_final_outputs = [i[0] for i in final_outputs]
    fear_final_outputs = [i[1] for i in final_outputs]
    happiness_final_outputs = [i[2] for i in final_outputs]
    hatred_final_outputs = [i[3] for i in final_outputs]
    sadness_final_outputs = [i[4] for i in final_outputs]
    wonder_final_outputs = [i[5] for i in final_outputs]

    anger_targets = [i[0] for i in targets]
    fear_targets = [i[1] for i in targets]
    happiness_targets = [i[2] for i in targets]
    hatred_targets = [i[3] for i in targets]
    sadness_targets = [i[4] for i in targets]
    wonder_targets = [i[5] for i in targets]

    label_finals = [anger_final_outputs, fear_final_outputs, happiness_final_outputs, hatred_final_outputs, sadness_final_outputs, wonder_final_outputs]
    label_targets = [anger_targets, fear_targets, happiness_targets, hatred_targets, sadness_targets, wonder_targets]
    label = ["Anger", "Fear", "Happiness", "Hatred", "Sadness", "Wonder"]

    for i in range(len(label)):
        print(f"Accuracy for {label[i]}: {accuracy_score(label_targets[i], label_finals[i])}")
        print(f"Precision for {label[i]}: {precision_score(label_targets[i], label_finals[i])}")
        print(f"Recall for {label[i]}: {recall_score(label_targets[i], label_finals[i])}")
        print(f"F1 for {label[i]}: {f1_score(label_targets[i], label_finals[i])}")
        print("\n ------------------------------------------------------------- \n")