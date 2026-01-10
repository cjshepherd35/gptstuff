import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
# from torchtext import data, datasets
# import tqdm
import os

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
# import transformers
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification
import gpt

pd.set_option('future.no_silent_downcasting', True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
df = pd.read_csv('IMDBDataset.csv')
df['sentiment'] = df['sentiment'].replace('positive', 1)
df['sentiment'] = df['sentiment'].replace('negative', 0 )

train_texts = df.iloc[:40_000]['review'].values

test_texts = df.iloc[40_000:]['review'].values

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)

test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)
class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings)

train_dataset = IMDbDataset(train_encodings)
test_dataset = IMDbDataset(test_encodings)


num_epochs = 8
print(f'- nr. of training examples {len(train_dataset)}')
mx = 512

model = gpt.Transformer(k=16, heads=4, depth=4,seq_length=16 , num_tokens=50_000 , next_seq=16).to('cuda:0')

opt = torch.optim.Adam(lr=0.0001, params=model.parameters())
seen = 0

for e in range(num_epochs):

    print(f'\n epoch {e+1}')
    model.train(True)
    
    for i in range(400):

        opt.zero_grad()
        
        #idk if this is right, using batch[inpu_ids] and labels as batch[labels]
        input = train_dataset[i:i+10]["input_ids"][:16].to('cuda:0')
        if input.size(0) > mx:
            input = input[:, :mx]
        out = model(input)
        loss = F.nll_loss(out, train_dataset[i:i+10]["input_ids"][1:17])

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        seen += input.size(0)
        i+=10
        if((i %100) ==0):
            print('ran 100')


#     with torch.no_grad():

#         model.train(False)
#         tot, cor= 0.0, 0.0

#         for batch in test_iter:

#             input = batch['input_ids'].to('cuda:0')
            
#             if input.size(1) > mx:
#                 input = input[:, :mx]
#             out = model(input).argmax(dim=1)

#             tot += float(input.size(0))
#             cor += float((label == out).sum().item())
#         acc = cor / tot
#         print(f'accuracy: {acc}')