'''Objectives:
- Objective 1: implement the multi-head attention from scratch

- Objective 2: use hugging face transformers as a feature extractor and imdb dataset
to do sentiment detection in pytorch
'''
import torchtext
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchtext import data

SEED = 407
TEXT = data.Field(tokenize = 'spacy', lower=True, include_lengths = True)
LABEL = data.LabelField(dtype = torch.float)

train_ds, test_ds = torchtext.datasets.IMDB.splits(TEXT, LABEL)
train_ds, valid_ds = train_ds.split(split_ratio=0.7)

MAX_VOCAB_SIZE = 10000
TEXT.build_vocab(train_data, max_size=10000, unk_init=torch.Tensor.normal_)
TEXT.build_vocab(train_data, 
                 max_size=MAX_VOCAB_SIZE,
                 vectors='glove.6B.300d',
                 unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)



train_dl = DataLoader(train_iter)
valid_dl = DataLoader(valid_iter)
test_dl = DataLoader(test_iter)




from transformers import BertModel, BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)

last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

with torch.no_grad():
    outputs = model(input_ids)

outputs = self.layer(outputs)


# to get the attention weights
config = BertConfig.from_pretrained("bert-base-cased", output_attentions=True, num_labels=2)
model = BertForSequenceClassification.from_pretrained("bert-base-cased", config=config)