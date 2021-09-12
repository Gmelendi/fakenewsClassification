# encoding=utf-8

# =================
# imports 
# =================

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertForSequenceClassification, TFTrainer, TFTrainingArguments

fake = pd.read_csv('data/Fake.csv')[['text']]
real = pd.read_csv('data/Real.csv')[['text']]
print('Original shape: Fake', fake.shape)
print('Original shape: Real', real.shape)

# create labels
fake['label'] = 1
real['label'] = 0

fake = fake.dropna(subset=['text'])
real = real.dropna(subset=['text'])


data = pd.concat([fake, real], axis=0).sample(frac=1)
# split into train, val, test
train_data, val_data = train_test_split(data, test_size=.1)
train_data, test_data = train_test_split(train_data, test_size=.1)
print('Train samples: ', len(train_data))
print('Val samples: ', len(val_data))
print('Test samples: ', len(test_data))

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_data.text.tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val_data.text.tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_data.text.tolist(), truncation=True, padding=True)

# build tf datasets
train_ds = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_data.label))
val_ds = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_data.label))
test_ds = tf.data.Dataset.from_tensor_slices((dict(test_encodings), test_data.label))


training_args = TFTrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

with training_args.strategy.scope():
    model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

trainer = TFTrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_ds,         # training dataset
    eval_dataset=val_ds             # evaluation dataset
)

trainer.train()
