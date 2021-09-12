# encoding=utf-8

# =================
# imports 
# =================

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertForSequenceClassification
import os

def read_or_create_split(split, force_generate=False):

    if split == 'train':
        file = 'data/train.csv'
    else:
        file = 'data/test.csv'

    if not os.path.isfile(file) or force_generate:

        print('crating splits train and test')
        fake = pd.read_csv('data/Fake.csv')[['text']]
        real = pd.read_csv('data/Real.csv')[['text']]
        print('Original shape: Fake', fake.shape)
        print('Original shape: Real', real.shape)

        # create labels
        fake['label'] = 1
        real['label'] = 0

        # remove source from real for same structure as fake
        real.loc[:, 'text'] = real.text.str.split('-').str[1:].str.join(' ').str.lower()
        fake.loc[:, 'text'] = fake.text.str.lower()

        data = pd.concat([fake, real], axis=0, ignore_index=True).sample(frac=1)
        data.loc[data.text == '', 'text'] = None
        data = data.dropna(how='any')
        data = data.drop_duplicates()
        # split into train, val, test
        train_data, test_data = train_test_split(data, test_size=.1, random_state=99)
        print('Train samples: ', len(train_data))
        print('Test samples: ', len(test_data))
        # save files
        train_data.to_csv('data/train.csv', index=False)
        test_data.to_csv('data/test.csv', index=False)

    return pd.read_csv(file)

train_data = read_or_create_split('train')
test_data = read_or_create_split('test')
print(train_data.isna().sum())
print(test_data.isna().sum())

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_data.text.tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_data.text.tolist(), truncation=True, padding=True)

# build tf datasets
train_ds = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_data.label))
test_ds = tf.data.Dataset.from_tensor_slices((dict(test_encodings), test_data.label))

batch_size = 16
train_ds = train_ds.batch(batch_size)
test_ds = test_ds.batch(batch_size)

model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=train_data.label.nunique())


optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
model.compile(optimizer=optimizer, loss=model.compute_loss, metrics='accuracy') # can also use any keras loss fn
model.fit(train_ds, epochs=1, validation_data=(test_ds))

model.save_pretrained('models')
tokenizer.save_pretrained('models')
