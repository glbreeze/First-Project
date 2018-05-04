# -*- coding: utf-8

import os
from contextlib import contextmanager
from functools import partial
from operator import itemgetter
from multiprocessing.pool import ThreadPool
from datetime import datetime
import time
import re
from typing import List, Dict

import keras as ks
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.pipeline import make_pipeline, make_union, Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
from sklearn.base import TransformerMixin, BaseEstimator

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.merge import concatenate, dot

from keras.layers import Dense, Input, GRU, Conv1D, Embedding, Activation, LSTM, Dropout

"""
apply embedding and gru layer to name and category separately

"""

BASE_DIR = os.getcwd()
BASE_DIR = os.path.dirname(BASE_DIR)
vocab_size = 100000  # Max number of different word, i.e. model input dimension
maxlen = 60  # Max number of words kept at the end of each text
max_text_length = 60  ###################


@contextmanager
def timer(name):
    print('=' * 50)
    print('[[{}]] begin at {}'.format(name, datetime.now()))
    t0 = time.time()
    yield
    print('[[{}]] done in {:.0f} s'.format(name, time.time() - t0))
    print('*' * 50)


########################################################################################################################
def clean_str(text):
    try:
        text = ' '.join([w for w in text.split()[:max_text_length]])
        text = text.lower()
        text = re.sub(u"é", u"e", text)
        text = re.sub(u"ē", u"e", text)
        text = re.sub(u"è", u"e", text)
        text = re.sub(u"ê", u"e", text)
        text = re.sub(u"à", u"a", text)
        text = re.sub(u"â", u"a", text)
        text = re.sub(u"ô", u"o", text)
        text = re.sub(u"ō", u"o", text)
        text = re.sub(u"ü", u"u", text)
        text = re.sub(u"ï", u"i", text)
        text = re.sub(u"ç", u"c", text)
        text = re.sub(u"\u2019", u"'", text)
        text = re.sub(u"\xed", u"i", text)
        text = re.sub(u"w\/", u" with ", text)

        text = re.sub(u"[^a-z0-9]", " ", text)
        text = u" ".join(re.split('(\d+)', text))
        text = re.sub(u"\s+", u" ", text).strip()
        text = ''.join(text)
    except:
        text = np.NaN
    return text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df['name'] = df['name'].fillna('').apply(lambda s: clean_str(s))
    df['item_description'] = df['item_description'].fillna('').apply(lambda s: clean_str(s))

    df['brand_name'] = df['brand_name'].fillna('')
    df['category_name'] = df['category_name'].fillna('')
    return df


def load_data():
    train = pd.read_csv(os.path.join(os.path.join(BASE_DIR, 'data'), 'train_mock.tsv'),
                        sep='\t', encoding='utf-8')
    train = train[train['price'] > 0].reset_index(drop=True)
    train = preprocess(train)

    valid = pd.read_csv(os.path.join(os.path.join(BASE_DIR, 'data'), 'test_mock.tsv'),
                        sep='\t', encoding='utf-8')
    valid = valid[valid['price'] > 0].reset_index(drop=True)
    valid = preprocess(valid)
    return train, valid


###########################################    transformers   ##########################################################
class ColumnExtractor(TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):  # stateless transformer
        return self

    def transform(self, X):  # assumes X is a DataFrame
        Xcols = X[self.cols]
        return Xcols


class ToDict(TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):  # stateless transformer
        return self

    def transform(self, X):  # assumes X is a DataFrame
        Xdict = X.to_dict('records')
        return Xdict


class TextsToSequences(Tokenizer, BaseEstimator, TransformerMixin):
    """ Sklearn transformer to convert texts to indices list
    (e.g. [["the cute cat"], ["the dog"]] -> [[1, 2, 3], [1, 4]])"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, texts, y=None):
        self.fit_on_texts(texts.ix[:, 0].append(texts.ix[:, 1]))
        return self

    def transform(self, texts, y=None):
        return [np.array(self.texts_to_sequences(texts.ix[:, 0])), np.array(self.texts_to_sequences(texts.ix[:, 1]))]


sequencer = TextsToSequences(num_words=vocab_size)


class Padder(BaseEstimator, TransformerMixin):
    """ Pad and crop uneven lists to the same length.
    Only the end of lists longernthan the maxlen attribute are kept,
    and lists shorter than maxlen are left-padded with zeros
    Attributes  ----------
    maxlen: int  sizes of sequences after padding
    max_index: int.  maximum index known by the Padder, if a higher index is met during transform it is transformed to a 0
    """

    def __init__(self, maxlen=500):
        self.maxlen = maxlen
        self.max_index = None

    def fit(self, X, y=None):
        self.max_index = max(pad_sequences(X[0], maxlen=self.maxlen).max(),
                             pad_sequences(X[1], maxlen=self.maxlen).max()
                             )
        return self

    def transform(self, X, y=None):
        for i in range(2):
            X[i] = pad_sequences(X[i], maxlen=self.maxlen)
            X[i][X[i] > self.max_index] = 0
        return [X[0], X[1]]


padder = Padder(maxlen)

###################################### Define transformers ###############################
vec_text = Pipeline([
    ('extract', ColumnExtractor(['name', 'item_description'])),
    ('text_to_seq', sequencer),
    ('padding', padder)
])

vec_other = FeatureUnion([
    ('shippingCondition', Pipeline([
        ('extract', ColumnExtractor(['shipping', 'item_condition_id'])),
        ('ToDict', ToDict(['shipping', 'item_condition_id'])),
        ('DictVectorizer', DictVectorizer())
    ])),
    ('brandCate', Pipeline([
        ('extract', ColumnExtractor(['brand_name', 'category_name'])),
        ('ToDict', ToDict(['brand_name', 'category_name'])),
        ('DictVectorizer', DictVectorizer())
    ])),
])
y_scaler = StandardScaler()

########################################################################################################################
#################### model
########################################################################################################################
train, valid = load_data()
print('data is loaded')

def main():
    y_train = y_scaler.fit_transform(np.log1p(train['price'].values.reshape(-1, 1)))
    X_train_name, X_train_desc = vec_text.fit_transform(train)  ### .astype(np.float32)
    X_train_other = vec_other.fit_transform(train).astype(np.float32)
    print('X_train_name dim')
    print(X_train_name.shape)

    X_valid_name, X_valid_desc = vec_text.transform(valid)  ### .astype(np.float32)
    X_valid_other = vec_other.transform(valid).astype(np.float32)
    print('X_valid_name dim:')
    print(X_valid_name.shape)

    vocab_size_model = min(vocab_size, max(X_train_name.max() + 1, X_train_desc.max() + 1))

    ####################################################################################
    def fit_predict(xs, y_train, train_pred=False):
        X_train_name, X_train_desc, X_train_other, X_valid_name, X_valid_desc, X_valid_other = xs
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(
            intra_op_parallelism_threads=1, use_per_session_threads=1, inter_op_parallelism_threads=1,
            gpu_options=gpu_options, allow_soft_placement=True)

        with tf.Session(graph=tf.Graph(),config=config) as sess:
            name_input = Input(shape=(maxlen,), dtype='int32', name='name_input')
            desc_input = Input(shape=(maxlen,), dtype='int32', name='desc_input')

            embed_layer = Embedding(vocab_size_model, 256, name='embedding_layer')
            gru_layer = GRU(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=False, name='gru_layer')

            name_embed = embed_layer(name_input)
            name_gru = gru_layer(name_embed)
            desc_embed = embed_layer(desc_input)
            desc_gru = gru_layer(desc_embed)

            other_in = ks.Input(shape=(X_train_other.shape[1],), dtype='float32', sparse=True, name='other_input')
            other_seq = ks.layers.Dense(128, activation='relu', name='other_dense')(other_in)

            out1 = concatenate([name_gru, desc_gru, other_seq], name='concat0')
            out1 = Dense(128, activation='relu', name='dense1')(out1)
            out1 = Dense(64, activation='relu', name='dense2')(out1)
            out = Dense(1, name='dense3')(out1)

            model = ks.Model([name_input, desc_input, other_in], out)
            model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(lr=3e-3))
            print(model.summary())

            for i in range(4):
                with timer('epoch {}'.format(i + 1)):
                    model.fit([X_train_name, X_train_desc, X_train_other], y_train, batch_size=2 ** (11 + i), epochs=1,
                              validation_split=0.1)
            if train_pred:


class ToDict(TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):  # stateless transformer
        return self

    def transform(self, X):  # assumes X is a DataFrame
        Xdict = X.to_dict('records')
        return Xdict


class TextsToSequences(Tokenizer, BaseEstimator, TransformerMixin):
    """ Sklearn transformer to convert texts to indices list
    (e.g. [["the cute cat"], ["the dog"]] -> [[1, 2, 3], [1, 4]])"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, texts, y=None):
        self.fit_on_texts(texts.ix[:, 0].append(texts.ix[:, 1]))
        return self

    def transform(self, texts, y=None):
        return [np.array(self.texts_to_sequences(texts.ix[:, 0])), np.array(self.texts_to_sequences(texts.ix[:, 1]))]


sequencer = TextsToSequences(num_words=vocab_size)


class Padder(BaseEstimator, TransformerMixin):
    """ Pad and crop uneven lists to the same length.
    Only the end of lists longernthan the maxlen attribute are kept,
    and lists shorter than maxlen are left-padded with zeros
    Attributes  ----------
    maxlen: int  sizes of sequences after padding
    max_index: int.  maximum index known by the Padder, if a higher index is met during transform it is transformed to a 0
    """

    def __init__(self, maxlen=500):
        self.maxlen = maxlen
        self.max_index = None

    def fit(self, X, y=None):
        self.max_index = max(pad_sequences(X[0], maxlen=self.maxlen).max(),
                             pad_sequences(X[1], maxlen=self.maxlen).max()
                             )
        return self

    def transform(self, X, y=None):
        for i in range(2):
            X[i] = pad_sequences(X[i], maxlen=self.maxlen)
            X[i][X[i] > self.max_index] = 0
        return [X[0], X[1]]


padder = Padder(maxlen)

###################################### Define transformers ###############################
vec_text = Pipeline([
    ('extract', ColumnExtractor(['name', 'item_description'])),
    ('text_to_seq', sequencer),
    ('padding', padder)
])

vec_other = FeatureUnion([
    ('shippingCondition', Pipeline([
        ('extract', ColumnExtractor(['shipping', 'item_condition_id'])),
        ('ToDict', ToDict(['shipping', 'item_condition_id'])),
        ('DictVectorizer', DictVectorizer())
    ])),
    ('brandCate', Pipeline([
        ('extract', ColumnExtractor(['brand_name', 'category_name'])),
        ('ToDict', ToDict(['brand_name', 'category_name'])),
        ('DictVectorizer', DictVectorizer())
    ])),
])
y_scaler = StandardScaler()

########################################################################################################################
#################### model
########################################################################################################################
train, valid = load_data()


def main():
    y_train = y_scaler.fit_transform(np.log1p(train['price'].values.reshape(-1, 1)))
    X_train_name, X_train_desc = vec_text.fit_transform(train)  ### .astype(np.float32)
    X_train_other = vec_other.fit_transform(train).astype(np.float32)
    print('X_train_name dim')
    print(X_train_name.shape)

    X_valid_name, X_valid_desc = vec_text.transform(valid)  ### .astype(np.float32)
    X_valid_other = vec_other.transform(valid).astype(np.float32)
    print('X_valid_name dim:')
    print(X_valid_name.shape)

    vocab_size_model = min(vocab_size, max(X_train_name.max() + 1, X_train_desc.max() + 1))

    ####################################################################################
    def fit_predict(xs, y_train, train_pred=False):
        X_train_name, X_train_desc, X_train_other, X_valid_name, X_valid_desc, X_valid_other = xs
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(
            intra_op_parallelism_threads=1, use_per_session_threads=1, inter_op_parallelism_threads=1,
            gpu_options=gpu_options, allow_soft_placement=True)

        with tf.Session(graph=tf.Graph()) as sess:
            name_input = Input(shape=(maxlen,), dtype='int32', name='name_input')
            desc_input = Input(shape=(maxlen,), dtype='int32', name='desc_input')

            embed_layer = Embedding(vocab_size_model, 256, name='embedding_layer')
            gru_layer = GRU(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=False, name='gru_layer')

            name_embed = embed_layer(name_input)
            name_gru = gru_layer(name_embed)
            desc_embed = embed_layer(desc_input)
            desc_gru = gru_layer(desc_embed)

            other_in = ks.Input(shape=(X_train_other.shape[1],), dtype='float32', sparse=True, name='other_input')
            other_seq = ks.layers.Dense(128, activation='relu', name='other_dense')(other_in)

            out1 = concatenate([name_gru, desc_gru, other_seq], name='concat0')
            out1 = Dense(128, activation='relu', name='dense1')(out1)
            out1 = Dense(64, activation='relu', name='dense2')(out1)
            out = Dense(1, name='dense3')(out1)

            model = ks.Model([name_input, desc_input, other_in], out)
            model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(lr=3e-3))
            print(model.summary())

            for i in range(4):
                with timer('epoch {}'.format(i + 1)):
                    model.fit([X_train_name, X_train_desc, X_train_other], y_train, batch_size=2 ** (11 + i), epochs=1,
                              validation_split=0.1)
            if train_pred:
                return model.predict([X_valid_name, X_valid_desc, X_valid_other])[:, 0], \
                       model.predict([X_train_name, X_train_desc, X_train_other])[:, 0]
            else:
                return model.predict([X_valid_name, X_valid_desc, X_valid_other])[:, 0]

                ######################################################################################

    print('model is defined')

    xs1 = [X_train_name, X_train_desc, X_train_other, X_valid_name, X_valid_desc, X_valid_other]

    dic = dict()
    for i in range(4):
        pred_name = 'pred' + str(i + 1)
        if i == 0:
            dic[pred_name], ytrain_pred = fit_predict(xs1, y_train, True)
        else:
            dic[pred_name] = fit_predict(xs1, y_train)

        y_pred = dic[pred_name]
        y_pred = np.expm1(y_scaler.inverse_transform(y_pred.reshape(-1, 1))[:, 0])
        print('Valid RMSLE ROUND{}: {:.4f}'.format(i + 1, np.sqrt(mean_squared_log_error(valid['price'], y_pred))))

        if i == 0:
            result = pd.concat([valid, pd.DataFrame(y_pred, columns=['pred'])], axis=1)
            result.to_csv(os.path.join(BASE_DIR, 'out/gru4_1round.tsv'), sep="\t")

            ytrain_pred = np.expm1(y_scaler.inverse_transform(ytrain_pred.reshape(-1, 1))[:, 0])
            print(
                'Valid RMSLE OF TRAINING: {:.4f}'.format(np.sqrt(mean_squared_log_error(train['price'], ytrain_pred))))

    y_pred = np.mean(np.array([dic['pred1'], dic['pred2'], dic['pred3'], dic['pred4']]), axis=0)
    y_pred = np.expm1(y_scaler.inverse_transform(y_pred.reshape(-1, 1))[:, 0])
    print('Valid RMSLE 4ROUNDS: {:.4f}'.format(np.sqrt(mean_squared_log_error(valid['price'], y_pred))))

    result = pd.concat([valid, pd.DataFrame(y_pred, columns=['pred'])], axis=1)
    result.to_csv(os.path.join(BASE_DIR, 'out/gru4_4round.tsv'), sep="\t")


if __name__ == '__main__':
    main()
