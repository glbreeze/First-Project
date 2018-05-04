"""
original TFIDF model
"""

import os; os.environ['OMP_NUM_THREADS'] = '1'
from contextlib import contextmanager
from functools import partial
from operator import itemgetter
from multiprocessing.pool import ThreadPool
import time
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
from sklearn.base import TransformerMixin

BASE_DIR = os.getcwd()
BASE_DIR = os.path.dirname(BASE_DIR)

# @contextmanager
def timer(name):
    t0 = time.time()
    yield
    #print(f'[{name}] done in {time.time() - t0:.0f} s')

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df['name'] = df['name'].fillna('') + ' ' + df['brand_name'].fillna('')
    df['text'] = (df['item_description'].fillna('') + ' ' + df['name'] + ' ' + df['category_name'].fillna(''))
    return df

def on_field(f: str, *vec) -> Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)

def to_records(df: pd.DataFrame) -> List[Dict]:
    return df.to_dict(orient='records')

def fit_predict(xs, y_train) -> np.ndarray:
    X_train, X_test = xs
   # config = tf.ConfigProto(
   #     intra_op_parallelism_threads=1, use_per_session_threads=1, inter_op_parallelism_threads=1)
    with tf.Session( graph=tf.Graph() ) as sess :
        ks.backend.set_session(sess)
        model_in = ks.Input(shape=(X_train.shape[1],), dtype='float32', sparse=True)
        out = ks.layers.Dense(192, activation='relu')(model_in)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(1)(out)
        model = ks.Model(model_in, out)
        model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(lr=3e-3))
        for i in range(3):
            # with timer(f'epoch {i + 1}'):
                model.fit(x=X_train, y=y_train, batch_size=2**(11 + i), epochs=1, verbose=0)
        return model.predict(X_test)[:, 0]

class ColumnExtractor(TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xcols = X[self.cols]
        return Xcols

class ToDict(TransformerMixin):
        def __init__(self, cols):
            self.cols = cols

        def fit(self, X, y=None):
            # stateless transformer
            return self

        def transform(self, X):
            # assumes X is a DataFrame
            Xdict = X.to_dict('records')
            return Xdict


vectorizer = FeatureUnion([
        ('continuous', Pipeline([
            ('extract', ColumnExtractor('name')),
            ('tfidf', Tfidf(max_features=100000, token_pattern='\w+'))
        ])),
        ('text', Pipeline([
            ('extract', ColumnExtractor('text')),
            ('tfidf', Tfidf(max_features=100000, token_pattern='\w+', ngram_range=(1, 2)))
        ])),
        ('shippingCondition', Pipeline([
            ('extract', ColumnExtractor(['shipping', 'item_condition_id'])),
            ('ToDict', ToDict(['shipping', 'item_condition_id'])),
            ('DictVectorizer', DictVectorizer())
        ])),
    ])

y_scaler = StandardScaler()


def load_data():
    train = pd.read_csv(os.path.join(os.path.join(BASE_DIR, 'Challenge\Data'), 'train_mock.tsv'),
                            sep='\t', encoding='utf-8')
    train = train[train['price'] > 0].reset_index(drop=True)
    train = preprocess(train)

    valid = pd.read_csv(os.path.join(os.path.join(BASE_DIR, 'data'), 'test_mock.tsv'),
                        sep='\t', encoding='utf-8')
    valid = valid[valid['price'] > 0].reset_index(drop=True)
    valid = preprocess(valid)

    return train, valid

def main():
    train, valid =load_data()
    y_train = y_scaler.fit_transform(np.log1p(train['price'].values.reshape(-1, 1)))
    X_train = vectorizer.fit_transform(preprocess(train)).astype(np.float32)
    print(f'X_train: {X_train.shape} of {X_train.dtype}')
    #del train
    X_valid = vectorizer.transform(preprocess(valid)).astype(np.float32)
    print('X_valid shape:')
    print(X_valid.shape)

    #################
    Xb_train, Xb_valid = [x.astype(np.bool).astype(np.float32) for x in [X_train, X_valid]]
    xsb = [Xb_train, Xb_valid]
    xs = [X_train, X_valid]
    y_pred1 = fit_predict(xs, y_train)
    y_pred2 = fit_predict(xs, y_train)
    y_pred1b = fit_predict(xsb, y_train)
    y_pred2b = fit_predict(xsb, y_train)

    y_pred = np.mean(np.array([y_pred1, y_pred2, y_pred1b, y_pred2b]), axis=0)
    y_pred = np.expm1(y_scaler.inverse_transform(y_pred.reshape(-1, 1))[:, 0])
    print('Avg Valid RMSLE: {:.4f}'.format(np.sqrt(mean_squared_log_error(valid['price'], y_pred))))

    valid = valid.reset_index(drop=True)
    result = pd.concat([valid, pd.DataFrame(y_pred, columns=['pred'])], axis=1)
    result.to_csv(os.path.join(BASE_DIR, 'out/result_avg.csv'))

    y_pred1 = np.expm1(y_scaler.inverse_transform(y_pred1.reshape(-1, 1))[:, 0])
    print('Avg Valid RMSLE1: {:.4f}'.format(np.sqrt(mean_squared_log_error(valid['price'], y_pred1))))
    result1 = pd.concat([valid, pd.DataFrame(y_pred1, columns=['pred'])], axis=1)
    result.to_csv(os.path.join(BASE_DIR, 'outresult_1.csv'))

    y_pred1b = np.expm1(y_scaler.inverse_transform(y_pred1b.reshape(-1, 1))[:, 0])
    print('Avg Valid RMSLE1: {:.4f}'.format(np.sqrt(mean_squared_log_error(valid['price'], y_pred1b))))
    result1 = pd.concat([valid, pd.DataFrame(y_pred1b, columns=['pred'])], axis=1)
    result.to_csv(os.path.join(BASE_DIR, 'out/result_1b.csv'))

    xs_train=[X_train, X_train]
    ytrain_pred1 = fit_predict(xs_train, y_train)
    ytrain_pred1 = np.expm1(y_scaler.inverse_transform(ytrain_pred1.reshape(-1, 1))[:, 0])
    print('Avg Valid RMSLE1: {:.4f}'.format(np.sqrt(mean_squared_log_error(train['price'], ytrain_pred1))))
    result_train1 = pd.concat([train, pd.DataFrame(ytrain_pred1, columns=['pred'])], axis=1)
    result.to_csv(os.path.join(BASE_DIR, 'out/result_trian1.csv'))

#############################
