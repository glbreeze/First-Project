"""
TFIDF DNN with matrix interaction
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
    X_train_name, X_train_text, X_train_num, X_valid_name, X_valid_text, X_valid_num = xs
   # config = tf.ConfigProto(
   #     intra_op_parallelism_threads=1, use_per_session_threads=1, inter_op_parallelism_threads=1)
    with tf.Session( graph=tf.Graph() ) as sess :
        ks.backend.set_session(sess)

        dense_size = 128

        def multiply(x):
            x1_prime = tf.reshape(x[0], (-1, dense_size, 1))
            x2_prime = tf.reshape(x[1], (-1, dense_size, 1))
            x2_transpose = tf.transpose(x2_prime, perm=[0, 2, 1])
            return tf.matmul(x1_prime, x2_transpose)

        name_input = ks.Input(shape=(X_train_name.shape[1],), dtype='float32', sparse=True, name='name_input')
        name_in = ks.layers.Dense(dense_size, activation='relu', name='name_input_process')(name_input)

        text_input = ks.Input(shape=((X_train_text.shape[1]),), dtype='float32', sparse=True, name='text_input')
        text_in = ks.layers.Dense(dense_size, activation='relu', name='text_input_process')(text_input)

        name_text_dot = ks.layers.Lambda(lambda x: multiply(x), output_shape=(dense_size, dense_size),
                                         name='name_text_matrix_product0')([name_in, text_in])
        name_text_dot1 = ks.layers.Reshape((name_text_dot.get_shape()[1].value * name_text_dot.get_shape()[1].value,
                                            ), name='name_text_matrix_product1'
                                           )(name_text_dot)

        other_input = ks.Input(shape=((X_train_num.shape[1]),), dtype='float32', sparse=True, name='num_input')
        other_in = ks.layers.Dense(6, activation='relu', name='num_input_process')(other_input)

        name_text_other_concat = ks.layers.concatenate([name_in, text_in, name_text_dot1, other_in], name='concat_all')
        out = ks.layers.Dense(64, activation='relu', name='dense1')(name_text_other_concat)
        out = ks.layers.Dense(64, activation='relu', name='dense2')(out)
        out = ks.layers.Dense(1)(out)
        model = ks.Model([name_input, text_input, other_input], out)

        model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(lr=3e-3))

        for i in range(2):
            # with timer(f'epoch {i + 1}'):
            model.fit([X_train_name, X_train_text, X_train_num], y_train, batch_size=2 ** (11 + i), epochs=1)# validation_split=0.1)
        return model.predict([X_valid_name, X_valid_text, X_valid_num])[:, 0]

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

vec_name = FeatureUnion([
        ('continuous', Pipeline([
            ('extract', ColumnExtractor('name')),
            ('tfidf', Tfidf(max_features=100000, token_pattern='\w+'))
        ])),
    ])
vec_text = Pipeline([
            ('extract', ColumnExtractor('text')),
            ('tfidf', Tfidf(max_features=100000, token_pattern='\w+', ngram_range=(1, 2)))
        ])
vec_num =  Pipeline([
            ('extract', ColumnExtractor(['shipping', 'item_condition_id'])),
            ('ToDict', ToDict(['shipping', 'item_condition_id'])),
            ('DictVectorizer', DictVectorizer())
        ])

y_scaler = StandardScaler()

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


def main():
    train, valid = load_data()

    y_train = y_scaler.fit_transform(np.log1p(train['price'].values.reshape(-1, 1)))
    X_train_name = vec_name.fit_transform(train).astype(np.float32)
    X_train_text = vec_text.fit_transform(train).astype(np.float32)
    X_train_num = vec_num.fit_transform(train).astype(np.float32)
    print('X_train_name shape')
    print(X_train_name.shape)
    # del train
    X_valid_name = vec_name.transform(valid).astype(np.float32)
    X_valid_text = vec_text.transform(valid).astype(np.float32)
    X_valid_text = vec_text.transform(valid).astype(np.float32)
    X_valid_num = vec_num.transform(valid).astype(np.float32)

###########################################################################################################
    Xb_train_name, Xb_train_text, Xb_train_num = [x.astype(np.bool).astype(np.float32) for x in [X_train_name, X_train_text, X_train_num]]
    Xb_valid_name, Xb_valid_text, Xb_valid_num = [x.astype(np.bool).astype(np.float32) for x in [X_valid_name, X_valid_text, X_valid_num]]
    xbs = [Xb_train_name, Xb_train_text, Xb_train_num, Xb_valid_name, Xb_valid_text, Xb_valid_num]
    xs =  [X_train_name, X_train_text, X_train_num, X_valid_name, X_valid_text, X_valid_num]

    y_pred1 = fit_predict(xs, y_train)
    y_pred = np.expm1(y_scaler.inverse_transform(y_pred1.reshape(-1, 1))[:, 0])
    print('Avg Valid RMSLE1: {:.4f}'.format(np.sqrt(mean_squared_log_error(valid['price'], y_pred))))
    result1 = pd.concat([valid, pd.DataFrame(y_pred, columns=['pred'])], axis=1)
    result1.to_csv(os.path.join(BASE_DIR, 'out/dnn_i2a_1.tsv'), sep="\t")

    y_pred2 = fit_predict(xs, y_train)

    y_pred1b = fit_predict(xbs, y_train)
    y_pred = np.expm1(y_scaler.inverse_transform(y_pred1b.reshape(-1, 1))[:, 0])
    print('Avg Valid RMSLE1b: {:.4f}'.format(np.sqrt(mean_squared_log_error(valid['price'], y_pred))))
    result1b = pd.concat([valid, pd.DataFrame(y_pred, columns=['pred'])], axis=1)
    result1b.to_csv(os.path.join(BASE_DIR, 'out/dnn_i2a_1b.tsv'), sep="\t")

    y_pred2b = fit_predict(xbs, y_train)

    y_pred = np.mean(np.array([y_pred1, y_pred2, y_pred1b, y_pred2b]), axis=0)
    y_pred = np.expm1(y_scaler.inverse_transform(y_pred.reshape(-1, 1))[:, 0])
    print('Avg Valid RMSLE avg: {:.4f}'.format(np.sqrt(mean_squared_log_error(valid['price'], y_pred))))
    result = pd.concat([valid, pd.DataFrame(y_pred, columns=['pred'])], axis=1)
    result.to_csv(os.path.join(BASE_DIR, 'out/dnn_i2a_avg.tsv'),sep="\t")

    xs_train=[X_train_name, X_train_text, X_train_num, X_train_name, X_train_text, X_train_num]
    ytrain_pred1 = fit_predict(xs_train, y_train)
    ytrain_pred1 = np.expm1(y_scaler.inverse_transform(ytrain_pred1.reshape(-1, 1))[:, 0])
    print('Avg train RMSLE1: {:.4f}'.format(np.sqrt(mean_squared_log_error(train['price'], ytrain_pred1))))
    result_train1 = pd.concat([train, pd.DataFrame(ytrain_pred1, columns=['pred'])], axis=1)
    result_train1.to_csv(os.path.join(BASE_DIR, 'out/dnn_i2a_train1.tsv'),sep="\t")

if __name__ == '__main__':
    main()

