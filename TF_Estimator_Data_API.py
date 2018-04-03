from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import sys
import numpy as np

import tensorflow as tf

_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]

########################################################################################################################

model_dir = 'D:/Work/Learning/Deep and wide model/census_model'

model_type = 'wide_deep'

train_epochs = 10

epochs_per_eval = 2

batch_size = 40

train_data = 'D:/Work/Learning/Deep and wide model/data/census_data/adult.data'

test_data = 'D:/Work/Learning/Deep and wide model/data/census_data/adult.test'

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}

########################################################################################################################
age = tf.feature_column.numeric_column('age')
education_num = tf.feature_column.numeric_column('education_num')
capital_gain = tf.feature_column.numeric_column('capital_gain')
capital_loss = tf.feature_column.numeric_column('capital_loss')
hours_per_week = tf.feature_column.numeric_column('hours_per_week')

education = tf.feature_column.categorical_column_with_vocabulary_list(
      'education', [
          'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
          'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
          '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
      'marital_status', [
          'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
          'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

relationship = tf.feature_column.categorical_column_with_vocabulary_list(
      'relationship', [
          'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
          'Other-relative'])

workclass = tf.feature_column.categorical_column_with_vocabulary_list(
      'workclass', [
          'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
          'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

# To show an example of hashing:
occupation = tf.feature_column.categorical_column_with_hash_bucket(
      'occupation', hash_bucket_size=1000)

# Transformations.
age_buckets = tf.feature_column.bucketized_column(
      age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

# Wide columns and deep columns.
base_columns = [
      education, marital_status, relationship, workclass, occupation, age_buckets,
  ]

crossed_columns = [
      tf.feature_column.crossed_column(
          ['education', 'occupation'], hash_bucket_size=1000),
      tf.feature_column.crossed_column(
          [age_buckets, 'education', 'occupation'], hash_bucket_size=1000),
  ]

wide_columns = base_columns + crossed_columns

deep_columns = [
      age,
      education_num,
      capital_gain,
      capital_loss,
      hours_per_week,
      tf.feature_column.indicator_column(workclass),
      tf.feature_column.indicator_column(education),
      tf.feature_column.indicator_column(marital_status),
      tf.feature_column.indicator_column(relationship),
      # To show an example of embedding
      tf.feature_column.embedding_column(occupation, dimension=8),
  ]

########################################################################################################################

  # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
  # trains faster than GPU for this model.
run_config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(device_count={'GPU': 0}))

model_mixed = tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 75, 50, 25],
        config=run_config)

model_linear = tf.estimator.LinearClassifier(
        model_dir=model_dir,
        feature_columns=wide_columns,
        config=run_config)

model_DNN = tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=[100, 75, 50, 25],
        config=run_config)

########################################################################################################################

def input_fn(data_file, num_epochs, shuffle, batch_size=40):

  def parse_csv(value):
    print('Parsing', data_file)
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    labels = features.pop('income_bracket')     # response variable
    return features, tf.equal(labels, '>50K')

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

  dataset = dataset.map(parse_csv, num_parallel_calls=5)  # number of elements to processed in parallel

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)          # repeat的功能就是将整个序列重复多次
  dataset = dataset.batch(batch_size)           # batch就是将多个元素组合成batch，每个batch中有batch_size行

  iterator = dataset.make_one_shot_iterator()
  features, labels = iterator.get_next()
  return features, labels


########################################################################################################################
#def main(unused_argv):
  # Clean up the model directory if present
shutil.rmtree(model_dir, ignore_errors=True)

# Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
model_linear.train(input_fn=lambda: input_fn(train_data, epochs_per_eval, True, batch_size))    # epoch_per_eval 数据被重复了几次

model.train(input_fn=lambda: input_fn(train_data, epochs_per_eval, True, batch_size))
