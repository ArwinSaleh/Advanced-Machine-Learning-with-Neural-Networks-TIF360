import os
import pickle
import numpy as np


def split(data, train_size=0.8):

    data_length = data.shape[0]
    train_length = int(train_size * data_length)

    train = data[0:train_length, :]
    val = data[train_length:data_length]

    return train, val

train, val = split(X, train_size=0.8)


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])
  return np.array(data), np.array(labels)
