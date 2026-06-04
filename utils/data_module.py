import pandas as pd
import numpy as np

# --------------------
# Missing value handle
# --------------------
def missing_value_cleanup(dataset: pd.DataFrame) -> pd.DataFrame:
  """
  Checks for NA values in a dataset and replaces them by the mean if they're numeric, otherwise drop
      @param dataset: the dataset to clean

      @return the new dataset cleaned from NA values
  """
  new_dataset = dataset.copy()
  ## show na values
  print(new_dataset.isna().sum())

  for column in new_dataset.columns:
    if new_dataset[column].dtype in ['int64', 'float64']:
      if new_dataset[column].isnull().any():
        mean_value = new_dataset[column].mean()
        new_dataset[column] = new_dataset[column].fillna(mean_value)

  ## dropping na values in categorical features
  new_dataset = new_dataset.dropna()
  return new_dataset

def missing_value_dropout(dataset: pd.DataFrame) -> pd.DataFrame:
  """
  Checks for NA values in a dataset and drops every row that contains them
      @param dataset: the dataset to clean

      @return the new dataset cleaned from NA values
  """
  new_dataset = dataset.copy()
  ## show na values
  print(new_dataset.isna().sum())

  new_dataset = new_dataset.dropna()
  return new_dataset

# --------
# Encoding
# --------
def one_hot_encoding(dataset, columns):
  """
  Applies one hot encoding to the columns of the given dataset.
      @param dataset: the dataset to encode
      @param columns: the column names of the dataset to be encoded

      @return encoded dataset
  """
  encoded_dataset = dataset.copy()

  for column in columns:
    oh_encoding = pd.get_dummies(dataset[column], prefix=column, dtype=int)
    encoded_dataset.pop(column)
    encoded_dataset = pd.concat([encoded_dataset, oh_encoding], axis=1)

  return encoded_dataset

def ordinal_encoding(dataset, columns):
  """
  Applies ordinal encoding to the columns of the given dataset.
      @param dataset: the dataset to encode
      @param columns: the column names of the dataset to be encoded

      @return encoded dataset
  """
  encoded_dataset = dataset.copy()

  for column in columns:
    encoded_dataset[column] = pd.factorize(encoded_dataset[column])[0]

  return encoded_dataset

def mapping_encoding(dataset, columns, map_func):
  """
  Applies mapping to the columns of the given dataset.
      @param dataset: the dataset to encode
      @param columns: the column names of the dataset to be mapped
      @param map_func: the mapping function to be applied to the columns

      @return encoded dataset
  """
  encoded_dataset = dataset.copy()

  for column in columns:
    encoded_dataset[column] = encoded_dataset[column].apply(map_func)

  return encoded_dataset

# -------------
# Normalization
# -------------
def normalize_min_max(column):
  """
  Apply min/max normalization to the given column
      @param column: the column to be normalized

      @return the normalized column
  """
  max_value = np.max(column)
  min_value = np.min(column)
  return (column - min_value)/(max_value - min_value)

def z_normalization(column):
  """
  Apply Z normalization to the given column
      @param column: the column to be normalized

      @return the normalized column
  """
  return (column - column.mean()) / column.std()

def normalize_columns(dataset, columns, norm_func):
  """
  Normalizes the columns of the given dataset with norm_func.
      @param dataset: dataset to be normalized
      @param columns: the column names of the dataset to be normalized
      @param norm_func: normalizing function to be used

      @return the normalized dataset
  """
  new_dataset = dataset.copy()

  for column in columns:
    new_dataset[column] = norm_func(new_dataset[column])

  return new_dataset

# ---------------
# Class balancing
# ---------------
def under_oversample_classes(dataset, target_column, target_class, oversample=False):
  """
  Undersample or oversample for the target_column to balance to target_class.
      @param dataset: the dataset to be balanced
      @param target_column: the column name of the target
      @param target_class: the class of the desired number of data
      @param oversample: set to true to oversample instead of undersample. Defaults to False

      @return the under or oversampled dataset
  """
  new_dataset = dataset.copy()

  values = new_dataset[target_column].value_counts()
  reference_class_count = values[target_class]

  sampled_dataset = new_dataset[new_dataset[target_column] == target_class]

  classes = list(new_dataset[target_column].unique())
  classes.remove(target_class)

  for data_class in classes:
    sampled_class = new_dataset[new_dataset[target_column] == data_class].sample(reference_class_count, replace=oversample, random_state=seed)
    sampled_dataset = pd.concat([sampled_dataset, sampled_class])

  sampled_dataset = sampled_dataset.sample(frac=1, random_state=seed).reset_index(drop=True)

  return sampled_dataset