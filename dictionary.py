# dictionary for numbers

import pandas as pd
import numpy as np
import string

# load the dataset as with pandas
train_lab = pd.read_csv("../sign-language-mnist/sign_mnist_train.csv")

# dictonary for numbers of the labels
un_labels = np.unique(train_lab.iloc[:, 0])
letters = np.delete(np.array(list(string.ascii_lowercase[0:25])), 9)
s = pd.Series(letters, index=un_labels)
dict_letters = s.to_dict()

# dictionary for prediction -> delete entry with j
s = pd.Series(letters)
dict_pred = s.to_dict()

del letters, s, train_lab, un_labels