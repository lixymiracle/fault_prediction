import random
import pandas as pd


raw_train_file = '/home/lixiangyu/Desktop/20003001#2017-03.csv'
# raw_train_file = '/home/lixiangyu/Desktop/20180104.csv'
# processed_train_file = '/home/lixiangyu/Desktop/test_data.csv'
processed_train_file = '/home/lixiangyu/Desktop/train_data.csv'
# processed_train_file = '/home/lixiangyu/Desktop/p_data.csv'
processed_test_file = '/home/lixiangyu/Desktop/test_data.csv'

train_num = 500000
test_num = 100000


removed_cols_index = [23, 24, 30, 31, 32, 33, 38, 43, 44, 58]
saved_cols_index = list(filter(lambda i: i not in removed_cols_index, range(5, 116)))

df = pd.read_csv(raw_train_file, usecols=saved_cols_index)

column_index = df.columns.tolist()
for ci in column_index[:-1]:
    X_min = min(df[ci])
    X_max = max(df[ci])
    if X_min == X_max:
        df[ci] = 0
    else:
        df[ci] = (df[ci] - X_min) / (X_max - X_min)

rows = random.sample(list(df.index), train_num)
df_train = df.loc[rows]
df_test = df.drop(rows)
df_test = df_test.sample(n=test_num)
# df = df.sample(frac=0.2)

df_train.to_csv(processed_train_file, index=False)
df_test.to_csv(processed_test_file, index=False)
