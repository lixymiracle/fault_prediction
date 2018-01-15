import csv
import os


raw_train_file = '/home/lixiangyu/Desktop/20003001#2017-03.csv'
# raw_train_file = '/home/lixiangyu/Desktop/20180104.csv'
processed_train_file = '/home/lixiangyu/Desktop/processed_data.csv'

removed_cols_index = [23, 24, 30, 31, 32, 33, 38, 43, 44, 58]
saved_cols_index = list(filter(lambda i: i not in removed_cols_index, range(5, 116)))
new_row = []

# reader = pd.read_csv(raw_train_file)
with open(raw_train_file, 'r') as f1, open(processed_train_file, 'w', newline='') as f2:
    reader = csv.reader(f1)
    writer = csv.writer(f2)
    for row in reader:
        new_row = list(row[i] for i in saved_cols_index)
        writer.writerow(new_row)
        # print(list(row[i] for i in saved_cols_index))
        # print(str(reader.line_num) + " ", end="")
        # for i in saved_cols_index:
        #     print(row[i] + " ", end="")
        # print("\n")
