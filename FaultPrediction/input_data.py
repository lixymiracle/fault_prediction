# -*- encoding:utf-8 -*-
import csv
import tensorflow as tf
import numpy as np


def get_data(file_path):
    para = []
    faultcode = []

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num != 1:
                para.append([float(i) for i in row[:-1]])
                faultcode.append(int(row[-1]))

    return para, faultcode


def get_batch(para, faultcode, batch_size, capacity):
    para = tf.cast(para, tf.float32)
    faultcode = tf.cast(faultcode, tf.int32)
    input_queue = tf.train.slice_input_producer([para, faultcode], shuffle=False)
    para_batch, faultcode_batch = tf.train.batch(input_queue,  # list
                                                 batch_size=batch_size,
                                                 num_threads=64,
                                                 capacity=capacity)  # 队列中最多能容纳的个数

    return para_batch, faultcode_batch


def test():
    BATCH_SIZE = 10
    CAPACITY = 256
    train_dir = '/home/lixiangyu/Desktop/2018.csv'
    para_list, faultcode_list = get_data(train_dir)
    para_batch, faultcode_batch = get_batch(para_list, faultcode_list, BATCH_SIZE, CAPACITY)

    with tf.Session() as sess:
        i = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop() and i < 1:
                para, faultcode = sess.run([para_batch, faultcode_batch])

                # just test one batch
                for j in np.arange(BATCH_SIZE):
                    print('para%d: %s' % (j, faultcode[j]))
                i += 1

        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        coord.join(threads)


# test()

# p, f = get_data('/home/lixiangyu/Desktop/2018.csv')
# print(f)
