# -*- encoding:utf-8 -*-
import tensorflow as tf
import input_data
import os


def create_tfrecord(para, faultcode, filepath):
    writer = tf.python_io.TFRecordWriter(path=filepath)
    for i in range(len(para)):
        # para_raw = para[i].tostring()

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "para_raw": tf.train.Feature(float_list=tf.train.FloatList(value=para[i])),
                    "faultcode": tf.train.Feature(int64_list=tf.train.Int64List(value=[faultcode[i]]))
                }
            )
        )
        writer.write(record=example.SerializeToString())

    writer.close()
    print("create tfrecord finish------")


def read_tfrecord(filepath, batch_size, capacity, num_threads):
    filename_queue = tf.train.string_input_producer([filepath], num_epochs=None)
    # create a reader from file queue
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example yyp
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'para_raw': tf.FixedLenFeature([100], tf.float32),
                                           'faultcode': tf.FixedLenFeature([], tf.int64)
                                       })
    para = features['para_raw']
    faultcode = features['faultcode']

    para_batch, faultcode_batch = tf.train.shuffle_batch([para, faultcode], batch_size=batch_size,
                                                         capacity=capacity, min_after_dequeue=1000,
                                                         num_threads=num_threads)

    return para_batch, faultcode_batch


def test(datapath, tfrecord_path):
    if not os.path.exists(tfrecord_path):
        para, faultcode = input_data.get_data(datapath)
        create_tfrecord(para, faultcode, tfrecord_path)

    para_batch, faultcode_batch = read_tfrecord(tfrecord_path, 20, 200, 2)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    tf.train.start_queue_runners(sess=sess)
    para_val, faultcode_val = sess.run([para_batch, faultcode_batch])
    print('first batch:')
    print('para_val: ', para_val)
    print('faultcode_val', faultcode_val)
    print('len para', para_val.shape)
    para_val, faultcode_val = sess.run([para_batch, faultcode_batch])
    print('second batch:')
    print('para_val: ', para_val)
    print('faultcode_val', faultcode_val)


if __name__ == '__main__':
    data_path = '/home/lixiangyu/Desktop/p_data.csv'
    tfrecord_path = 'test.tfrecord'
    test(data_path, tfrecord_path)

# p, f = input_data.get_data('/home/lixiangyu/Desktop/p_data.csv')
# print(len(p))
# create_tfrecord(p, f)
# test()
# print(f)
