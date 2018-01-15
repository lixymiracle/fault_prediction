# -*- coding: utf-8 -*-
import tensorflow as tf

# output file name string to a queue
filename_queue = tf.train.string_input_producer(['test.tfrecord'], num_epochs=None)
# create a reader from file queue
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
# get feature from serialized example yyp
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'a': tf.FixedLenFeature([], tf.float32),
                                       'b': tf.FixedLenFeature([2], tf.int64),
                                       'c': tf.FixedLenFeature([], tf.string)
                                   })
a_out = features['a']
b_out = features['b']
# c_out = features['c']
c_raw_out = features['c']
# c_raw_out = tf.sparse_to_dense(features['c'])
c_out = tf.decode_raw(c_raw_out, tf.uint8)
c_out = tf.reshape(c_out, [2, 3])

print(a_out)
print(b_out)
print(c_out)
# c_out = tf.reshape(c_out, [2, 3])
a_batch, b_batch, c_batch = tf.train.shuffle_batch([a_out, b_out, c_out], batch_size=3,
                                                   capacity=200, min_after_dequeue=100, num_threads=2)
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

tf.train.start_queue_runners(sess=sess)
a_val, b_val, c_val = sess.run([a_batch, b_batch, c_batch])
# print(a_val, b_val, c_val)
print('first batch:')
print('  a_val:', a_val)
print('  b_val:', b_val)
print('  c_val:', c_val)
a_val, b_val, c_val = sess.run([a_batch, b_batch, c_batch])
print('second batch:')
print('  a_val:', a_val)
print('  b_val:', b_val)
print('  c_val:', c_val)
