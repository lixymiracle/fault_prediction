# -*- encoding:utf-8 -*-
import tensorflow as tf
import numpy as np
import input_data
import os

# data_path = '/home/lixiangyu/Desktop/2018.csv'
data_path = '/home/lixiangyu/Desktop/train_data.csv'
# data_path = '/home/lixiangyu/Desktop/processed_data3.csv'
logs_train_dir = '/home/lixiangyu/Documents/logs/'
BATCH_SIZE = 32
CAPACITY = 2000


def addLayer(inputData, inSize, outSize, activate_function=None):
    weights = tf.Variable(tf.random_normal([inSize, outSize], dtype=tf.float32))
    basis = tf.Variable(tf.zeros([1, outSize]) + 0.1, dtype=tf.float32)
    weights_plus_b = tf.matmul(inputData, weights) + basis
    if activate_function is None:
        ans = weights_plus_b
    else:
        ans = activate_function(weights_plus_b)
    return ans


def train():
    para, faultcode = input_data.get_data(data_path)
    para_batch, faultcode_batch = input_data.get_batch(para, faultcode, BATCH_SIZE, CAPACITY)

    x_data = para_batch # 转为列向量
    # noise = np.random.normal(0, 0.05, x_data.shape)
    y_data = faultcode_batch

    # x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # 转为列向量
    # noise = np.random.normal(0, 0.05, x_data.shape)
    # y_data = np.square(x_data) + 0.5 + noise

    xs = tf.placeholder(tf.float32, [None, 100])  # 样本数未知，特征数为1，占位符最后要以字典形式在运行中填入
    ys = tf.placeholder(tf.int32, [None, 1])

    l1 = addLayer(x_data, 100, 20, activate_function=tf.nn.relu)  # relu是激励函数的一种
    l2 = addLayer(l1, 20, 887, activate_function=None)

    # train_loss = tf.reduce_mean(tf.reduce_sum(tf.square((y_data - l2)), reduction_indices=[0]))
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=l2, labels=y_data)
    train_loss = tf.reduce_mean(cross_entropy)
    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(train_loss)
    train_acc = tf.nn.in_top_k(l2, y_data, 1)
    train_acc = tf.reduce_mean(tf.cast(train_acc, tf.float16))

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    # loss = tf.reduce_mean(tf.reduce_sum(tf.square((y_data - l2)), reduction_indices=[0]))  # 需要向相加索引号，reduce执行跨纬度操作
    #
    # train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 选择梯度下降法
    #
    init = tf.global_variables_initializer()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in range(10000):
            if coord.should_stop():
                break
            _, tr_loss, tr_acc = sess.run([train_op, train_loss, train_acc])
            # aaa, bbb, cc = sess.run([l2, y_data, cross_entropy])
            if step % 50 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tr_loss, tr_acc * 100))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
            if step % 2000 == 0 or (step + 1) == 10000:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
    except tf.errors.OutOfRangeError:
        print("done")
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    train()

# for i in range(10000):
#     sess.run(train, feed_dict={xs: x_data, ys: y_data})
#     # para_batch_v, faultcode_batch_v = sess.run([para_batch, faultcode_batch])
#     if i % 50 == 0:
#         print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
