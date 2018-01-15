# -*- encoding:utf-8 -*-
import tensorflow as tf
import input_data
import tfrecord_manager
import os


data_path = '/home/lixiangyu/Desktop/train_data.csv'
logs_train_dir = '/home/lixiangyu/Documents/logs/'
tfrecord_path = 'train.tfrecord'
BATCH_SIZE = 128
CAPACITY = 2000
NUM_THREADS = 32


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
    if not os.path.exists(tfrecord_path):
        para, faultcode = input_data.get_data(data_path)
        tfrecord_manager.create_tfrecord(para, faultcode, tfrecord_path)

    para_batch, faultcode_batch = tfrecord_manager.read_tfrecord(tfrecord_path, BATCH_SIZE, CAPACITY, NUM_THREADS)

    p_data = para_batch
    fc_data = faultcode_batch

    l1 = addLayer(p_data, 100, 40, activate_function=tf.nn.relu)  # layer1: 100 x 40
    l2 = addLayer(l1, 40, 887, activate_function=None)  # layer2: 40 x 887

    # train_loss = tf.reduce_mean(tf.reduce_sum(tf.square((y_data - l2)), reduction_indices=[0]))
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=l2, labels=fc_data)
    train_loss = tf.reduce_mean(cross_entropy)
    train_op = tf.train.AdamOptimizer(0.01).minimize(train_loss)
    train_acc = tf.nn.in_top_k(l2, fc_data, 1)
    train_acc = tf.reduce_mean(tf.cast(train_acc, tf.float16))

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        with open('loss_accuracy.csv', 'w+') as fp:
            for step in range(10000):
                if coord.should_stop():
                    break
                _, tr_loss, tr_acc = sess.run([train_op, train_loss, train_acc])

                if step % 50 == 0:
                    print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tr_loss, tr_acc * 100))
                    fp.write(str(tr_loss))
                    fp.write(',')
                    fp.write(str(tr_acc * 100))
                    fp.write('\n')
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
