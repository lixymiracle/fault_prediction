import tensorflow as tf
import os
import rbm
import input_data
import tfrecord_manager

data_path = '/home/lixiangyu/Desktop/train_data.csv'
logs_train_dir = '/home/lixiangyu/Documents/logs_rbm/'
tfrecord_path = 'train.tfrecord'
BATCH_SIZE = 128
CAPACITY = 2000
NUM_THREADS = 32


def build_model(X, w1, b1, wo, bo):
    h1 = tf.nn.sigmoid(tf.matmul(X, w1) + b1)
    model = tf.nn.sigmoid(tf.matmul(h1, wo) + bo)
    return model


def init_weight(shape):
    return tf.Variable(tf.random_normal(shape, mean=0.0, stddev=0.01))


def init_bias(dim):
    return tf.Variable(tf.zeros([dim]))


def train():
    # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    #
    # X = tf.placeholder("float", [None, 784])
    # Y = tf.placeholder("float", [None, 10])

    # rbm_layer = rbm.RBM("mnist", 784, 500)

    # for i in range(10):
    #     print("RBM CD: ", i)
    #     rbm_layer.cd1(trX)

    # rbm_w, rbm_vb, rbm_hb = rbm_layer.cd1(trX)
    #
    # wo = init_weight([500, 10])
    # bo = init_bias(10)
    # py_x = build_model(X, rbm_w, rbm_hb, wo, bo)

    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
    # train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
    # predict_op = tf.argmax(py_x, 1)
    #
    # sess = tf.Session()
    # init = tf.initialize_all_variables()
    # sess.run(init)
    #
    # for i in range(10):
    #     for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
    #         sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
    #     print(i, np.mean(np.argmax(teY, axis=1) ==
    #                      sess.run(predict_op, feed_dict={X: teX, Y: teY})))

    if not os.path.exists(tfrecord_path):
        para, faultcode = input_data.get_data(data_path)
        tfrecord_manager.create_tfrecord(para, faultcode, tfrecord_path)

    para_batch, faultcode_batch = tfrecord_manager.read_tfrecord(tfrecord_path, BATCH_SIZE, CAPACITY, NUM_THREADS)

    p_data = para_batch
    fc_data = faultcode_batch

    rbm_layer = rbm.RBM("faultprediction", 100, 500)

    rbm_w, rbm_vb, rbm_hb = rbm_layer.cd1(p_data)

    wo = init_weight([500, 887])
    bo = init_bias(887)
    logits = build_model(p_data, rbm_w, rbm_hb, wo, bo)

    # l1 = addLayer(p_data, 100, 40, activate_function=tf.nn.relu)  # layer1: 100 x 40
    # l2 = addLayer(l1, 40, 887, activate_function=None)  # layer2: 40 x 887

    # train_loss = tf.reduce_mean(tf.reduce_sum(tf.square((y_data - l2)), reduction_indices=[0]))
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=fc_data)
    train_loss = tf.reduce_mean(cross_entropy)
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(train_loss)
    train_acc = tf.nn.in_top_k(logits, fc_data, 1)
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
        for step in range(10000):
            if coord.should_stop():
                break
            _, tr_loss, tr_acc = sess.run([train_op, train_loss, train_acc])

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
