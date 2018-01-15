import math
import numpy as np
import tensorflow as tf
import os
import input_data
import bpnn
import tfrecord_manager


def test():
    log_dir = '/home/lixiangyu/Documents/logs/'
    test_dir = '/home/lixiangyu/Desktop/test_data.csv'
    tfrecord_path = 'test.tfrecord'
    BATCH_SIZE = 64
    n_test = 50000
    CAPACITY = 2000
    NUM_THREADS = 32

    with tf.Graph().as_default():

        if not os.path.exists(tfrecord_path):
            test_para, test_faultcode = input_data.get_data(test_dir)
            tfrecord_manager.create_tfrecord(test_para, test_faultcode, tfrecord_path)

        para_test_batch, faultcode_test_batch = tfrecord_manager.read_tfrecord(tfrecord_path, BATCH_SIZE, CAPACITY, NUM_THREADS)

        # logits = model.alexnet(test_batch, BATCH_SIZE, n_classes, keep_prob)
        l1 = bpnn.addLayer(para_test_batch, 100, 40, activate_function=tf.nn.relu)  # relu是激励函数的一种
        logits = bpnn.addLayer(l1, 40, 887, activate_function=None)
        top_k_op = tf.nn.in_top_k(logits, faultcode_test_batch, 1)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                num_iter = int(math.ceil(n_test / BATCH_SIZE))
                true_count = 0
                total_sample_count = num_iter * BATCH_SIZE
                step = 0
                with open('submission.csv', 'w+') as fp:
                    while step < num_iter and not coord.should_stop():
                        predictions = sess.run([top_k_op])
                        for p in predictions:
                            for pp in p:
                                fp.write(str(pp))
                                fp.write('\n')
                        true_count += np.sum(predictions)
                        step += 1
                        precision = true_count / total_sample_count

                print('precision = %.4f' % precision)
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)


if __name__ == '__main__':
    test()
