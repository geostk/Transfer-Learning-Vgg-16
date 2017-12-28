import tensorflow as tf
from vgg16 import vgg16
import numpy as np
import os
from datalab import DataLabTrain


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train(n_iters):
    model, params = vgg16(fine_tune_last=True, n_classes=2)
    X = model['input']
    Z = model['out']
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 2])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z[:, 0, 0, :], labels=Y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        try:
            sess.run(tf.global_variables_initializer())
            for i in range(n_iters):
                dl = DataLabTrain('./datasets/train_set/')
                train_gen = dl.generator()
                dev_gen = DataLabTrain('./datasets/dev_set/').generator()
                for X_train, Y_train in train_gen:
                    print('Samples seen: '.format(dl.cur_index), end='\r')
                    sess.run(train_step, feed_dict={X: X_train, Y: Y_train})
                print()
                l = 0
                count = 0
                for X_test, Y_test in dev_gen:
                    count += 1
                    l += sess.run(loss, feed_dict={X: X_test, Y: Y_test})

                print('Epoch: {}\tLoss: {}'.format(i, l/count))
                saver.save(sess, './model/vgg16-dog-vs-cat.ckpt')
                print("Model Saved")

        finally:
            sess.close()

if __name__ == '__main__':
    train(n_iters=1)
