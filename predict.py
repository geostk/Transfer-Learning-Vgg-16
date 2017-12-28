import tensorflow as tf
import numpy as np
from datalab import DataLabTest
from vgg16 import vgg16
import matplotlib.pyplot as plt
from make_file import make_sub


def predict(model_path, batch_size):
    model, params = vgg16(fine_tune_last=True, n_classes=2)
    X = model['input']
    Y_hat = tf.nn.softmax(model['out'])

    saver = tf.train.Saver()

    dl_test = DataLabTest('./datasets/test_set/')
    test_gen = dl_test.generator()

    Y = []
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        for i in range(12500//batch_size+1):
            y = sess.run(Y_hat, feed_dict={X: next(test_gen)})
            #print(y.shape, end='   ')
            Y.append(y[:,0, 0, 1])
            print('Complete: {}%'.format(round(len(Y) / dl_test.max_len * 100, 2)), end='\r')
    Y = np.concatenate(Y)

    print()
    print('Total Predictions: '.format(Y.shape))
    return Y


if __name__ == '__main__':
    Y = predict('./model/vgg16-dog-vs-cat.ckpt', 16)
    np.save('out.npy', Y)
    make_sub('sub_1.csv')
