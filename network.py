import tensorflow as tf
import tensorflow.contrib.layers as initializers
slim = tf.contrib.slim
import tflearn

def slim_conv_net(input_images, NROFCLASSES):
    with slim.arg_scope([slim.conv2d], kernel_size=3, padding='SAME'):
        with slim.arg_scope([slim.max_pool2d], kernel_size=2):
            x = slim.conv2d(input_images, num_outputs=32, weights_initializer=initializers.xavier_initializer(),
                            scope='conv1_1')
            x = slim.conv2d(x, num_outputs=32, weights_initializer=initializers.xavier_initializer(), biases_initializer=tf.zeros_initializer(), scope='conv1_2')
            x = slim.max_pool2d(x, scope='pool1')
            x = slim.conv2d(x, num_outputs=64, weights_initializer=initializers.xavier_initializer(), biases_initializer=tf.zeros_initializer(), scope='conv2_1')
            x = slim.conv2d(x, num_outputs=64, weights_initializer=initializers.xavier_initializer(), biases_initializer=tf.zeros_initializer(), scope='conv2_2')
            x = slim.max_pool2d(x, scope='pool2')
            x = slim.conv2d(x, num_outputs=128, weights_initializer=initializers.xavier_initializer(), biases_initializer=tf.zeros_initializer(), scope='conv3_1')
            x = slim.conv2d(x, num_outputs=128, weights_initializer=initializers.xavier_initializer(), biases_initializer=tf.zeros_initializer(), scope='conv3_2')
            x = slim.max_pool2d(x, scope='pool3')
            x = slim.flatten(x, scope='flatten')
            feature = slim.fully_connected(x, num_outputs=128, activation_fn=None, scope='fc1')
            x = tflearn.prelu(feature)
            x = slim.fully_connected(x, num_outputs=NROFCLASSES, activation_fn=None, scope='fc2')
    return x
