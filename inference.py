import tensorflow as tf
import network, utils
import cv2
import numpy as np

NROFCLASSES = 2
X = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name="input")
Y = tf.placeholder(tf.float32, shape=(None, NROFCLASSES), name="labels")

def inference(img):
    print("For: ", img)
    images = []
    img = cv2.imread(img)
    img = cv2.resize(img, (32, 32))
    images.append(img)
    images = np.asarray(images)
    images = utils.normalize(images)
    output = network.slim_conv_net(X, NROFCLASSES)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state("model")
        saver.restore(sess, "model/model-epoch500.ckpt")
        result = sess.run(output, feed_dict={X:images})
        result = tf.nn.softmax(result)
        inf_result = tf.argmax(result, 1)
        class_result = sess.run(result)
        out = sess.run(inf_result)
        print(out, class_result)

if __name__ == "__main__":
    inference("Dataset/Train/SFW/9.png")