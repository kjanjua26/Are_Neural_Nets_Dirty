import tensorflow as tf
import utils
import network
import alexnet

NROFCLASSES = 2
X = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name="input")
Y = tf.placeholder(tf.float32, shape=(None, NROFCLASSES), name="labels")
global_step = tf.Variable(0, trainable=False, name='global_step')
batch_size = 64
num_epochs = 1501

def build_network(input_images, labels):
    logits = alexnet.alexnet_v2(input_images)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
    return loss, accuracy

def train(x_train, x_val, y_train, y_val):
    loss, accuracy = build_network(X, Y)
    optimizer = tf.train.AdamOptimizer(0.0001) 
    train_op = optimizer.minimize(loss, global_step=global_step)
    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for i in range(num_epochs):
        batch_images, batch_labels = utils.next_batch(batch_size, x_train, y_train)
        _, train_loss, train_acc = sess.run([train_op, loss, accuracy], feed_dict={X:batch_images, Y:batch_labels})
        print("Step: {} Train Loss: {} Train Acc: {}".format(i, train_loss, train_acc))
        if i % 500 == 0:
            val_acc, val_loss = sess.run([accuracy, loss], feed_dict={X:x_val, Y:y_val})
            print("")
            print("Step: {} Val Loss: {} Val Acc: {}".format(i, val_loss, val_acc))
            save_path = saver.save(sess, "model/model-epoch{}.ckpt".format(i))
            print("Model saved for epoch # {}".format(i))
            print("")

if __name__ == "__main__":
    x_train, x_val, y_train, y_val = utils.train_test_split_data(NROFCLASSES)
    train(x_train, x_val, y_train, y_val)