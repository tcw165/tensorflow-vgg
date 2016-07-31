"""
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

import shutil as sh

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

print("Downloading MNIST data ...")
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Log directory path.
log_path = "./log/train"
model_path = "./models/cnn_28x28_model.npy"
# Remove logging path.
sh.rmtree(log_path, ignore_errors=True)

# Parameters
learning_rate = 0.001
training_iters = 200000
# training_iters = 1000
batch_size = 128
display_step = 100
save_step = 1000

# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    # tf.image_summary("x", x, max_images=batch_size)

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # tf.image_summary("conv1",
    #                  tf.reshape(conv1[:, :, :, 1], [-1, 28, 28, 1]),
    #                  max_images=1)
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2,
                     [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(init)
        step = 1

        # Create a summary to monitor cost tensor
        tf.scalar_summary("loss", cost)
        # Create a summary to monitor accuracy tensor
        tf.scalar_summary("accuracy", accuracy)
        # Merge all summaries into a single op
        merged_summary_op = tf.merge_all_summaries()
        # op to write logs to Tensorboard
        summary_writer = tf.train.SummaryWriter(log_path,
                                                graph=tf.get_default_graph())

        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            # Run optimization op (backprop)
            _, summary = sess.run([optimizer, merged_summary_op],
                                  feed_dict={x: batch_x, y: batch_y,
                                             keep_prob: dropout})

            # Write logs at every iteration
            summary_writer.add_summary(summary, step * batch_size)

            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                  y: batch_y,
                                                                  keep_prob:
                                                                      1.})
                print "Iter " + str(step * batch_size) + ", Minibatch Loss= " \
                      + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc)

            if step % save_step == 0:
                # Save the model
                model_data = {
                    "weights": {
                        # 5x5 conv, 1 input, 32 outputs
                        'wc1': weights["wc1"].eval(),
                        # 5x5 conv, 32 inputs, 64 outputs
                        'wc2': weights["wc2"].eval(),
                        # fully connected, 7*7*64 inputs, 1024 outputs
                        'wd1': weights["wd1"].eval(),
                        # 1024 inputs, 10 outputs (class prediction)
                        'out': weights["out"].eval()
                    },
                    "biases": {
                        'bc1': biases["bc1"].eval(),
                        'bc2': biases["bc2"].eval(),
                        'bd1': biases["bd1"].eval(),
                        'out': biases["out"].eval()
                    }
                }
                # np.save(model_path,
                #         np.array(model_data.items(),
                #                  dtype=np.dtype))
                np.save(model_path, model_data)
                print ("Save the model to %s" % model_path)

            step += 1
        print "Optimization Finished!"

        # Calculate accuracy for 256 mnist test images
        print "Testing Accuracy:", \
            sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                          y: mnist.test.labels[:256],
                                          keep_prob: 1.})
