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
log_path = "./log/test"
model_path = "./models/cnn_28x28_model.npy"
# Remove logging path.
sh.rmtree(log_path, ignore_errors=True)

# Parameters
batch_size = 1

# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = 1.

print ("Loading pre-trained model ...")
restore_data = np.load(model_path).item()

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.constant(restore_data["weights"]["wc1"]),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.constant(restore_data["weights"]["wc2"]),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.constant(restore_data["weights"]["wd1"]),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.constant(restore_data["weights"]["out"])
}

biases = {
    'bc1': tf.constant(restore_data["biases"]["bc1"]),
    'bc2': tf.constant(restore_data["biases"]["bc2"]),
    'bd1': tf.constant(restore_data["biases"]["bd1"]),
    'out': tf.constant(restore_data["biases"]["out"])
}


def visualize_conv_layer(x, ix, iy, channels, cx=8):
    """
    Aggregate the feature maps to an image from the given tensor of a
    convolution layer.

    Reference:
    http://stackoverflow.com/questions/33802336/visualizing-output-of
    -convolutional-layer-in-tensorflow

    :param x:           The tensor of a convolution layer.
    :param ix:          The width.
    :param iy:          The height.
    :param channels:    The depth (channel number).
    :param cx:          The number of how many feature maps in a row.
    :return:            The aggregated feature map.
    """
    cy = channels / cx
    print("ix=%d, iy=%d, channels=%d, cx=%d, cy=%d" %
          (ix, iy, channels, cx, cy))

    # First slice off 1 image and remove the image dimension.
    img = tf.slice(x, [0, 0, 0, 0], [1, -1, -1, -1])
    img = tf.reshape(img, [iy, ix, channels])

    # Add a couple of pixels of zero padding around the image
    ix += 4
    iy += 4
    img = tf.image.resize_image_with_crop_or_pad(img, iy, ix)

    img = tf.reshape(img, [iy, ix, cy, cx])
    img = tf.transpose(img, perm=[2, 0, 3, 1])
    img = tf.reshape(img, [1, cy * iy, cx * ix, 1])

    return img


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
def conv_net(x, weights, biases):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    tf.image_summary("x", x, max_images=batch_size)

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    print ""
    print ("conv1=%s" % conv1)
    tf.image_summary("conv1",
                     visualize_conv_layer(conv1,
                                          conv1.get_shape().as_list()[1],
                                          conv1.get_shape().as_list()[2],
                                          conv1.get_shape().as_list()[3],
                                          8),
                     max_images=1)

    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    print ("maxpool2d(conv1)=%s" % conv1)
    tf.image_summary("maxpool2d(conv1)",
                     visualize_conv_layer(conv1,
                                          conv1.get_shape().as_list()[1],
                                          conv1.get_shape().as_list()[2],
                                          conv1.get_shape().as_list()[3],
                                          8),
                     max_images=1)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    print ""
    print ("conv2=%s" % conv2)
    tf.image_summary("conv2",
                     visualize_conv_layer(conv2,
                                          conv2.get_shape().as_list()[1],
                                          conv2.get_shape().as_list()[2],
                                          conv2.get_shape().as_list()[3],
                                          8),
                     max_images=1)

    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    print ("maxpool2d(conv2)=%s" % conv2)
    tf.image_summary("maxpool2d(conv2)",
                     visualize_conv_layer(conv2,
                                          conv2.get_shape().as_list()[1],
                                          conv2.get_shape().as_list()[2],
                                          conv2.get_shape().as_list()[3],
                                          8),
                     max_images=1)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2,
                     [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    out = tf.nn.softmax(out)
    print ""

    return out


# Construct model
pred = conv_net(x, weights, biases)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(init)

        # Merge all summaries into a single op
        merged_summary_op = tf.merge_all_summaries()
        # op to write logs to Tensorboard
        summary_writer = tf.train.SummaryWriter(log_path,
                                                graph=tf.get_default_graph())

        # Prepare the test data randomly.
        choose = np.random.randint(len(mnist.test.images))
        batch_x = mnist.test.images[choose].reshape([-1, 784])

        # Run the prediction.
        final_pred, summary = sess.run([pred, merged_summary_op],
                                       feed_dict={x: batch_x})
        print ("The outcome is %s" % final_pred)

        # Write logs at every iteration
        summary_writer.add_summary(summary)
        print ("Use \"tensorboard --logdir=./log\" to launch the TensorBoard.")
