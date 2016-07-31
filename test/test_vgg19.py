import numpy as np
import tensorflow as tf

import test_lib
import utils
from vgg19 import Vgg19

model = np.load(test_lib.model_vgg19).item()
print("The VGG model is loaded.")

imgs = utils.load_images("./img-airplane-224x224.jpg",
                         "./img-guitar-224x224.jpg",
                         "./img-puzzle-224x224.jpg",
                         "./img-tatoo-plane-224x224.jpg",
                         "./img-dog-224x224.jpg",
                         "./img-paper-plane-224x224.jpg",
                         "./img-pyramid-224x224.jpg",
                         "./img-tiger-224x224.jpg")
print("The input image(s) is loaded.")

# Design the graph.
graph = tf.Graph()
with graph.as_default():
    nn = Vgg19(model=model)

# Run the graph in the session.
with tf.Session(graph=graph) as sess:
    tf.initialize_all_variables().run()
    print("Tensorflow initialized all variables.")

    preds = sess.run(nn.preds,
                     feed_dict={
                         nn.inputRGB: imgs
                     })
    print("There you go the predictions.")

    for pred in preds:
        utils.print_prediction(pred, test_lib.label_vgg)
