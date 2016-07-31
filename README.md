# Tensorflow VGG

This is a [Tensorflow](https://github.com/tensorflow/tensorflow) implemention of VGG forked from [tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg) repo.

The main change from the original repo are:

* The `Vgg` class won't load the VGG model in the constructor so that you are allowed to share the model among multiple `Vgg` instances.
* The `Vgg` class is now able to do training and prediction and you could shared a model among multiple `Vgg` instance. That makes you more flexible to design the algorithm infrastructure.

To use the VGG networks, you should download the `*.npy` files from <a href="https://dl.dropboxusercontent.com/u/50333326/vgg16.npy">VGG16</a> or <a href="https://dl.dropboxusercontent.com/u/50333326/vgg19.npy">VGG19</a>.

Usage
-----

It is still under development, please stay tuned. This is a rough example for the prediction.

```python
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
```
