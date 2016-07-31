# Tensorflow VGG

This is a [Tensorflow](https://github.com/tensorflow/tensorflow) implemention of VGG forked from [tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg) repo.

The main change from the original repo are:

* The `Vgg` class won't load the VGG model in the constructor so that you are allowed to share the model among multiple `Vgg` instances.
* The `Vgg` class is now able to do training and prediction and you could shared a model among multiple `Vgg` instance. That makes you more flexible to design the algorithm infrastructure.

To use the VGG networks, you should download the `*.npy` files from <a href="https://dl.dropboxusercontent.com/u/50333326/vgg16.npy">VGG16</a> or <a href="https://dl.dropboxusercontent.com/u/50333326/vgg19.npy">VGG19</a> and put them under the `models` directory.

Usage
-----

It is still under development, please stay tuned. This is a rough example for the prediction.

###Use It for Predicting

First, load the model (around 500MB) to the memory.

```python
model = np.load("/path/to/your/vgg19.npy").item()
print("The VGG model is loaded.")
```

New the `Vgg19` instance in the scope of your `tf.Graph`. Run the *graph* in a `tf.Session`.

```python
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
```

You could also add a `tf.train.SummaryWriter` to summarize whatever you want. Then enter the command, `tensorboard --log=/path/to/your/log/dir`, to see the summary in the web page.

```python
# Run the graph in the session.
with tf.Session(graph=graph) as sess:
    tf.initialize_all_variables().run()
    print("Tensorflow initialized all variables.")

    # The OP to write logs to Tensorboard.
    summary_writer = tf.train.SummaryWriter("/path/to/your/log/dir",
                                            graph=sess.graph)

    preds = sess.run(nn.preds,
                     feed_dict={
                         nn.inputRGB: imgs
                     })
```
