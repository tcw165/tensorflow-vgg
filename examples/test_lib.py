import os
import sys

# Add the module path.
sys.path.append(os.path.abspath("../"))

tf_log_path = os.path.abspath("./log/test")
model_vgg19 = os.path.abspath("../models/vgg19.npy")
model_vgg16 = os.path.abspath("../models/vgg16.npy")
label_vgg = os.path.abspath("../models/synset.txt")
