import os
import sys

# Add the module path.
sys.path.append(os.path.abspath("../"))

model_vgg19 = os.path.abspath("../models/vgg19.npy")
label_vgg = os.path.abspath("../models/synset.txt")
