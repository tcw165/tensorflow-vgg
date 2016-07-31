import numpy as np
from scipy import ndimage


def load_images(*paths):
    """
    Load multiple images.
    :param paths: The image paths.
    """
    imgs = []
    for path in paths:
        img = ndimage.imread(path, mode="RGB").astype(float)
        imgs.append(img)
    return imgs


def print_prediction(pred, label_file_path, img_path=None):
    """
    Print the top 5 prediction with labels.
    :param pred: The prediction 1d-array.
    :param label_file_path:  The label reference.
    """
    synset = [l.strip() for l in open(label_file_path).readlines()]

    # Sort the prediction in ascending order and get the indices.
    indices = np.argsort(pred)[::-1]

    # Get top5 label
    if img_path:
        print("%s -> %s" % (img_path,
                            [(synset[indices[i]], pred[indices[i]])
                             for i in range(5)]))
    else:
        print([(synset[indices[i]], pred[indices[i]]) for i in range(5)])
