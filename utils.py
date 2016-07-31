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


def print_prediction(pred, label_file_path):
    """
    Print the top 5 prediction with labels.
    :param pred: The prediction 1d-array.
    :param label_file_path:  The label reference.
    """
    synset = [l.strip() for l in open(label_file_path).readlines()]

    # print prob
    pred = np.argsort(pred)[::-1]

    # Get top5 label
    top5 = [(synset[pred[i]], pred[pred[i]]) for i in range(5)]
    print("Top5: ", top5)
