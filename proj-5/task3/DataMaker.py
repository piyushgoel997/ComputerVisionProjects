# This file contains the code for part A of task 3, which is to create the csv files out of the data.
# The way this code is written it'll work with any number of categories and any number of examples. Provided the files
# have are following the naming convention category name followed by an underscore, the rest could be anything.

import os

import cv2
import numpy as np
import pandas as pd


def create_dataframe_from_data(folder, invert=True):
    cat = []
    data = []
    category_map = {}

    for filename in os.listdir(folder):
        # set the category
        c = filename.split("_")[0]
        if c not in category_map:
            category_map[c] = len(category_map)
        cat.append(category_map[c])

        # load and process the image
        img = cv2.imread(os.path.join(folder, filename))
        img = cv2.resize(img, (28, 28))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # invert the image and then set the data
        if invert:
            data.append(((255.0 - np.array(img).astype("float32")) / 255).flatten())
        else:
            data.append((np.array(img).astype("float32") / 255).flatten())

    data = pd.DataFrame(data)
    cat = pd.DataFrame(np.array(cat).transpose())
    return data, cat, category_map


# save to csv files
data, cat, _ = create_dataframe_from_data("greek/")
data.to_csv("data.csv")
cat.to_csv("cat.csv")
