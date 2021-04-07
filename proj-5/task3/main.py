import numpy as np
import pandas as pd
import tensorflow as tf

from task3.DataMaker import create_dataframe_from_data

# B create the truncated model
loaded_model = tf.keras.models.load_model("../task1/my_model.h5")
truncated_model = tf.keras.Model(inputs=loaded_model.input, outputs=loaded_model.get_layer(index=5).output)
truncated_model.summary()

# load the data
data = pd.read_csv("data.csv", index_col=0)
data = np.array(data)
data = data.reshape((data.shape[0], 28, 28, 1))
cat = pd.read_csv("cat.csv", index_col=0)
cat = np.array(cat)

# prints (1, 128) and ensures that the output is a vector of size 128
print(truncated_model.predict(data[0:1]).shape)

# C project the greek symbols into embedding space
embedding = truncated_model.predict(data)


# calculates the sum of squared distance between two vectors (np arrays).
def ssd(a, b):
    return np.sum((a - b) ** 2)


# D distances in the embedding space
for row in embedding:
    print(ssd(row, embedding[0]))
# pattern explained in the report

# E
# load data
test_data, test_cat, category_map = create_dataframe_from_data("test_files", invert=False)
test_data = np.array(test_data)
test_data = test_data.reshape((test_data.shape[0], 28, 28, 1))

test_pred = truncated_model.predict(test_data)

index_to_cat = [""] * len(category_map)
for k, v in category_map.items():
    index_to_cat[v] = k

# find the best category match for each test point
for t in test_pred:
    min_ssd = 99999999
    c = -1
    for i, e in enumerate(embedding):
        s = ssd(e, t)
        if s < min_ssd:
            min_ssd = s
            c = index_to_cat[cat[i][0]]
    print(c)
