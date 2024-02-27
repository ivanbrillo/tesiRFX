import matplotlib.pyplot as plt

import helper
import numpy as np

import visualizer
import seaborn as sns

# plt.switch_backend('TkAgg')
#
# # set the default style
# sns.set()
# # plt.style.use("ggplot")
# vis = visualizer.Visualizer()
#
# plt.show()

list_data = helper.parser("./data")

print(list_data.shape)

