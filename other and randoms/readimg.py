import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

img = pd.read_csv('img.csv')
img = img *255
plt.imshow(img)
plt.show()