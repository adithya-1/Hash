from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os
import matplotlib
import webcolors
image = cv2.cv2.imread('newBalance.png')
image = cv2.cv2.cvtColor(image, cv2.cv2.COLOR_BGR2RGB)


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

modified_image = cv2.cv2.resize(image, (600, 400), interpolation = cv2.cv2.INTER_AREA)
modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
clf = KMeans(n_clusters = 3)
labels = clf.fit_predict(modified_image)
counts = Counter(labels)
center_colors = clf.cluster_centers_

# We get ordered colors by iterating through the keys
ordered_colors = [center_colors[i] for i in counts.keys()]
hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
rgb_colors = [ordered_colors[i] for i in counts.keys()]
plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
print(hex_colors)
file1 = open("nike.txt","a+") 
file1.write(str(hex_colors[0])+"\n\n\n")
file1.write(str(hex_colors[1])+"\n\n\n")
file1.write(str(hex_colors[2])+"\n\n\n")

