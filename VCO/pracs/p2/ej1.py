import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
img = np.asarray(Image.open('c:/Users/Kai/Desktop/Tercero/VCO/pracs/p2/galaxia.jpg'))
hr, edges_r = np.histogram(img[:,:,0],256)
fig, axs = plt.subplots(1, 2, figsize=(12,4)) # Dos subplots en horizontal
axs[1].stairs(hr, edges_r, label='Red histogram', ec='r')
axs[1].set_title("Step Histograms")
axs[1].legend()
plt.sca(axs[0]) # Para poner en el subplot izquierdo la imagen
plt.imshow(img)
plt.show()