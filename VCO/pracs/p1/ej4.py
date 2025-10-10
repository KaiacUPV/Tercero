
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

# Leer la imagen y convertir a escala de grises
im = Image.open("amapola.jpg").convert('L')
arr = np.array(im)

# Crear la malla para la superficie
x = np.arange(arr.shape[1])
y = np.arange(arr.shape[0])
X, Y = np.meshgrid(x, y)

# Graficar la superficie 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, arr, cmap='viridis', edgecolor='none')
ax.set_title('Superficie 3D de amapola.jpg en escala de grises')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Intensidad')
plt.show()
