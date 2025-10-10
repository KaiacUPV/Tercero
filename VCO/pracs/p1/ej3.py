from PIL import Image, ImageChops

# 1. Leer las imágenes
img1 = Image.open("cameraman.tif")
img2 = Image.open("moon.tif")

# 2. Redimensionar a 256x256
img1_resized = img1.resize((256, 256))
img2_resized = img2.resize((256, 256))

# 3. Operaciones aritméticas
# Suma
img_sum = ImageChops.add(img1_resized, img2_resized)
img_sum.show(title="Suma")

# Resta
img_sub = ImageChops.subtract(img1_resized, img2_resized)
img_sub.show(title="Resta")

# Diferencia absoluta
img_diff = ImageChops.difference(img1_resized, img2_resized)
img_diff.show(title="Diferencia absoluta")

# Combinación lineal: S = img1 * 1.8 – img2 * 1.2 + 128
import numpy as np

# Convertir a arrays numpy
arr1 = np.array(img1_resized, dtype=np.float32)
arr2 = np.array(img2_resized, dtype=np.float32)

# Calcular la combinación lineal
S = arr1 * 1.8 - arr2 * 1.2 + 128

# Limitar valores al rango [0, 255]
S = np.clip(S, 0, 255).astype(np.uint8)

# Convertir de nuevo a imagen y mostrar
img_S = Image.fromarray(S)
img_S.show(title="Combinación lineal S")
