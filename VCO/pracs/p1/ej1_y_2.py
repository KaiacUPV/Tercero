# Mostrar imagen original y versión con 16 colores
from PIL import Image

# Abrir imagen original
im = Image.open("galaxia.tiff")
im.show(title="Original")

#Reducir a 16 colores usando paleta adaptativa 
im_16 = im.convert('P', palette=Image.ADAPTIVE, colors=16) 

im_16.show(title="16 colores") 


# Transformaciones estándar
# 1. Rotar 90 grados
im_rotated = im.rotate(90, expand=True)
im_rotated.show(title="Rotada 90°")

# 2. Voltear horizontalmente
im_flipped = im.transpose(Image.FLIP_LEFT_RIGHT)
im_flipped.show(title="Volteada horizontalmente")

# 3. Convertir a niveles de grises
im_gray = im.convert('L')
im_gray.show(title="Niveles de grises")


# 4. Redimensionar a la mitad
im_small = im.resize((im.width // 2, im.height // 2))
im_small.show(title="Redimensionada a la mitad")


# Mostrar los componentes R, G y B de una imagen RGB como imágenes de grises
im_rgb = Image.open("AloeVera.jpg")
r, g, b = im_rgb.split()
r.show(title="Componente Rojo (R)")
g.show(title="Componente Verde (G)")
b.show(title="Componente Azul (B)")


