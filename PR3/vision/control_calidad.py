import cv2
import numpy as np
import tkinter as tk

# ==========================
# CONFIGURACIÓN
# ==========================

CAMERA_INDEX = 0  # 0 suele ser la webcam del portátil

# Área mínima y máxima del croissant en píxeles
AREA_MIN = 8000
AREA_MAX = 90000

# Umbrales de brillo/color.
# Se ajustan según vuestra iluminación.
BRILLO_MIN_OK = 80
BRILLO_MAX_OK = 170

# Relación de forma aproximada
# 1.0 = casi circular/cuadrado
# valores mayores = más alargado
ASPECT_RATIO_MIN = 1.1
ASPECT_RATIO_MAX = 4.5

# ==========================
# FUNCIONES
# ==========================

def get_screen_resolution():
    try:
        root = tk.Tk()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return w, h
    except:
        return 1920, 1080  # Resolucion por defecto si falla Tkinter

SCREEN_W, SCREEN_H = get_screen_resolution()

def pad_frame_to_fullscreen(img, tgt_w, tgt_h):
    h, w = img.shape[:2]
    scale = min(tgt_w / w, tgt_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    
    if len(img.shape) == 3:
        canvas = np.zeros((tgt_h, tgt_w, 3), dtype=np.uint8)
    else:
        canvas = np.zeros((tgt_h, tgt_w), dtype=np.uint8)
        
    x_offset = (tgt_w - new_w) // 2
    y_offset = (tgt_h - new_h) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def clasificar_croissant(frame, contour):
    area = cv2.contourArea(contour)

    x, y, w, h = cv2.boundingRect(contour)
    
    # Usar minAreaRect hace que el aspect ratio no dependa de la rotación (horizontal o vertical)
    rect = cv2.minAreaRect(contour)
    rw, rh = rect[1]
    if min(rw, rh) > 0:
        aspect_ratio = max(rw, rh) / min(rw, rh)
    else:
        aspect_ratio = 0

    # Máscara solo del croissant
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    # Color medio dentro del croissant
    mean_bgr = cv2.mean(frame, mask=mask)
    b, g, r = mean_bgr[:3]

    # Brillo aproximado
    brillo = (r + g + b) / 3

    # Clasificación
    if area < AREA_MIN:
        estado = "SIN PRODUCTO / MUY PEQUENO"
        color_estado = (0, 0, 255)
    elif area > AREA_MAX:
        estado = "DEMASIADO GRANDE / DOS PIEZAS"
        color_estado = (0, 0, 255)
    elif aspect_ratio < ASPECT_RATIO_MIN or aspect_ratio > ASPECT_RATIO_MAX:
        estado = "DEFORMADO"
        color_estado = (0, 165, 255)
    elif brillo < BRILLO_MIN_OK:
        estado = "MUY OSCURO / QUEMADO"
        color_estado = (0, 0, 255)
    elif brillo > BRILLO_MAX_OK:
        estado = "MUY CLARO / CRUDO"
        color_estado = (0, 165, 255)
    else:
        estado = "OK"
        color_estado = (0, 255, 0)

    datos = {
        "area": area,
        "brillo": brillo,
        "aspect_ratio": aspect_ratio,
        "bbox": (x, y, w, h),
        "estado": estado,
        "color_estado": color_estado
    }

    return datos


# ==========================
# PROGRAMA PRINCIPAL
# ==========================

# Cargar la calibracion de la camara
try:
    calib_data = np.load('camera_calibration.npz')
    mtx = calib_data['mtx']
    dist = calib_data['dist']
    print("Calibración cargada correctamente.")
except Exception as e:
    print(f"Error cargando calibración: {e}")
    mtx, dist = None, None

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Error: no se pudo abrir la webcam.")
    exit()

print("Sistema iniciado.")
print("Pulsa 'q' para salir.")

# Configuracion de ventanas en pantalla completa manteniendo la proporcion
cv2.namedWindow("Control de calidad - Croissants", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.setWindowProperty("Control de calidad - Croissants", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.namedWindow("Mascara deteccion", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.setWindowProperty("Mascara deteccion", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: no se pudo leer la imagen.")
        break

    # Corregir la distorsion usando la calibracion (si se cargo)
    if mtx is not None and dist is not None:
        frame = cv2.undistort(frame, mtx, dist, None, mtx)

    # Redimensionar para trabajar más cómodo
    frame = cv2.resize(frame, (800, 600))

    # Convertir a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ==========================
    # SEGMENTACIÓN DEL CROISSANT
    # ==========================
    # Rango aproximado para colores marrón/dorado
    # Ampliado para detectar croissants muy oscuros/quemados
    lower_croissant = np.array([0, 15, 10])
    upper_croissant = np.array([45, 255, 255])

    mask = cv2.inRange(hsv, lower_croissant, upper_croissant)

    # Limpieza de ruido
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Buscar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    croissants_detectados = 0

    if contours:
        for contour in contours:
            if cv2.contourArea(contour) > AREA_MIN:
                croissants_detectados += 1
                datos = clasificar_croissant(frame, contour)

                x, y, w, h = datos["bbox"]
                estado = datos["estado"]
                color = datos["color_estado"]

                # Dibujar rectángulo y contorno
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.drawContours(frame, [contour], -1, color, 2)

                # Mostrar datos visuales al lado de cada croissant
                cv2.putText(frame, f"{estado}", (x, max(20, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if croissants_detectados == 0:
        cv2.putText(frame, "SIN CROISSANT", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    else:
        cv2.putText(frame, f"Detectados: {croissants_detectados}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Preparar imagenes para pantalla completa con franjas negras
    frame_padded = pad_frame_to_fullscreen(frame, SCREEN_W, SCREEN_H)
    mask_padded = pad_frame_to_fullscreen(mask, SCREEN_W, SCREEN_H)

    # Mostrar ventanas
    cv2.imshow("Control de calidad - Croissants", frame_padded)
    cv2.imshow("Mascara deteccion", mask_padded)

    # Salir con q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()