import cv2
import numpy as np
import time


# Si conoces la distancia real entre el punto azul y el amarillo, ponla aqui
# para estimar Z con una camara normal. Ejemplo: 0.20 para 20 cm.
REAL_LINE_LENGTH_M = 0.20

# Valor aproximado para una webcam 640x480. Para mas precision hay que calibrar
# la camara con un tablero de ajedrez y obtener la matriz intrinseca real.
FOCAL_LENGTH_PX = 700

SCENE_WIDTH = 720
SCENE_HEIGHT = 520
SCENE_FOCAL = 520
SMOOTHING = 0.75


def get_color_center(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest) < 300:
        return None

    moments = cv2.moments(largest)
    if moments["m00"] == 0:
        return None

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    return cx, cy, largest


def pixel_to_approx_3d(x, y, z, frame_width, frame_height):
    cx = frame_width / 2
    cy = frame_height / 2

    x_3d = (x - cx) * z / FOCAL_LENGTH_PX
    y_3d = (y - cy) * z / FOCAL_LENGTH_PX

    return np.array([x_3d, y_3d, z])


def rotate_point(point, yaw, pitch):
    x, y, z = point

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    x, z = x * cos_yaw + z * sin_yaw, -x * sin_yaw + z * cos_yaw

    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    y, z = y * cos_pitch - z * sin_pitch, y * sin_pitch + z * cos_pitch

    return np.array([x, y, z])


def project_scene_point(point):
    # La escena mira desde una camara virtual, no desde la webcam.
    rotated = rotate_point(point, yaw=np.deg2rad(-35), pitch=np.deg2rad(22))
    camera_z = rotated[2] + 2.0

    if camera_z <= 0.05:
        return None

    sx = int(SCENE_WIDTH / 2 + SCENE_FOCAL * rotated[0] / camera_z)
    sy = int(SCENE_HEIGHT / 2 - SCENE_FOCAL * rotated[1] / camera_z)

    return sx, sy


def draw_3d_line(scene, p1, p2, color, thickness=2):
    start = project_scene_point(p1)
    end = project_scene_point(p2)

    if start is not None and end is not None:
        cv2.line(scene, start, end, color, thickness, cv2.LINE_AA)


def draw_3d_point(scene, point, color, label):
    projected = project_scene_point(point)

    if projected is None:
        return

    cv2.circle(scene, projected, 7, color, -1, cv2.LINE_AA)
    cv2.putText(scene, label, (projected[0] + 9, projected[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def render_3d_scene(blue_3d=None, yellow_3d=None):
    scene = np.full((SCENE_HEIGHT, SCENE_WIDTH, 3), (22, 24, 28), dtype=np.uint8)

    grid_color = (60, 65, 72)
    axis_x_color = (70, 90, 255)
    axis_y_color = (90, 210, 90)
    axis_z_color = (230, 230, 230)

    for value in np.linspace(-0.6, 0.6, 7):
        draw_3d_line(scene, np.array([-0.6, value, 0.0]), np.array([0.6, value, 0.0]), grid_color, 1)
        draw_3d_line(scene, np.array([value, -0.6, 0.0]), np.array([value, 0.6, 0.0]), grid_color, 1)

    draw_3d_line(scene, np.array([-0.7, 0.0, 0.0]), np.array([0.7, 0.0, 0.0]), axis_x_color, 2)
    draw_3d_line(scene, np.array([0.0, -0.7, 0.0]), np.array([0.0, 0.7, 0.0]), axis_y_color, 2)
    draw_3d_line(scene, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), axis_z_color, 2)

    cv2.putText(scene, "X", project_scene_point(np.array([0.75, 0.0, 0.0])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, axis_x_color, 2)
    cv2.putText(scene, "Y", project_scene_point(np.array([0.0, 0.75, 0.0])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, axis_y_color, 2)
    cv2.putText(scene, "Z", project_scene_point(np.array([0.0, 0.0, 1.05])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, axis_z_color, 2)

    if blue_3d is not None and yellow_3d is not None:
        draw_3d_line(scene, blue_3d, yellow_3d, (0, 0, 255), 4)
        draw_3d_point(scene, blue_3d, (255, 0, 0), "Azul")
        draw_3d_point(scene, yellow_3d, (0, 255, 255), "Amarillo")

        length = np.linalg.norm(yellow_3d - blue_3d)
        cv2.putText(scene, f"Recta 3D aprox: {length:.3f} m", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2)
        cv2.putText(scene, f"Z: {blue_3d[2]:.2f} m", (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 2)
    else:
        cv2.putText(scene, "Buscando puntos azul y amarillo...", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2)

    return scene


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la camara del portatil")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

smoothed_blue_3d = None
smoothed_yellow_3d = None
last_print_time = 0

while True:
    ok, frame = cap.read()

    if not ok:
        break

    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rangos HSV aproximados. Ajustalos si tu iluminacion cambia.
    lower_blue = np.array([90, 80, 50])
    upper_blue = np.array([130, 255, 255])

    lower_yellow = np.array([20, 80, 80])
    upper_yellow = np.array([35, 255, 255])

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel = np.ones((5, 5), np.uint8)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)

    blue_data = get_color_center(mask_blue)
    yellow_data = get_color_center(mask_yellow)

    if blue_data is not None:
        bx, by, blue_contour = blue_data
        cv2.drawContours(frame, [blue_contour], -1, (255, 0, 0), 2)
        cv2.circle(frame, (bx, by), 6, (255, 0, 0), -1)
        cv2.putText(frame, "Azul", (bx + 8, by - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if yellow_data is not None:
        yx, yy, yellow_contour = yellow_data
        cv2.drawContours(frame, [yellow_contour], -1, (0, 255, 255), 2)
        cv2.circle(frame, (yx, yy), 6, (0, 255, 255), -1)
        cv2.putText(frame, "Amarillo", (yx + 8, yy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if blue_data is not None and yellow_data is not None:
        cv2.line(frame, (bx, by), (yx, yy), (0, 0, 255), 3)

        line_length_px = np.linalg.norm(np.array([yx - bx, yy - by]))
        if line_length_px < 1:
            continue

        z_estimated = FOCAL_LENGTH_PX * REAL_LINE_LENGTH_M / line_length_px

        blue_3d = pixel_to_approx_3d(bx, by, z_estimated, width, height)
        yellow_3d = pixel_to_approx_3d(yx, yy, z_estimated, width, height)

        if smoothed_blue_3d is None or smoothed_yellow_3d is None:
            smoothed_blue_3d = blue_3d
            smoothed_yellow_3d = yellow_3d
        else:
            smoothed_blue_3d = SMOOTHING * smoothed_blue_3d + (1 - SMOOTHING) * blue_3d
            smoothed_yellow_3d = SMOOTHING * smoothed_yellow_3d + (1 - SMOOTHING) * yellow_3d

        direction = yellow_3d - blue_3d

        cv2.putText(
            frame,
            f"Longitud 2D: {line_length_px:.1f} px",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Z estimada: {z_estimated:.2f} m",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

        now = time.time()
        if now - last_print_time > 0.5:
            print("Azul 2D:", (bx, by))
            print("Amarillo 2D:", (yx, yy))
            print("Azul 3D aprox:", blue_3d)
            print("Amarillo 3D aprox:", yellow_3d)
            print("Vector direccion aprox:", direction)
            print()
            last_print_time = now

    cv2.imshow("Deteccion azul-amarillo webcam", frame)
    cv2.imshow("Entorno 3D recta", render_3d_scene(smoothed_blue_3d, smoothed_yellow_3d))

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
