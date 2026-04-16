import cv2
import cv2.aruco as aruco
import numpy as np
from pathlib import Path

# ===== Constantes =====
PRIMARY_MARKER = 17
CUBE_SIZE_M = 0.065
MARKER_SIZE = CUBE_SIZE_M
HALF_SIZE = CUBE_SIZE_M / 2.0
DEAD_ZONE = 15.0
PRISM_LENGTH = 0.20
SHOW_FACE_INFO = False


def create_combined_rotation(rotation_axis, additional_z_rotation=0):
    R1 = cv2.Rodrigues(np.array(rotation_axis))[0]
    if additional_z_rotation != 0:
        R2 = cv2.Rodrigues(np.array([0, 0, additional_z_rotation]))[0]
        return R1 @ R2
    return R1


# Geometría del cubo (rotaciones relativas respecto a la cara 17)
CUBE_GEOMETRY = {
    17: {"relative_rotation": np.eye(3)},
    3:  {"relative_rotation": create_combined_rotation([np.pi/2, 0, 0], -np.pi/2)},
    7:  {"relative_rotation": create_combined_rotation([-np.pi/2, 0, 0], 0)},
    15: {"relative_rotation": create_combined_rotation([0, np.pi/2, 0], 0)},
    22: {"relative_rotation": create_combined_rotation([0, -np.pi/2, 0], -np.pi)},
}

# Precalcular las coordenadas 3D de todas las esquinas del cubo referenciadas al centro de la cara 17
CUBE_MARKERS_3D = {}
base_corners = np.array([
    [-HALF_SIZE,  HALF_SIZE, 0],
    [ HALF_SIZE,  HALF_SIZE, 0],
    [ HALF_SIZE, -HALF_SIZE, 0],
    [-HALF_SIZE, -HALF_SIZE, 0]
], dtype=np.float32)

for marker_id, geo in CUBE_GEOMETRY.items():
    R_rel = geo["relative_rotation"]
    # El vector offset del origen (cara 17) hacia el origen de la cara relativa en 3D
    t_adj = np.array([0, 0, -HALF_SIZE]) + R_rel[:, 2] * HALF_SIZE
    geo["relative_translation"] = t_adj.astype(np.float32)
    corners_3d = (R_rel @ base_corners.T).T + t_adj
    CUBE_MARKERS_3D[marker_id] = corners_3d.astype(np.float32)

# Prisma fijo anclado a la cara 17
_primary_base = CUBE_MARKERS_3D[PRIMARY_MARKER]
_primary_normal = CUBE_GEOMETRY[PRIMARY_MARKER]["relative_rotation"][:, 2].astype(np.float32)
_primary_top = _primary_base + _primary_normal * PRISM_LENGTH
PRISM_POINTS_3D = np.vstack((_primary_base, _primary_top)).astype(np.float32)


def load_calibration(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    mtx_key = "mtx" if "mtx" in data.files else "camera_matrix"
    dist_key = "dist" if "dist" in data.files else "dist_coeffs"
    return (data[mtx_key].astype(np.float32), 
            data[dist_key].astype(np.float32))

def select_reference_face(corners, ids):
    """Prioriza la cara 17; si no aparece, elige la cara con mayor área visible."""
    ids_flat = ids.flatten()
    if PRIMARY_MARKER in ids_flat:
        idx = int(np.where(ids_flat == PRIMARY_MARKER)[0][0])
        return PRIMARY_MARKER, idx

    best_marker_id = None
    best_idx = None
    max_area = 0.0
    for idx, (marker_corners, marker_id) in enumerate(zip(corners, ids_flat)):
        if marker_id not in CUBE_MARKERS_3D:
            continue
        area = cv2.contourArea(marker_corners[0].astype(np.float32))
        if area > max_area:
            max_area = area
            best_marker_id = marker_id
            best_idx = idx

    return best_marker_id, best_idx


def estimate_cube_pose_from_marker_pose(marker_id, marker_rvec, marker_tvec):
    """Calcula la pose del cubo usando únicamente la pose de una cara visible."""
    if marker_id not in CUBE_GEOMETRY:
        return None, None

    face_geo = CUBE_GEOMETRY[marker_id]
    R_cm = face_geo["relative_rotation"].astype(np.float32)
    t_cm = face_geo["relative_translation"].reshape(3, 1).astype(np.float32)

    R_cam_marker, _ = cv2.Rodrigues(marker_rvec)
    t_cam_marker = marker_tvec.reshape(3, 1).astype(np.float32)

    # marker -> cube: p_c = R_cm * p_m + t_cm
    # camera <- cube: R_cam_cube = R_cam_marker * R_cm^T
    R_cam_cube = R_cam_marker @ R_cm.T
    t_cam_cube = t_cam_marker - (R_cam_cube @ t_cm)

    cube_rvec, _ = cv2.Rodrigues(R_cam_cube)
    cube_tvec = t_cam_cube.astype(np.float32)
    return cube_rvec, cube_tvec


cap = cv2.VideoCapture(0)
# Buscamos el archivo subiendo un nivel y entrando en Proyecto_VR3 o en la misma carpeta
calibration_path = Path(__file__).parent.parent / "Proyecto_VR3" / "camera_calibration.npz"
if not calibration_path.exists():
    calibration_path = Path(__file__).with_name("camera_calibration.npz")
camera_matrix, dist_coeffs = load_calibration(calibration_path)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    marker_labels = [] if SHOW_FACE_INFO else None

    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners)
        
        # Pose de los marcadores (se usa para estimar la pose del cubo)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, camera_matrix, dist_coeffs)
        if SHOW_FACE_INFO:
            # Guardar etiquetas info básica
            for i, (marker_corners, marker_id) in enumerate(zip(corners, ids.flatten())):
                cx, cy = int(marker_corners[0][:, 0].mean()), int(marker_corners[0][:, 1].mean())
                x, y, z = tvecs[i][0]
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                marker_labels.append({
                    "cx": cx, "cy": cy,
                    "text": f"ID {marker_id} C({cx},{cy}) X:{x:+.3f} Y:{y:+.3f} Z:{z:+.3f}m"
                })

        # Elegir la cara de referencia: 17 si está visible, si no, la más visible
        best_marker_id, best_idx = select_reference_face(corners, ids)

        # Estimar pose del cubo usando SOLO esa cara
        prism_rvec, prism_tvec = None, None
        if best_marker_id is not None and best_idx is not None:
            prism_rvec, prism_tvec = estimate_cube_pose_from_marker_pose(
                best_marker_id,
                rvecs[best_idx][0],
                tvecs[best_idx][0]
            )

        if prism_rvec is not None and prism_tvec is not None:
            R, _ = cv2.Rodrigues(prism_rvec)
            z_normal = R[:, 2]
            angle_deg = np.degrees(np.arccos(np.clip(z_normal[2], -1.0, 1.0)))

            if DEAD_ZONE < angle_deg < (180.0 - DEAD_ZONE):
                # Prisma fijo sobre la cara 17; la pose se obtiene de la cara de referencia visible
                img_pts, _ = cv2.projectPoints(PRISM_POINTS_3D, prism_rvec, prism_tvec, camera_matrix, dist_coeffs)
                img_pts = img_pts.reshape(-1, 2).astype(np.int32)

                base_2d = img_pts[:4]
                top_2d = img_pts[4:]

                # Relleno suave para que se vea que ocupa la superficie completa de la cara 17
                overlay = frame.copy()
                cv2.fillConvexPoly(overlay, base_2d, (0, 120, 0))
                for i in range(4):
                    side = np.array([
                        base_2d[i],
                        base_2d[(i + 1) % 4],
                        top_2d[(i + 1) % 4],
                        top_2d[i]
                    ], dtype=np.int32)
                    cv2.fillConvexPoly(overlay, side, (0, 70, 0))
                cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

                # Aristas del prisma
                cv2.polylines(frame, [base_2d], True, (0, 255, 0), 2)
                cv2.polylines(frame, [top_2d], True, (0, 220, 255), 2)
                for i in range(4):
                    cv2.line(frame, tuple(base_2d[i]), tuple(top_2d[i]), (0, 255, 0), 2)

    # Mostrar frame
    mirrored_frame = cv2.flip(frame, 1)
    h, w = mirrored_frame.shape[:2]
    
    if SHOW_FACE_INFO:
        for label in marker_labels:
            mirrored_x = w - 1 - label["cx"]
            text_x = max(10, min(mirrored_x + 10, w - 550))
            text_y = max(20, min(label["cy"] - 10, h - 10))
            cv2.putText(mirrored_frame, label["text"], (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Detectar ArUco", mirrored_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
