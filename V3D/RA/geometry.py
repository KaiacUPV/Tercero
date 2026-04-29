# Auto-extracted from id.py without modifying the original file.
"""Geometria y utilidades de proyeccion para el sable RA.

Contiene la posicion 3D de las caras ArUco del cubo, conversion de pose de cara
a pose de cubo, primitivas 3D simples y helpers para el modo de bloques.
"""
import cv2
import cv2.aruco as aruco
import json
import numpy as np
import os
import queue
import random
import threading
import time
from pathlib import Path

try:
    import pygame
except ImportError:
    pygame = None

try:
    from openal import oalOpen, oalQuit
except ImportError:
    oalOpen = None
    oalQuit = None

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    from vosk import KaldiRecognizer, Model, SetLogLevel
except ImportError:
    KaldiRecognizer = None
    Model = None
    SetLogLevel = None

from .constants import *
from .pose import create_combined_rotation

CUBE_GEOMETRY = {
    # Cada id representa una cara del cubo y su rotacion relativa al frame base.
    17: {"relative_rotation": np.eye(3)},
    3: {"relative_rotation": create_combined_rotation([np.pi / 2, 0, 0], -np.pi / 2)},
    7: {"relative_rotation": create_combined_rotation([-np.pi / 2, 0, 0], 0)},
    15: {"relative_rotation": create_combined_rotation([0, np.pi / 2, 0], 0)},
    22: {"relative_rotation": create_combined_rotation([0, -np.pi / 2, 0], -np.pi)},
}

CUBE_MARKERS_3D = {}
base_corners = np.array(
    [
        [-HALF_SIZE, HALF_SIZE, 0],
        [HALF_SIZE, HALF_SIZE, 0],
        [HALF_SIZE, -HALF_SIZE, 0],
        [-HALF_SIZE, -HALF_SIZE, 0],
    ],
    dtype=np.float32,
)

for marker_id, geo in CUBE_GEOMETRY.items():
    # Precalcula esquinas 3D de cada marcador para poder estimar poses estables.
    R_rel = geo["relative_rotation"]
    t_adj = np.array([0, 0, -HALF_SIZE]) + (R_rel[:, 2] * HALF_SIZE)
    geo["relative_translation"] = t_adj.astype(np.float32)
    corners_3d = (R_rel @ base_corners.T).T + t_adj
    CUBE_MARKERS_3D[marker_id] = corners_3d.astype(np.float32)

_primary_base = CUBE_MARKERS_3D[PRIMARY_MARKER]
_primary_normal = CUBE_GEOMETRY[PRIMARY_MARKER]["relative_rotation"][:, 2].astype(np.float32)
_primary_top = _primary_base + (_primary_normal * PRISM_LENGTH)
PRISM_POINTS_3D = np.vstack((_primary_base, _primary_top)).astype(np.float32)
ARUCO_WHITE_BORDER_M = 0.005

def get_prism_points_with_offset(marker_id: int) -> np.ndarray:
    """
    Retorna las coordenadas 3D del prisma.
    Si se usa una cara lateral (no 17), desplaza el prisma:
    - 5mm hacia adentro del cubo (contra la normal de la cara lateral)
    - 5mm en el eje de la normal de la cara 17 (para respetar borde blanco en altura)
    Esto simula que siempre sale desde el centro de la cara 17, respetando bordes blancos.
    """
    prism_base = CUBE_MARKERS_3D[PRIMARY_MARKER].copy()
    prism_normal = CUBE_GEOMETRY[PRIMARY_MARKER]["relative_rotation"][:, 2].astype(np.float32)
    
    # Si estamos usando una cara lateral, desplazar el prisma
    if marker_id != PRIMARY_MARKER:
        # Normal hacia adentro desde la cara lateral usada
        face_geo = CUBE_GEOMETRY[marker_id]
        face_normal = face_geo["relative_rotation"][:, 2].astype(np.float32)
        # Desplazar el prisma hacia adentro: contra la normal de la cara lateral (borde radial)
        prism_base -= face_normal * ARUCO_WHITE_BORDER_M
        # Desplazar el prisma en el eje de la normal de la cara 17 (borde en altura)
        prism_base += prism_normal * ARUCO_WHITE_BORDER_M
    
    prism_top = prism_base + prism_normal * PRISM_LENGTH
    return np.vstack((prism_base, prism_top)).astype(np.float32)


def load_calibration(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Carga la calibración de cámara desde un .npz.

    Soporta claves antiguas (mtx/dist) y nuevas (camera_matrix/dist_coeffs).
    """
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


def get_control_anchor_tvec(cube_rvec, cube_tvec):
    """
    Retorna un punto de control invariante a la orientacion: el centro del cubo.
    El frame del cubo esta anclado en el centro de la cara 17; por eso el centro
    del cubo en ese frame es (0, 0, -HALF_SIZE).
    """
    R_cam_cube, _ = cv2.Rodrigues(cube_rvec)
    t_cam_cube = cube_tvec.reshape(3, 1).astype(np.float32)
    anchor_local = np.array([0.0, 0.0, -HALF_SIZE], dtype=np.float32).reshape(3, 1)
    t_cam_anchor = (R_cam_cube @ anchor_local) + t_cam_cube
    return t_cam_anchor.reshape(3).astype(np.float32)


def fit_frame_to_window(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Reescala maximo posible manteniendo aspect ratio y centra con bandas negras."""
    src_h, src_w = frame.shape[:2]
    if src_h <= 0 or src_w <= 0 or target_w <= 0 or target_h <= 0:
        return frame

    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    interpolation = cv2.INTER_LINEAR if scale >= 1.0 else cv2.INTER_AREA
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)

    canvas = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def build_cylinder_points(
    center_start: np.ndarray,
    center_end: np.ndarray,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
    radius: float,
    segments: int,
) -> np.ndarray:
    """Genera puntos 3D para dibujar un cilindro (dos circunferencias apiladas).

    Retorna (segments*2) puntos: primero la circunferencia base y luego la superior.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False, dtype=np.float32)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    start = center_start.astype(np.float32).reshape(3, 1)
    dir_x = axis_x.astype(np.float32).reshape(3, 1)
    dir_y = axis_y.astype(np.float32).reshape(3, 1)
    base_circle = start + dir_x * (cos_t * radius) + dir_y * (sin_t * radius)
    offset = (center_end.astype(np.float32) - center_start.astype(np.float32)).reshape(3, 1)
    top_circle = base_circle + offset
    return np.vstack((base_circle.T, top_circle.T)).astype(np.float32)


def build_frustum_points(
    center_start: np.ndarray,
    center_end: np.ndarray,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
    radius_start: float,
    radius_end: float,
    segments: int,
) -> np.ndarray:
    """Genera puntos 3D para un tronco de cono (frustum) entre dos centros."""
    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False, dtype=np.float32)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    start = center_start.astype(np.float32).reshape(3, 1)
    end = center_end.astype(np.float32).reshape(3, 1)
    dir_x = axis_x.astype(np.float32).reshape(3, 1)
    dir_y = axis_y.astype(np.float32).reshape(3, 1)
    base_circle = start + dir_x * (cos_t * radius_start) + dir_y * (sin_t * radius_start)
    top_circle = end + dir_x * (cos_t * radius_end) + dir_y * (sin_t * radius_end)
    return np.vstack((base_circle.T, top_circle.T)).astype(np.float32)


def build_box_points(
    center: np.ndarray,
    half_size: float,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
    axis_z: np.ndarray,
) -> np.ndarray:
    """Construye los 8 vértices de una caja orientada por ejes (OBB)."""
    center = center.astype(np.float32)
    axis_x = axis_x.astype(np.float32)
    axis_y = axis_y.astype(np.float32)
    axis_z = axis_z.astype(np.float32)

    vertices = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                offset = (axis_x * (sx * half_size)) + (axis_y * (sy * half_size)) + (axis_z * (sz * half_size))
                vertices.append(center + offset)

    return np.array(vertices, dtype=np.float32)


def point_to_segment_distance(point: np.ndarray, segment_start: np.ndarray, segment_end: np.ndarray) -> float:
    """Distancia mínima de un punto a un segmento 3D."""
    point = point.astype(np.float32)
    segment_start = segment_start.astype(np.float32)
    segment_end = segment_end.astype(np.float32)

    segment = segment_end - segment_start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-8:
        return float(np.linalg.norm(point - segment_start))

    t = float(np.dot(point - segment_start, segment) / denom)
    t = float(np.clip(t, 0.0, 1.0))
    closest = segment_start + (segment * t)
    return float(np.linalg.norm(point - closest))


def create_beat_block() -> dict:
    """Crea un bloque tipo Beat Saber (spawn aleatorio en carriles y color)."""
    lane_x, lane_y = random.choice(BEAT_BLOCK_LANES)
    color_key = random.choice(tuple(SABER_COLOR_PRESETS.keys()))
    return {
        "distance": float(random.uniform(BEAT_BLOCK_MIN_DISTANCE_M, BEAT_BLOCK_MAX_DISTANCE_M)),
        "lane_x": float(lane_x),
        "lane_y": float(lane_y),
        "color_key": color_key,
    }


def advance_beat_blocks(blocks: list[dict], dt: float) -> tuple[list[dict], int]:
    """Avanza los bloques hacia la cámara y cuenta fallos (cuando pasan de largo)."""
    moved_blocks = []
    misses = 0
    for block in blocks:
        updated_block = dict(block)
        updated_block["distance"] = float(updated_block["distance"] - (BEAT_BLOCK_SPEED_MPS * dt))
        if updated_block["distance"] < -BEAT_BLOCK_SIZE_M:
            misses += 1
            continue
        moved_blocks.append(updated_block)

    return moved_blocks, misses


def projected_points_are_safe(points_2d: np.ndarray, max_abs_px: float) -> bool:
    """Evita coordenadas proyectadas enormes o NaN/Inf que pueden romper el render."""
    pts = points_2d.reshape(-1, 2)
    if pts.size == 0:
        return False
    if not np.all(np.isfinite(pts)):
        return False
    return float(np.max(np.abs(pts))) <= max_abs_px
