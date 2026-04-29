# Auto-extracted from openglsable.py without modifying the original file.
"""Utilidades matematicas compartidas por tracking, gameplay y render."""
import argparse
import json
import queue
import random
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import cv2.aruco as aruco
import numpy as np
import pygame
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

from OpenGL.GL import (
	GL_BLEND,
	GL_COLOR_BUFFER_BIT,
	GL_DEPTH_BUFFER_BIT,
	GL_DEPTH_TEST,
	GL_LINE_LOOP,
	GL_LEQUAL,
	GL_LINES,
	GL_MODELVIEW,
	GL_ONE,
	GL_ONE_MINUS_SRC_ALPHA,
	GL_PROJECTION,
	GL_QUADS,
	GL_SRC_ALPHA,
	GL_TRIANGLES,
	glBegin,
	glBlendFunc,
	glClear,
	glClearColor,
	glColor4f,
	glDisable,
	glDepthFunc,
	glDepthMask,
	glEnable,
	glEnd,
	glLoadIdentity,
	glMatrixMode,
	glMultMatrixf,
	glOrtho,
	glPopMatrix,
	glPushMatrix,
	glVertex3f,
	glViewport,
)
from OpenGL.GLU import gluLookAt, gluPerspective
from pygame.locals import DOUBLEBUF, FULLSCREEN, KEYDOWN, K_c, K_ESCAPE, K_g, K_i, K_m, K_r, K_SPACE, K_v, K_z, K_b, OPENGL, QUIT, RESIZABLE, VIDEORESIZE

from .constants import *

def parse_resolution(resolution: str) -> tuple[int, int]:
	"""Parsea textos tipo '1280x720'."""
	pieces = resolution.lower().split("x")
	if len(pieces) != 2:
		raise ValueError("Resolucion invalida. Usa WIDTHxHEIGHT, ejemplo 1280x720")
	width = int(pieces[0])
	height = int(pieces[1])
	if width <= 0 or height <= 0:
		raise ValueError("La resolucion debe tener valores positivos")
	return width, height


def clamp(value: float, min_value: float, max_value: float) -> float:
	"""Limita un numero al rango [min_value, max_value]."""
	return float(max(min_value, min(max_value, value)))


def orthonormalize_rotation(rotation: np.ndarray) -> np.ndarray:
	"""Corrige una matriz de rotacion con SVD para quitar deriva numerica."""
	u, _, vt = np.linalg.svd(rotation.astype(np.float64))
	result = u @ vt
	if np.linalg.det(result) < 0.0:
		u[:, -1] *= -1.0
		result = u @ vt
	return result.astype(np.float32)


def rotation_matrix_from_euler_xyz_deg(x_deg: float, y_deg: float, z_deg: float) -> np.ndarray:
	"""Crea una matriz de rotacion desde Euler XYZ en grados."""
	x = np.radians(x_deg)
	y = np.radians(y_deg)
	z = np.radians(z_deg)

	cx, sx = np.cos(x), np.sin(x)
	cy, sy = np.cos(y), np.sin(y)
	cz, sz = np.cos(z), np.sin(z)

	rx = np.array(
		[[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
		dtype=np.float32,
	)
	ry = np.array(
		[[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
		dtype=np.float32,
	)
	rz = np.array(
		[[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
		dtype=np.float32,
	)
	return orthonormalize_rotation(rz @ ry @ rx)


def rotation_matrix_from_quaternion_xyzw(x: float, y: float, z: float, w: float) -> np.ndarray:
	"""Convierte quaternion XYZW (formato Android/hyperimu habitual) a matriz."""
	q = np.array([x, y, z, w], dtype=np.float64)
	n = float(np.linalg.norm(q))
	if n <= 1e-8:
		return np.eye(3, dtype=np.float32)
	q /= n
	x, y, z, w = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

	xx = x * x
	yy = y * y
	zz = z * z
	xy = x * y
	xz = x * z
	yz = y * z
	wx = w * x
	wy = w * y
	wz = w * z

	rot = np.array(
		[
			[1.0 - (2.0 * (yy + zz)), 2.0 * (xy - wz), 2.0 * (xz + wy)],
			[2.0 * (xy + wz), 1.0 - (2.0 * (xx + zz)), 2.0 * (yz - wx)],
			[2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - (2.0 * (xx + yy))],
		],
		dtype=np.float32,
	)
	return orthonormalize_rotation(rot)


def rotation_matrix_from_rodrigues(rvec: np.ndarray) -> np.ndarray:
	"""Convierte vector Rodrigues/OpenCV a matriz de rotacion limpia."""
	rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
	rot, _ = cv2.Rodrigues(rvec)
	return orthonormalize_rotation(rot.astype(np.float32))


def parse_csv_floats(message: str) -> list[float]:
	"""Extrae floats desde mensajes UDP separados por coma o punto y coma."""
	if not message:
		return []
	clean = message.strip().replace(";", ",")
	values: list[float] = []
	for part in clean.split(","):
		part = part.strip()
		if not part:
			continue
		try:
			values.append(float(part))
		except ValueError:
			continue
	return values

def normalize(vector: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
	"""Normaliza un vector y usa fallback si su longitud es casi cero."""
	length = float(np.linalg.norm(vector))
	if length <= 1e-6:
		if fallback is None:
			return np.zeros_like(vector, dtype=np.float32)
		return fallback.astype(np.float32)
	return (vector / length).astype(np.float32)


def distance_point_to_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
	"""Distancia minima entre un punto y un segmento 3D."""
	segment = end - start
	length_sq = float(np.dot(segment, segment))
	if length_sq <= 1e-8:
		return float(np.linalg.norm(point - start))
	t = clamp(float(np.dot(point - start, segment) / length_sq), 0.0, 1.0)
	closest = start + (segment * t)
	return float(np.linalg.norm(point - closest))


def lerp_vec(start: np.ndarray, end: np.ndarray, t: float) -> np.ndarray:
	"""Interpolacion lineal vectorial."""
	return (start + ((end - start) * float(t))).astype(np.float32)


def smoothstep01(value: float) -> float:
	"""Curva suave 0..1 para animaciones sin cortes bruscos."""
	t = clamp(value, 0.0, 1.0)
	return float(t * t * (3.0 - (2.0 * t)))


def min_distance_point_to_moving_segment(
	point_start: np.ndarray,
	point_end: np.ndarray,
	segment_start_a: np.ndarray,
	segment_start_b: np.ndarray,
	segment_end_a: np.ndarray,
	segment_end_b: np.ndarray,
	samples: int = SABER_COLLISION_SAMPLES,
) -> float:
	"""Aproxima la distancia minima entre un punto y un segmento en movimiento."""
	safe_samples = max(2, int(samples))
	best = float("inf")
	for i in range(safe_samples):
		t = i / float(safe_samples - 1)
		point = lerp_vec(point_start, point_end, t)
		seg_a = lerp_vec(segment_start_a, segment_end_a, t)
		seg_b = lerp_vec(segment_start_b, segment_end_b, t)
		best = min(best, distance_point_to_segment(point, seg_a, seg_b))
	return float(best)


def rotation_matrix_from_forward(forward: np.ndarray) -> np.ndarray:
	"""Construye una base ortonormal usando `forward` como eje principal."""
	forward_n = normalize(forward.astype(np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32))
	up_seed = np.array([0.0, 1.0, 0.0], dtype=np.float32)
	if abs(float(np.dot(forward_n, up_seed))) > 0.92:
		up_seed = np.array([1.0, 0.0, 0.0], dtype=np.float32)

	right = normalize(np.cross(up_seed, forward_n), np.array([1.0, 0.0, 0.0], dtype=np.float32))
	up = normalize(np.cross(forward_n, right), np.array([0.0, 1.0, 0.0], dtype=np.float32))

	rotation = np.eye(3, dtype=np.float32)
	rotation[:, 0] = right
	rotation[:, 1] = up
	rotation[:, 2] = forward_n
	return orthonormalize_rotation(rotation)


def rotation_matrix_from_blade_direction(blade_direction: np.ndarray) -> np.ndarray:
	"""Adapta una direccion de hoja al sistema de ejes del modelo del sable."""
	blade_dir_n = normalize(blade_direction.astype(np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32))
	return rotation_matrix_from_forward(-blade_dir_n)
