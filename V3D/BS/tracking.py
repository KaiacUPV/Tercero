# Auto-extracted from openglsable.py without modifying the original file.
"""Tracking ArUco para el juego 3D.

Se ejecuta en un hilo aparte para que la captura de camara no bloquee el render
OpenGL. Convierte marcadores visibles del cubo en una pose estable del centro
del cubo, que luego BS.app usa como posicion/control del sable.
"""
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
from .math_utils import *

@dataclass
class PoseSample:
	"""Instantanea thread-safe de la ultima pose estimada."""
	rotation_cv: np.ndarray
	translation_cv: np.ndarray
	timestamp_s: float


def create_combined_rotation(rotation_axis, additional_z_rotation=0.0):
	"""Compone una rotacion Rodrigues con una rotacion extra sobre Z."""
	r1 = cv2.Rodrigues(np.array(rotation_axis, dtype=np.float32))[0]
	if additional_z_rotation != 0.0:
		r2 = cv2.Rodrigues(np.array([0.0, 0.0, additional_z_rotation], dtype=np.float32))[0]
		return r1 @ r2
	return r1


def build_cube_geometry(cube_size_m: float):
	"""Construye la geometria 3D de las caras ArUco pegadas al cubo."""
	half_size = cube_size_m / 2.0

	cube_geometry = {
		17: {"relative_rotation": np.eye(3, dtype=np.float32)},
		3: {"relative_rotation": create_combined_rotation([np.pi / 2.0, 0.0, 0.0], -np.pi / 2.0)},
		7: {"relative_rotation": create_combined_rotation([-np.pi / 2.0, 0.0, 0.0], 0.0)},
		15: {"relative_rotation": create_combined_rotation([0.0, np.pi / 2.0, 0.0], 0.0)},
		22: {"relative_rotation": create_combined_rotation([0.0, -np.pi / 2.0, 0.0], -np.pi)},
	}

	cube_markers_3d = {}
	base_corners = np.array(
		[
			[-half_size, half_size, 0.0],
			[half_size, half_size, 0.0],
			[half_size, -half_size, 0.0],
			[-half_size, -half_size, 0.0],
		],
		dtype=np.float32,
	)

	for marker_id, geo in cube_geometry.items():
		r_rel = geo["relative_rotation"].astype(np.float32)
		t_adj = np.array([0.0, 0.0, -half_size], dtype=np.float32) + r_rel[:, 2] * half_size
		geo["relative_translation"] = t_adj.astype(np.float32)
		corners_3d = (r_rel @ base_corners.T).T + t_adj
		cube_markers_3d[marker_id] = corners_3d.astype(np.float32)

	return cube_geometry, cube_markers_3d


def load_calibration(npz_path: Path):
	"""Carga matriz de camara y distorsion desde formato nuevo o antiguo."""
	data = np.load(npz_path)
	mtx_key = "mtx" if "mtx" in data.files else "camera_matrix"
	dist_key = "dist" if "dist" in data.files else "dist_coeffs"
	return data[mtx_key].astype(np.float32), data[dist_key].astype(np.float32)


def select_reference_face(corners, ids, cube_markers_3d):
	"""Elige cara 17 si aparece; si no, la cara valida con mas area visible."""
	ids_flat = ids.flatten()
	if PRIMARY_MARKER in ids_flat:
		idx = int(np.where(ids_flat == PRIMARY_MARKER)[0][0])
		return PRIMARY_MARKER, idx

	best_marker_id = None
	best_idx = None
	max_area = 0.0

	for idx, (marker_corners, marker_id) in enumerate(zip(corners, ids_flat)):
		if marker_id not in cube_markers_3d:
			continue
		area = cv2.contourArea(marker_corners[0].astype(np.float32))
		if area > max_area:
			max_area = area
			best_marker_id = int(marker_id)
			best_idx = idx

	return best_marker_id, best_idx


def estimate_cube_pose_from_marker_pose(marker_id, marker_rvec, marker_tvec, cube_geometry):
	"""Convierte la pose de una cara ArUco en pose del cubo completo."""
	if marker_id not in cube_geometry:
		return None, None

	face_geo = cube_geometry[marker_id]
	r_cm = face_geo["relative_rotation"].astype(np.float32)
	t_cm = face_geo["relative_translation"].reshape(3, 1).astype(np.float32)

	r_cam_marker, _ = cv2.Rodrigues(marker_rvec)
	t_cam_marker = marker_tvec.reshape(3, 1).astype(np.float32)

	r_cam_cube = r_cam_marker @ r_cm.T
	t_cam_cube = t_cam_marker - (r_cam_cube @ t_cm)

	cube_rvec, _ = cv2.Rodrigues(r_cam_cube)
	return cube_rvec, t_cam_cube.astype(np.float32)


def estimate_cube_pose_from_visible_markers(
	corners,
	ids,
	marker_size_m: float,
	cube_geometry,
	cube_markers_3d,
	camera_matrix: np.ndarray,
	dist_coeffs: np.ndarray,
	previous_rvec: Optional[np.ndarray] = None,
	previous_tvec: Optional[np.ndarray] = None,
):
	"""Estima la pose del cubo usando una o varias caras visibles.

	Con varias caras usa solvePnP sobre todas las esquinas, que suele ser mas
	estable. Con una sola cara cae al metodo directo por marcador.
	"""
	ids_flat = ids.flatten()
	valid_entries = []

	for marker_corners, marker_id in zip(corners, ids_flat):
		if int(marker_id) not in cube_markers_3d:
			continue
		valid_entries.append((marker_corners[0].astype(np.float32), int(marker_id)))

	if not valid_entries:
		return None, None

	if len(valid_entries) == 1:
		marker_corners, marker_id = valid_entries[0]
		rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
			[np.asarray(marker_corners, dtype=np.float32)],
			float(marker_size_m),
			camera_matrix,
			dist_coeffs,
		)
		return estimate_cube_pose_from_marker_pose(
			marker_id,
			rvecs[0][0],
			tvecs[0][0],
			cube_geometry,
		)

	object_points = np.concatenate(
		[cube_markers_3d[marker_id] for _, marker_id in valid_entries],
		axis=0,
	).astype(np.float32)
	image_points = np.concatenate(
		[marker_corners for marker_corners, _ in valid_entries],
		axis=0,
	).astype(np.float32)

	use_guess = previous_rvec is not None and previous_tvec is not None
	initial_rvec = None if previous_rvec is None else previous_rvec.reshape(3, 1).astype(np.float32)
	initial_tvec = None if previous_tvec is None else previous_tvec.reshape(3, 1).astype(np.float32)

	ok, cube_rvec, cube_tvec = cv2.solvePnP(
		object_points,
		image_points,
		camera_matrix,
		dist_coeffs,
		rvec=initial_rvec,
		tvec=initial_tvec,
		useExtrinsicGuess=use_guess,
		flags=cv2.SOLVEPNP_ITERATIVE,
	)
	if not ok:
		return None, None

	return cube_rvec.astype(np.float32), cube_tvec.astype(np.float32)


def resolve_calibration_path(primary: str, fallback: str) -> Path:
	"""Busca la calibracion en rutas habituales del proyecto."""
	script_dir = Path(__file__).resolve().parent
	project_root = script_dir.parent
	calibration_dir = project_root / "calibracion"
	calibration_dir_alt = project_root / "calibración"
	candidates = [
		Path(primary),
		script_dir / primary,
		project_root / primary,
		calibration_dir / primary,
		calibration_dir_alt / primary,
		Path(fallback),
		script_dir / fallback,
		project_root / fallback,
		calibration_dir / fallback,
		calibration_dir_alt / fallback,
	]

	for candidate in candidates:
		if candidate.exists():
			return candidate

	searched = "\n".join(f"- {path}" for path in candidates)
	raise FileNotFoundError(
		"No se encontro archivo de calibracion de camara. Revisado:\n" + searched
	)

class ArucoPoseTracker(threading.Thread):
	"""Hilo de captura/deteccion ArUco con lectura segura desde el render."""

	def __init__(
		self,
		camera_index: int,
		camera_matrix: np.ndarray,
		dist_coeffs: np.ndarray,
		cube_size_m: float,
		marker_size_m: float,
		cube_geometry,
		cube_markers_3d,
		width: int,
		height: int,
		capture_fps: float,
		show_camera: bool,
	):
		super().__init__(daemon=True)
		self.camera_index = int(camera_index)
		self.camera_matrix = camera_matrix
		self.dist_coeffs = dist_coeffs
		self.cube_size_m = float(cube_size_m)
		self.marker_size_m = float(marker_size_m)
		self.cube_geometry = cube_geometry
		self.cube_markers_3d = cube_markers_3d
		self.width = int(width)
		self.height = int(height)
		self.capture_fps = float(capture_fps)
		self.show_camera = bool(show_camera)

		self.error_message: Optional[str] = None

		aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
		self.detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())

		self._running = threading.Event()
		self._running.set()
		self._lock = threading.Lock()
		self._latest_pose: Optional[PoseSample] = None
		self._last_cube_rvec: Optional[np.ndarray] = None
		self._last_cube_tvec: Optional[np.ndarray] = None

	def stop(self):
		"""Solicita terminar el bucle de captura."""
		self._running.clear()

	def get_latest_pose(self) -> Optional[PoseSample]:
		"""Devuelve una copia de la ultima pose para evitar carreras entre hilos."""
		with self._lock:
			if self._latest_pose is None:
				return None
			return PoseSample(
				rotation_cv=self._latest_pose.rotation_cv.copy(),
				translation_cv=self._latest_pose.translation_cv.copy(),
				timestamp_s=self._latest_pose.timestamp_s,
			)

	def run(self):
		"""Bucle de camara: captura frame, detecta marcadores y publica pose."""
		cap = cv2.VideoCapture(self.camera_index)
		if not cap.isOpened():
			self.error_message = f"No se pudo abrir la camara indice {self.camera_index}"
			return

		cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
		cap.set(cv2.CAP_PROP_FPS, float(self.capture_fps))
		cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

		while self._running.is_set():
			ok, frame = cap.read()
			if not ok:
				continue

			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			corners, ids, _ = self.detector.detectMarkers(gray)

			cube_rvec = None
			cube_tvec = None

			if ids is not None and len(ids) > 0:
				rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
					corners,
					self.marker_size_m,
					self.camera_matrix,
					self.dist_coeffs,
				)

				cube_rvec, cube_tvec = estimate_cube_pose_from_visible_markers(
					corners,
					ids,
					self.marker_size_m,
					self.cube_geometry,
					self.cube_markers_3d,
					self.camera_matrix,
					self.dist_coeffs,
					previous_rvec=self._last_cube_rvec,
					previous_tvec=self._last_cube_tvec,
				)

				if cube_rvec is None or cube_tvec is None:
					marker_id, marker_idx = select_reference_face(
						corners,
						ids,
						self.cube_markers_3d,
					)

					if marker_id is not None and marker_idx is not None:
						cube_rvec, cube_tvec = estimate_cube_pose_from_marker_pose(
							marker_id,
							rvecs[marker_idx][0],
							tvecs[marker_idx][0],
							self.cube_geometry,
						)

				if cube_rvec is not None and cube_tvec is not None:
					self._last_cube_rvec = cube_rvec.reshape(3, 1).astype(np.float32)
					self._last_cube_tvec = cube_tvec.reshape(3, 1).astype(np.float32)

					r_cam_cube, _ = cv2.Rodrigues(cube_rvec)
					t_cam_cube = cube_tvec.reshape(3, 1).astype(np.float32)
					anchor_local = np.array(
						[0.0, 0.0, -0.5 * self.cube_size_m],
						dtype=np.float32,
					).reshape(3, 1)
					t_cam_anchor = (r_cam_cube @ anchor_local) + t_cam_cube

					sample = PoseSample(
						rotation_cv=r_cam_cube.astype(np.float32),
						translation_cv=t_cam_anchor.reshape(3).astype(np.float32),
						timestamp_s=time.perf_counter(),
					)
					with self._lock:
						self._latest_pose = sample

			if self.show_camera:
				display = frame.copy()
				if ids is not None and len(ids) > 0:
					aruco.drawDetectedMarkers(display, corners)
				if cube_rvec is not None and cube_tvec is not None:
					cv2.drawFrameAxes(
						display,
						self.camera_matrix,
						self.dist_coeffs,
						cube_rvec,
						cube_tvec,
						0.06,
					)

				cv2.imshow("Aruco tracker OpenGL", display)
				if (cv2.waitKey(1) & 0xFF) == 27:
					self.stop()
					break

		cap.release()
		if self.show_camera:
			cv2.destroyWindow("Aruco tracker OpenGL")
