# Auto-extracted from openglsable.py without modifying the original file.
"""CLI del juego 3D.

Parsea argumentos, prepara calibracion/camara/tracker y finalmente entrega el
control a OpenGLSaberApp.
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

from .app import OpenGLSaberApp
from .constants import *
from .math_utils import parse_resolution
from .tracking import (
	ArucoPoseTracker,
	build_cube_geometry,
	load_calibration,
	resolve_calibration_path,
)

def parse_args():
	"""Define las opciones de lanzamiento del juego."""
	parser = argparse.ArgumentParser(
		description="OpenGL saber prototype con tracking ArUco desde camara local"
	)
	parser.add_argument(
		"--camera",
		type=int,
		default=0,
		help="Indice de camara",
	)
	parser.add_argument(
		"--calibration",
		type=str,
		default="camera_calibration.npz",
		help="Archivo principal de calibracion",
	)
	parser.add_argument(
		"--fallback-calibration",
		type=str,
		default="calibracion_camara.npz",
		help="Archivo de calibracion alternativo",
	)
	parser.add_argument(
		"--capture-resolution",
		type=str,
		default="640x480",
		help="Resolucion de captura de camara WIDTHxHEIGHT",
	)
	parser.add_argument(
		"--capture-fps",
		type=float,
		default=60.0,
		help="FPS de captura solicitados",
	)
	parser.add_argument(
		"--cube-size-m",
		type=float,
		default=DEFAULT_CUBE_SIZE_M,
		help="Tamano del cubo ArUco en metros",
	)
	parser.add_argument(
		"--marker-size-m",
		type=float,
		default=DEFAULT_CUBE_SIZE_M,
		help="Tamano del marcador detectado en metros",
	)
	parser.add_argument(
		"--resolution",
		type=str,
		default=f"{DEFAULT_WIDTH}x{DEFAULT_HEIGHT}",
		help="Resolucion de ventana WIDTHxHEIGHT",
	)
	parser.add_argument(
		"--pose-scale",
		type=float,
		default=4.0,
		help="Escala de movimiento del tracking",
	)
	parser.add_argument(
		"--pos-smoothing",
		type=float,
		default=0.35,
		help="Suavizado de posicion en [0,1]",
	)
	parser.add_argument(
		"--lost-timeout",
		type=float,
		default=0.25,
		help="Tiempo maximo sin pose antes de marcar perdida",
	)
	parser.add_argument(
		"--fps",
		type=int,
		default=120,
		help="FPS maximos de render",
	)
	parser.add_argument(
		"--vsync",
		action="store_true",
		help="Activa vsync si la plataforma lo permite",
	)
	parser.add_argument(
		"--show-camera",
		action="store_true",
		help="Muestra una ventana OpenCV de depuracion con detecciones",
	)
	return parser.parse_args()


def main() -> int:
	"""Arranca tracking y render; devuelve 0 si la app cierra limpiamente."""
	args = parse_args()
	window_width, window_height = parse_resolution(args.resolution)
	capture_width, capture_height = parse_resolution(args.capture_resolution)

	calibration_path = resolve_calibration_path(args.calibration, args.fallback_calibration)
	camera_matrix, dist_coeffs = load_calibration(calibration_path)
	cube_geometry, cube_markers_3d = build_cube_geometry(args.cube_size_m)

	tracker = ArucoPoseTracker(
		camera_index=args.camera,
		camera_matrix=camera_matrix,
		dist_coeffs=dist_coeffs,
		cube_size_m=args.cube_size_m,
		marker_size_m=args.marker_size_m,
		cube_geometry=cube_geometry,
		cube_markers_3d=cube_markers_3d,
		width=capture_width,
		height=capture_height,
		capture_fps=args.capture_fps,
		show_camera=args.show_camera,
	)
	# El tracker corre en otro hilo: dejamos que capture algun frame antes de abrir
	# todo el runtime OpenGL, asi detectamos fallos de camara pronto.
	tracker.start()

	time.sleep(0.2)
	if tracker.error_message:
		print(tracker.error_message)
		tracker.stop()
		tracker.join(timeout=1.0)
		return 1

	app = OpenGLSaberApp(
		tracker=tracker,
		width=window_width,
		height=window_height,
		pose_scale=args.pose_scale,
		pos_smoothing=args.pos_smoothing,
		lost_timeout_s=args.lost_timeout,
		max_fps=args.fps,
		vsync=args.vsync,
	)

	try:
		app.run()
	finally:
		tracker.stop()
		tracker.join(timeout=1.0)

	return 0
