# Auto-extracted from openglsable.py without modifying the original file.
"""Primitivas de render OpenGL inmediato.

El proyecto usa OpenGL legacy (`glBegin`/`glEnd`) para mantener el prototipo
simple. Estas funciones dibujan cajas, diamantes, rejilla, corredor y numeros
del HUD sin depender de texturas externas.
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

def draw_box(center: tuple[float, float, float], size: tuple[float, float, float], color_rgba: tuple[float, float, float, float]):
	"""Dibuja una caja alineada a ejes en coordenadas de mundo/locales."""
	cx, cy, cz = center
	sx, sy, sz = size
	hx = sx * 0.5
	hy = sy * 0.5
	hz = sz * 0.5

	v = [
		(cx - hx, cy - hy, cz - hz),
		(cx + hx, cy - hy, cz - hz),
		(cx + hx, cy + hy, cz - hz),
		(cx - hx, cy + hy, cz - hz),
		(cx - hx, cy - hy, cz + hz),
		(cx + hx, cy - hy, cz + hz),
		(cx + hx, cy + hy, cz + hz),
		(cx - hx, cy + hy, cz + hz),
	]
	faces = [
		(0, 1, 2, 3),
		(4, 5, 6, 7),
		(0, 4, 7, 3),
		(1, 5, 6, 2),
		(3, 2, 6, 7),
		(0, 1, 5, 4),
	]

	glColor4f(*color_rgba)
	glBegin(GL_QUADS)
	for face in faces:
		for idx in face:
			glVertex3f(*v[idx])
	glEnd()


def draw_floor_grid():
	"""Rejilla de suelo para dar profundidad y referencia espacial."""
	glColor4f(0.14, 0.17, 0.21, 1.0)
	glBegin(GL_LINES)

	y = -0.62
	z_start = -0.6
	z_end = -10.0
	x_limit = 2.2

	for i in range(-11, 12):
		x = i * 0.2
		glVertex3f(x, y, z_start)
		glVertex3f(x, y, z_end)

	for i in range(0, 48):
		z = z_start - (i * 0.2)
		glVertex3f(-x_limit, y, z)
		glVertex3f(x_limit, y, z)

	glEnd()

def draw_rgb_corridor(time_t: float):
	"""Lineas animadas del pasillo, usadas como fondo del juego."""
	y_floor = -0.62
	y_ceil = 2.2
	z_start = -0.6
	z_end = -12.0
	x_limit = 2.2

	glBegin(GL_LINES)
	for i in range(0, 12):
		z = z_start - (i * 1.0)
		r = (np.sin(time_t * 2.0 + i * 0.5) + 1.0) * 0.5
		g = (np.sin(time_t * 2.7 + i * 0.7) + 1.0) * 0.5
		b = (np.sin(time_t * 3.1 + i * 0.9) + 1.0) * 0.5
		alpha = max(0.1, 1.0 - (abs(z) / 15.0))
		glColor4f(float(r), float(g), float(b), float(alpha))

		# Pared izq
		glVertex3f(-x_limit, y_floor, z)
		glVertex3f(-x_limit, y_ceil, z)

		# Techo
		glVertex3f(-x_limit, y_ceil, z)
		glVertex3f(x_limit, y_ceil, z)

		# Pared der
		glVertex3f(x_limit, y_ceil, z)
		glVertex3f(x_limit, y_floor, z)

	# Rieles longitudinales
	r_l = (np.sin(time_t * 1.5) + 1.0) * 0.5
	g_l = (np.sin(time_t * 1.8) + 1.0) * 0.5
	b_l = (np.sin(time_t * 0.9) + 1.0) * 0.5
	glColor4f(float(r_l), float(g_l), float(b_l), 0.6)

	glVertex3f(-x_limit, y_floor, z_start)
	glVertex3f(-x_limit, y_floor, z_end)
	glVertex3f(-x_limit, y_ceil, z_start)
	glVertex3f(-x_limit, y_ceil, z_end)
	glVertex3f(x_limit, y_ceil, z_start)
	glVertex3f(x_limit, y_ceil, z_end)
	glVertex3f(x_limit, y_floor, z_start)
	glVertex3f(x_limit, y_floor, z_end)
	
	glEnd()

def draw_oriented_box(position: np.ndarray, rotation: np.ndarray, size: np.ndarray, color_rgba: tuple[float, float, float, float]):
	"""Dibuja una caja orientada aplicando una matriz modelo temporal."""
	glPushMatrix()
	transform = np.eye(4, dtype=np.float32)
	transform[:3, :3] = rotation.astype(np.float32)
	transform[:3, 3] = position.astype(np.float32)
	glMultMatrixf(transform.T)
	draw_box((0.0, 0.0, 0.0), (float(size[0]), float(size[1]), float(size[2])), color_rgba)
	glPopMatrix()


def draw_oriented_diamond(position: np.ndarray, rotation: np.ndarray, size: np.ndarray, color_rgba: tuple[float, float, float, float]):
	"""Dibuja una bipiramide/diamante orientado, alternativa visual a la caja."""
	glPushMatrix()
	transform = np.eye(4, dtype=np.float32)
	transform[:3, :3] = rotation.astype(np.float32)
	transform[:3, 3] = position.astype(np.float32)
	glMultMatrixf(transform.T)
	hx = float(size[0]) * 0.5
	hy = float(size[1]) * 0.5
	hz = float(size[2]) * 0.5
	top = (0.0, hy, 0.0)
	bottom = (0.0, -hy, 0.0)
	left = (-hx, 0.0, 0.0)
	right = (hx, 0.0, 0.0)
	front = (0.0, 0.0, hz)
	back = (0.0, 0.0, -hz)
	faces = [
		(top, front, right), (top, right, back), (top, back, left), (top, left, front),
		(bottom, right, front), (bottom, back, right), (bottom, left, back), (bottom, front, left),
	]
	glColor4f(*color_rgba)
	glBegin(GL_TRIANGLES)
	for face in faces:
		for vertex in face:
			glVertex3f(*vertex)
	glEnd()
	glPopMatrix()


def draw_screen_rect(x: float, y: float, w: float, h: float, color_rgba: tuple[float, float, float, float], outline: bool = False):
	"""Rectangulo en espacio de pantalla para HUD 2D."""
	glColor4f(*color_rgba)
	if outline:
		glBegin(GL_LINE_LOOP)
		glVertex3f(x, y, 0.0)
		glVertex3f(x + w, y, 0.0)
		glVertex3f(x + w, y + h, 0.0)
		glVertex3f(x, y + h, 0.0)
		glEnd()
		return

	glBegin(GL_QUADS)
	glVertex3f(x, y, 0.0)
	glVertex3f(x + w, y, 0.0)
	glVertex3f(x + w, y + h, 0.0)
	glVertex3f(x, y + h, 0.0)
	glEnd()


SEVEN_SEGMENT_MAP = {
	# Segmentos estilo display digital: a arriba, d abajo, g centro, etc.
	0: ("a", "b", "c", "d", "e", "f"),
	1: ("b", "c"),
	2: ("a", "b", "g", "e", "d"),
	3: ("a", "b", "g", "c", "d"),
	4: ("f", "g", "b", "c"),
	5: ("a", "f", "g", "c", "d"),
	6: ("a", "f", "g", "e", "c", "d"),
	7: ("a", "b", "c"),
	8: ("a", "b", "c", "d", "e", "f", "g"),
	9: ("a", "b", "c", "d", "f", "g"),
}


def draw_digit_7seg(x: float, y: float, w: float, h: float, digit: int, color_rgba: tuple[float, float, float, float]):
	"""Dibuja un digito usando rectangulos de siete segmentos."""
	thickness = max(2.0, min(w, h) * 0.16)
	mid_y = y + (h * 0.5)
	segments = {
		"a": (x + thickness, y, w - (2.0 * thickness), thickness),
		"d": (x + thickness, y + h - thickness, w - (2.0 * thickness), thickness),
		"g": (x + thickness, mid_y - (thickness * 0.5), w - (2.0 * thickness), thickness),
		"f": (x, y + thickness, thickness, (h * 0.5) - thickness),
		"b": (x + w - thickness, y + thickness, thickness, (h * 0.5) - thickness),
		"e": (x, mid_y + (thickness * 0.5), thickness, (h * 0.5) - thickness),
		"c": (x + w - thickness, mid_y + (thickness * 0.5), thickness, (h * 0.5) - thickness),
	}

	for key in SEVEN_SEGMENT_MAP.get(int(digit), ()):
		draw_screen_rect(*segments[key], color_rgba)


def draw_number_7seg(x: float, y: float, digit_w: float, digit_h: float, value: int, color_rgba: tuple[float, float, float, float]):
	"""Dibuja un entero no negativo con varios digitos siete-segmentos."""
	text = str(max(0, int(value)))
	spacing = digit_w * 0.22
	cursor_x = x
	for ch in text:
		draw_digit_7seg(cursor_x, y, digit_w, digit_h, int(ch), color_rgba)
		cursor_x += digit_w + spacing
