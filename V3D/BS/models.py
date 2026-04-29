# Auto-extracted from openglsable.py without modifying the original file.
"""Modelos de datos del gameplay 3D.

Son dataclasses ligeras para separar estado de juego (bloques, fragmentos,
proyectiles y sable enemigo) de la logica de actualizacion/render.
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
class BeatBlock:
	"""Bloque que avanza hacia el jugador y puede cortarse con la hoja."""
	position: np.ndarray
	velocity: np.ndarray
	shape_key: str
	size: np.ndarray
	color_rgba: tuple[float, float, float, float]
	rotation: np.ndarray = None
	was_cut: bool = False
	passed_player: bool = False

	def __post_init__(self):
		# La rotacion se corrige para evitar matrices deformadas por acumulacion.
		if self.rotation is None:
			self.rotation = np.eye(3, dtype=np.float32)
		else:
			self.rotation = orthonormalize_rotation(self.rotation)


@dataclass
class BlockFragment:
	"""Trozo generado al cortar un bloque; vive unos segundos y cae/rota."""
	position: np.ndarray
	velocity: np.ndarray
	size: np.ndarray
	shape_key: str
	rotation: np.ndarray
	angular_velocity: np.ndarray
	lifetime_s: float
	color_rgba: tuple[float, float, float, float]


@dataclass
class Projectile:
	"""Disparo que puede golpear al jugador o rebotar al tocar el sable."""
	position: np.ndarray
	velocity: np.ndarray
	radius: float
	color_rgba: tuple[float, float, float, float]
	deflected: bool = False
	lifetime_s: float = 3.0


@dataclass
class EnemyCombatSaber:
	"""Estado del sable enemigo en modo combate/parry."""
	guard_position: np.ndarray
	guard_direction: np.ndarray
	position: np.ndarray
	previous_position: np.ndarray
	blade_direction: np.ndarray
	previous_blade_direction: np.ndarray
	pose_from_position: np.ndarray
	pose_to_position: np.ndarray
	pose_from_direction: np.ndarray
	pose_to_direction: np.ndarray
	strike_target_position: np.ndarray
	strike_target_direction: np.ndarray
	phase: str = "idle"
	phase_elapsed_s: float = 0.0
	phase_duration_s: float = 0.0
	idle_wait_s: float = COMBAT_ATTACK_INITIAL_DELAY_S
	color_rgba: tuple[float, float, float, float] = (1.0, 0.18, 0.16, 0.98)

	def __post_init__(self):
		# Normalizamos tipos y direcciones para que el resto del gameplay pueda
		# asumir arrays float32 y vectores unitarios.
		for field_name in (
			"guard_position",
			"guard_direction",
			"position",
			"previous_position",
			"blade_direction",
			"previous_blade_direction",
			"pose_from_position",
			"pose_to_position",
			"pose_from_direction",
			"pose_to_direction",
			"strike_target_position",
			"strike_target_direction",
		):
			value = getattr(self, field_name)
			setattr(self, field_name, value.astype(np.float32))

		for field_name, fallback in (
			("guard_direction", np.array([0.0, -1.0, -0.2], dtype=np.float32)),
			("blade_direction", np.array([0.0, -1.0, -0.2], dtype=np.float32)),
			("previous_blade_direction", np.array([0.0, -1.0, -0.2], dtype=np.float32)),
			("pose_from_direction", np.array([0.0, -1.0, -0.2], dtype=np.float32)),
			("pose_to_direction", np.array([0.0, -1.0, -0.2], dtype=np.float32)),
			("strike_target_direction", np.array([0.0, -1.0, -0.2], dtype=np.float32)),
		):
			setattr(self, field_name, normalize(getattr(self, field_name), fallback))

		self.phase_duration_s = float(max(0.0, self.phase_duration_s))
		self.idle_wait_s = float(max(0.0, self.idle_wait_s))
