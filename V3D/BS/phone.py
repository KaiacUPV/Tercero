# Auto-extracted from openglsable.py without modifying the original file.
"""Entrada opcional de rotacion del movil por UDP.

Pensado para hyperimu u otra app que envie rotation vector/Euler al PC. El hilo
solo guarda el mensaje mas reciente para que la app no acumule latencia.
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

class PhoneRotationListener(threading.Thread):
	"""Recibe paquetes UDP del movil y expone el ultimo mensaje disponible."""

	def __init__(self, udp_ip: str, udp_port: int):
		super().__init__(daemon=True)
		self.udp_ip = str(udp_ip)
		self.udp_port = int(udp_port)
		self.error_message: str | None = None
		self._stop_event = threading.Event()
		self._sock: socket.socket | None = None
		self._queue: queue.Queue[str] = queue.Queue(maxsize=32)

	def start_listening(self) -> bool:
		"""Abre el socket UDP y arranca el hilo si el puerto esta libre."""
		try:
			sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
			sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
			sock.bind((self.udp_ip, self.udp_port))
			sock.settimeout(0.5)
			self._sock = sock
		except OSError as exc:
			self.error_message = (
				f"[Phone] No se pudo abrir UDP {self.udp_ip}:{self.udp_port} ({exc}). "
				"Cierra giro.py si lo tienes abierto o cambia el puerto."
			)
			self._sock = None
			return False

		super().start()
		return True

	def run(self):
		"""Lee datagramas UDP y mantiene una cola corta de mensajes recientes."""
		if self._sock is None:
			return
		while not self._stop_event.is_set():
			try:
				data, _addr = self._sock.recvfrom(2048)
			except socket.timeout:
				continue
			except OSError:
				break

			try:
				message = data.decode("utf-8", errors="ignore").strip()
			except Exception:
				continue

			if not message:
				continue

			try:
				self._queue.put_nowait(message)
			except queue.Full:
				try:
					_ = self._queue.get_nowait()
				except queue.Empty:
					pass
				try:
					self._queue.put_nowait(message)
				except queue.Full:
					pass

	def stop(self):
		"""Cierra el socket para desbloquear recvfrom y terminar el hilo."""
		self._stop_event.set()
		if self._sock is not None:
			try:
				self._sock.close()
			except OSError:
				pass
			self._sock = None

	def pop_latest_message(self) -> str | None:
		"""Devuelve solo el ultimo mensaje, descartando valores antiguos."""
		latest = None
		while True:
			try:
				latest = self._queue.get_nowait()
			except queue.Empty:
				break
		return latest
