# Auto-extracted from openglsable.py without modifying the original file.
"""Audio del juego 3D.

Centraliza los efectos del sable y del gameplay. Igual que en RA/audio.py,
intenta OpenAL primero y pygame.mixer despues, dejando el juego operativo aunque
no haya backend de audio disponible.
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

AUDIO_BACKEND = "none"
# Fuentes OpenAL y sonidos pygame precargados. Se guardan globalmente porque el
# juego dispara efectos con mucha frecuencia durante cortes, impactos y parrys.
OPENAL_ON_SOURCE = None
OPENAL_LOOP_SOURCE = None
OPENAL_OFF_SOURCE = None
OPENAL_PROJECTILE_REFLECT_SOURCE = None
OPENAL_LIFE_LOST_SOURCE = None
OPENAL_CUT_SOURCE = None
OPENAL_PARRY_SOURCE = None
PYGAME_ON_SOUND = None
PYGAME_LOOP_SOUND = None
PYGAME_OFF_SOUND = None
PYGAME_PROJECTILE_REFLECT_SOUND = None
PYGAME_LIFE_LOST_SOUND = None
PYGAME_CUT_SOUND = None
PYGAME_PARRY_SOUND = None
PYGAME_ON_CHANNEL = None
PYGAME_LOOP_CHANNEL = None
PYGAME_EFFECT_CHANNEL = None

def init_audio_backend(
	saber_on_path: Path,
	saber_loop_path: Path,
	saber_off_path: Path,
	projectile_reflect_path: Path,
	life_lost_path: Path,
	cut_path: Path,
	parry_path: Path,
) -> str:
	"""Inicializa y precarga todos los sonidos usados por el juego."""
	global AUDIO_BACKEND
	global OPENAL_ON_SOURCE, OPENAL_LOOP_SOURCE, OPENAL_OFF_SOURCE, OPENAL_PROJECTILE_REFLECT_SOURCE, OPENAL_LIFE_LOST_SOURCE, OPENAL_CUT_SOURCE, OPENAL_PARRY_SOURCE
	global PYGAME_ON_SOUND, PYGAME_LOOP_SOUND, PYGAME_OFF_SOUND, PYGAME_PROJECTILE_REFLECT_SOUND, PYGAME_LIFE_LOST_SOUND, PYGAME_CUT_SOUND, PYGAME_PARRY_SOUND
	global PYGAME_ON_CHANNEL, PYGAME_LOOP_CHANNEL, PYGAME_EFFECT_CHANNEL

	AUDIO_BACKEND = "none"
	OPENAL_ON_SOURCE = None
	OPENAL_LOOP_SOURCE = None
	OPENAL_OFF_SOURCE = None
	OPENAL_PROJECTILE_REFLECT_SOURCE = None
	OPENAL_LIFE_LOST_SOURCE = None
	OPENAL_CUT_SOURCE = None
	OPENAL_PARRY_SOURCE = None
	PYGAME_ON_SOUND = None
	PYGAME_LOOP_SOUND = None
	PYGAME_OFF_SOUND = None
	PYGAME_PROJECTILE_REFLECT_SOUND = None
	PYGAME_LIFE_LOST_SOUND = None
	PYGAME_CUT_SOUND = None
	PYGAME_PARRY_SOUND = None
	PYGAME_ON_CHANNEL = None
	PYGAME_LOOP_CHANNEL = None
	PYGAME_EFFECT_CHANNEL = None

	# Preferencia 1: OpenAL, por sus fuentes persistentes y control de ganancia.
	if oalOpen is not None:
		try:
			if saber_on_path.exists():
				OPENAL_ON_SOURCE = oalOpen(str(saber_on_path))
			if saber_loop_path.exists():
				OPENAL_LOOP_SOURCE = oalOpen(str(saber_loop_path))
				OPENAL_LOOP_SOURCE.set_looping(True)
			if saber_off_path.exists():
				OPENAL_OFF_SOURCE = oalOpen(str(saber_off_path))
			if projectile_reflect_path.exists():
				OPENAL_PROJECTILE_REFLECT_SOURCE = oalOpen(str(projectile_reflect_path))
			if life_lost_path.exists():
				OPENAL_LIFE_LOST_SOURCE = oalOpen(str(life_lost_path))
			if cut_path.exists():
				OPENAL_CUT_SOURCE = oalOpen(str(cut_path))
			if parry_path.exists():
				OPENAL_PARRY_SOURCE = oalOpen(str(parry_path))
				set_audio_source_gain(OPENAL_PARRY_SOURCE, PARRY_SOUND_VOLUME)

			AUDIO_BACKEND = "openal"
			return AUDIO_BACKEND
		except Exception:
			# Si algun wav/backend falla, se descarta OpenAL entero y se prueba pygame.
			OPENAL_ON_SOURCE = None
			OPENAL_LOOP_SOURCE = None
			OPENAL_OFF_SOURCE = None
			OPENAL_PROJECTILE_REFLECT_SOURCE = None
			OPENAL_LIFE_LOST_SOURCE = None
			OPENAL_CUT_SOURCE = None
			OPENAL_PARRY_SOURCE = None
			if oalQuit is not None:
				try:
					oalQuit()
				except Exception:
					pass

	# Preferencia 2: pygame.mixer, mas habitual en entornos Python sencillos.
	try:
		if not pygame.mixer.get_init():
			pygame.mixer.init()

		if saber_on_path.exists():
			PYGAME_ON_SOUND = pygame.mixer.Sound(str(saber_on_path))
		if saber_loop_path.exists():
			PYGAME_LOOP_SOUND = pygame.mixer.Sound(str(saber_loop_path))
		if saber_off_path.exists():
			PYGAME_OFF_SOUND = pygame.mixer.Sound(str(saber_off_path))
		if projectile_reflect_path.exists():
			PYGAME_PROJECTILE_REFLECT_SOUND = pygame.mixer.Sound(str(projectile_reflect_path))
		if life_lost_path.exists():
			PYGAME_LIFE_LOST_SOUND = pygame.mixer.Sound(str(life_lost_path))
		if cut_path.exists():
			PYGAME_CUT_SOUND = pygame.mixer.Sound(str(cut_path))
		if parry_path.exists():
			PYGAME_PARRY_SOUND = pygame.mixer.Sound(str(parry_path))
			PYGAME_PARRY_SOUND.set_volume(float(max(0.0, min(1.0, PARRY_SOUND_VOLUME))))

		PYGAME_ON_CHANNEL = pygame.mixer.Channel(0)
		PYGAME_LOOP_CHANNEL = pygame.mixer.Channel(1)
		PYGAME_EFFECT_CHANNEL = pygame.mixer.Channel(2)
		AUDIO_BACKEND = "pygame"
		return AUDIO_BACKEND
	except Exception:
		PYGAME_ON_SOUND = None
		PYGAME_LOOP_SOUND = None
		PYGAME_OFF_SOUND = None
		PYGAME_PROJECTILE_REFLECT_SOUND = None
		PYGAME_LIFE_LOST_SOUND = None
		PYGAME_CUT_SOUND = None
		PYGAME_PARRY_SOUND = None
		PYGAME_ON_CHANNEL = None
		PYGAME_LOOP_CHANNEL = None
		PYGAME_EFFECT_CHANNEL = None

	return AUDIO_BACKEND


def play_openal_source(source) -> bool:
	"""Reproduce una fuente OpenAL desde el inicio, sin solaparla consigo misma."""
	if source is None:
		return False
	try:
		source.stop()
	except Exception:
		pass
	try:
		source.rewind()
	except Exception:
		pass
	try:
		source.play()
		return True
	except Exception:
		return False


def _play_backend_sound(openal_source, pygame_sound, pygame_channel, *, loops: int = 0) -> bool:
	"""Abstrae la diferencia entre OpenAL y pygame para lanzar un sonido."""
	if AUDIO_BACKEND == "openal":
		return play_openal_source(openal_source)
	if AUDIO_BACKEND == "pygame":
		if pygame_sound is None or pygame_channel is None:
			return False
		try:
			pygame_channel.play(pygame_sound, loops=int(loops))
			return True
		except pygame.error:
			return False
	return False


def set_audio_source_gain(source, gain: float) -> None:
	"""Ajusta volumen en wrappers OpenAL que exponen nombres de API distintos."""
	if source is None:
		return
	safe_gain = float(max(0.0, min(4.0, gain)))

	for method_name in ("set_gain", "setGain"):
		method = getattr(source, method_name, None)
		if callable(method):
			try:
				method(safe_gain)
				return
			except Exception:
				pass

	for attr_name in ("gain", "volume"):
		if hasattr(source, attr_name):
			try:
				setattr(source, attr_name, safe_gain)
				return
			except Exception:
				pass


def stop_all_sounds() -> None:
	"""Detiene sonidos activos; se usa al apagar el sable o cerrar la app."""
	if AUDIO_BACKEND == "openal":
		for source in (
			OPENAL_ON_SOURCE,
			OPENAL_LOOP_SOURCE,
			OPENAL_OFF_SOURCE,
			OPENAL_PROJECTILE_REFLECT_SOURCE,
			OPENAL_LIFE_LOST_SOURCE,
			OPENAL_CUT_SOURCE,
			OPENAL_PARRY_SOURCE,
		):
			if source is None:
				continue
			try:
				source.stop()
			except Exception:
				pass

	if AUDIO_BACKEND == "pygame":
		try:
			if PYGAME_ON_CHANNEL is not None:
				PYGAME_ON_CHANNEL.stop()
			if PYGAME_LOOP_CHANNEL is not None:
				PYGAME_LOOP_CHANNEL.stop()
			if PYGAME_EFFECT_CHANNEL is not None:
				PYGAME_EFFECT_CHANNEL.stop()
		except pygame.error:
			pass


def play_saber_on_sound() -> bool:
	"""Efecto puntual de encendido."""
	return _play_backend_sound(OPENAL_ON_SOURCE, PYGAME_ON_SOUND, PYGAME_ON_CHANNEL, loops=0)


def play_saber_off_sound() -> bool:
	"""Efecto puntual de apagado."""
	return _play_backend_sound(OPENAL_OFF_SOURCE, PYGAME_OFF_SOUND, PYGAME_ON_CHANNEL, loops=0)


def play_saber_loop_sound() -> bool:
	"""Loop continuo de hoja encendida."""
	return _play_backend_sound(OPENAL_LOOP_SOURCE, PYGAME_LOOP_SOUND, PYGAME_LOOP_CHANNEL, loops=-1)


def play_projectile_reflect_sound() -> bool:
	return _play_backend_sound(
		OPENAL_PROJECTILE_REFLECT_SOURCE,
		PYGAME_PROJECTILE_REFLECT_SOUND,
		PYGAME_EFFECT_CHANNEL,
		loops=0,
	)


def play_life_lost_sound() -> bool:
	return _play_backend_sound(OPENAL_LIFE_LOST_SOURCE, PYGAME_LIFE_LOST_SOUND, PYGAME_EFFECT_CHANNEL, loops=0)


def play_cut_sound() -> bool:
	return _play_backend_sound(OPENAL_CUT_SOURCE, PYGAME_CUT_SOUND, PYGAME_EFFECT_CHANNEL, loops=0)


def play_parry_sound() -> bool:
	return _play_backend_sound(OPENAL_PARRY_SOURCE, PYGAME_PARRY_SOUND, PYGAME_EFFECT_CHANNEL, loops=0)


def shutdown_audio_backend() -> None:
	"""Libera recursos del mixer/fuentes al salir."""
	stop_all_sounds()
	if AUDIO_BACKEND == "openal" and oalQuit is not None:
		try:
			oalQuit()
		except Exception:
			pass

	if AUDIO_BACKEND == "pygame":
		try:
			if pygame.mixer.get_init():
				pygame.mixer.quit()
		except pygame.error:
			pass
