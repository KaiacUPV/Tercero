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


PRIMARY_MARKER = 17
DEFAULT_CUBE_SIZE_M = 0.065

LASER_ON_DURATION_S = 0.34
LASER_OFF_DURATION_S = 0.26
LASER_MIN_VISIBLE_POWER = 0.001
SABER_COLLISION_SAMPLES = 6
SABER_BLADE_BLOCK_RADIUS = 0.12
SABER_CUT_RADIUS_BONUS = 0.06
SABER_ON_SOUND_FILE = "sable-on.wav"
SABER_LOOP_SOUND_FILE = "loop.wav"
SABER_OFF_SOUND_FILE = "sable-off.wav"
PROJECTILE_REFLECT_SOUND_FILE = "disparo.wav"
LIFE_LOST_SOUND_FILE = "vida.wav"
CUT_SOUND_FILE = "corte.wav"
PARRY_SOUND_FILE = "parry.wav"
PARRY_SOUND_VOLUME = 1.0
SABER_LOOP_START_DELAY_S = 0.09
VOICE_CONTROL_ENABLED = True
VOICE_MODEL_DIR = "vosk-model-small-es-0.42"
VOICE_SAMPLE_RATE = 16000
VOICE_BLOCK_SIZE = 8000
VOICE_COMMAND_COOLDOWN_S = 0.75
VOICE_INPUT_DEVICE_HINT = "C-Media"

PHONE_ROTATION_ENABLED = True
PHONE_ROTATION_UDP_IP = "0.0.0.0"
PHONE_ROTATION_UDP_PORT = 8888
PHONE_ROTATION_AXIS_FLIP = (1.0, 1.0, 1.0)
PHONE_VR_LANDSCAPE_MAPPING = True
PHONE_VR_MAP_X = 1  # Eje del movil (0=X, 1=Y, 2=Z) mapeado al PITCH (mirar arriba/abajo)
PHONE_VR_MAP_Y = 2  # Eje del movil mapeado al YAW (mirar izquierda/derecha)
PHONE_VR_MAP_Z = 0  # Eje del movil mapeado al ROLL (inclinar cabeza sobre hombros)
PHONE_VR_INVERT_X = True
PHONE_VR_INVERT_Y = False
PHONE_VR_INVERT_Z = True
PHONE_ROTATION_3FLOAT_MODE = "rodrigues"  # "rodrigues" (rotation vector) o "euler_deg" (x,y,z en grados)
PHONE_ROTATION_STATUS_PRINT_EVERY_S = 1.0
PROJECTILE_RADIUS = 0.055
PROJECTILE_SPEED = 3.35
PROJECTILE_SPAWN_INTERVAL_S = 1.15
PROJECTILE_REBOUND_SPEED = 4.15
PROJECTILE_LASER_LENGTH = 0.30
GAME_MODE_BLOCKS = "blocks"
GAME_MODE_COMBAT = "combat"
COMBAT_ATTACK_INITIAL_DELAY_S = 0.85
COMBAT_IDLE_INTERVAL_MIN_S = 0.55
COMBAT_IDLE_INTERVAL_MAX_S = 1.05
COMBAT_WINDUP_DURATION_S = 0.34
COMBAT_STRIKE_DURATION_S = 0.40
COMBAT_RECOVER_DURATION_S = 0.42
COMBAT_BLADE_LENGTH = 1.26
COMBAT_BLADE_GLOW_RADIUS = 0.17
COMBAT_BLADE_CORE_RADIUS = 0.070
COMBAT_PARRY_DISTANCE = 0.16
COMBAT_PARRY_MIN_SPEED = 0.42
COMBAT_ATTACK_HIT_Z = -0.10
COMBAT_IDLE_SWAY_AMOUNT = 0.045
COMBAT_IDLE_LIFT_AMOUNT = 0.024
LIFE_LOST_FLASH_DURATION_S = 0.34
LIFE_LOST_SHAKE_DURATION_S = 0.22
LIFE_LOST_SHAKE_OFFSET = 0.018

VR_EYE_SEPARATION = 0.065

WINDOW_TITLE = "OpenGL Saber - Step 1"
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720

SABER_COLOR_PRESETS = {
	"r": {
		"name": "red",
		"outer": (1.0, 0.25, 0.18),
		"core": (1.0, 0.92, 0.90),
	},
	"g": {
		"name": "green",
		"outer": (0.30, 1.0, 0.34),
		"core": (0.90, 1.0, 0.92),
	},
	"b": {
		"name": "blue",
		"outer": (0.24, 0.62, 1.0),
		"core": (0.92, 0.98, 1.0),
	},
}

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


class VoiceCommandListener:
	def __init__(
		self,
		model_path: Path,
		sample_rate: int,
		block_size: int,
		device_hint: str | None = None,
	):
		self.model_path = Path(model_path)
		self.sample_rate = int(sample_rate)
		self.block_size = int(block_size)
		self.device_hint = (device_hint or "").strip().lower() or None
		self.error_message = None
		self.active_device = None
		self.active_device_name = None
		self.active_sample_rate = None

		self._model = None
		self._recognizer = None
		self._running = threading.Event()
		self._worker = None
		self._audio_queue: queue.Queue[bytes] = queue.Queue()
		self._command_queue: queue.Queue[tuple[str, str]] = queue.Queue()
		self._device_candidates: list[int] = []
		self._last_enqueued_text = ""

	def start(self) -> bool:
		if sd is None or Model is None or KaldiRecognizer is None:
			self.error_message = (
				"[Voice] Librerias faltantes. Instala: pip install vosk sounddevice"
			)
			return False

		if not self.model_path.exists():
			self.error_message = (
				f"[Voice] Modelo no encontrado en {self.model_path}. "
				"Descarga un modelo en https://alphacephei.com/vosk/models y extraelo ahi."
			)
			return False

		try:
			if SetLogLevel is not None:
				SetLogLevel(-1)

			self._model = Model(str(self.model_path))
			self._device_candidates = self._build_device_candidates()
			if not self._device_candidates:
				self.error_message = "[Voice] No hay microfonos de entrada disponibles."
				return False

			self._running.set()
			self._worker = threading.Thread(target=self._run, daemon=True)
			self._worker.start()
			return True
		except Exception as exc:
			self.error_message = f"[Voice] Error al iniciar voz: {exc}"
			self._running.clear()
			return False

	def stop(self) -> None:
		self._running.clear()
		if self._worker is not None:
			self._worker.join(timeout=1.0)
			self._worker = None

	def pop_commands(self) -> list[tuple[str, str]]:
		commands = []
		while True:
			try:
				commands.append(self._command_queue.get_nowait())
			except queue.Empty:
				break
		return commands

	@staticmethod
	def _extract_text(raw_json: str, key: str) -> str:
		try:
			payload = json.loads(raw_json)
		except json.JSONDecodeError:
			return ""

		text = str(payload.get(key, "")).strip().lower()
		return " ".join(text.split())

	@staticmethod
	def _parse_command(recognized_text: str) -> str | None:
		if not recognized_text:
			return None

		tokens = set(recognized_text.split())

		red_words = {"rojo", "roja", "red", "erre"}
		green_words = {"verde", "green", "ge"}
		blue_words = {"azul", "blue", "be"}
		combat_words = {"combate", "combat", "duelo", "parry"}
		block_mode_words = {"cubo", "cubos", "bloque", "bloques", "entrenamiento", "normal"}
		flip_words = {
			"invertir",
			"invierte",
			"invertido",
			"invertida",
			"espejo",
			"espejar",
			"flip",
			"mirror",
		}
		saber_words = {
			"sable",
			"sables",
			"saber",
			"sabre",
			"sabe",
			"sale",
		}
		restart_words = {
			"iniciar",
			"inciar",
			"inicia",
			"reiniciar",
			"reinicia",
			"reinicio",
			"restart",
		}

		if recognized_text in {"r", "g", "b"}:
			return f"color_{recognized_text}"
		if tokens & red_words:
			return "color_r"
		if tokens & green_words:
			return "color_g"
		if tokens & blue_words:
			return "color_b"
		if tokens & combat_words:
			return "start_combat"
		if tokens & block_mode_words:
			return "start_blocks_mode"
		if tokens & flip_words:
			return "toggle_flip"
		if tokens & restart_words:
			return "restart_mode"
		if "realidad virtual" in recognized_text or ("realidad" in tokens and "virtual" in tokens) or (tokens & {"vr"}):
			return "toggle_vr"

		toggle_phrases = (
			"encender sable",
			"enciende sable",
			"apagar sable",
			"apaga sable",
			"prender sable",
			"prende sable",
			"activar sable",
			"desactivar sable",
			"encender espada",
			"enciende espada",
			"apagar espada",
			"apaga espada",
			"activar espada",
			"desactivar espada",
		)
		if tokens & saber_words:
			return "toggle_power"
		if any(phrase in recognized_text for phrase in toggle_phrases):
			return "toggle_power"

		return None

	def _build_device_candidates(self) -> list[int]:
		try:
			devices = sd.query_devices()
		except Exception:
			return []

		ranked = []
		default_input = None
		try:
			default_pair = sd.default.device
			if isinstance(default_pair, (tuple, list)) and len(default_pair) > 0:
				maybe_index = int(default_pair[0])
				if maybe_index >= 0:
					default_input = maybe_index
		except Exception:
			default_input = None

		for idx, device in enumerate(devices):
			max_inputs = int(device.get("max_input_channels", 0))
			if max_inputs <= 0:
				continue

			name = str(device.get("name", ""))
			name_lower = name.lower()
			try:
				hostapi_name = str(sd.query_hostapis(device["hostapi"])["name"])
			except Exception:
				hostapi_name = "unknown"
			hostapi_lower = hostapi_name.lower()

			score = 0
			if default_input is not None and idx == default_input:
				score += 70
			if self.device_hint and self.device_hint in name_lower:
				score += 2000
			if "wasapi" in hostapi_lower:
				score += 260
			elif "wdm-ks" in hostapi_lower:
				score += 220
			elif "directsound" in hostapi_lower:
				score += 140
			elif "mme" in hostapi_lower:
				score += 80

			default_sr = float(device.get("default_samplerate", 0.0))
			if abs(default_sr - float(self.sample_rate)) < 1.0:
				score += 120
			if max_inputs >= 2:
				score += 40
			if "webcam" in name_lower and self.device_hint is None:
				score -= 180
			if ("hands-free" in name_lower or "auriculares con micr" in name_lower) and self.device_hint is None:
				score -= 140
			if "microsoft sound mapper" in name_lower or "controlador primario" in name_lower:
				score -= 900

			ranked.append((score, idx))

		ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
		return [idx for _, idx in ranked]

	@staticmethod
	def _describe_device(device_index: int) -> str:
		try:
			device = sd.query_devices(device_index)
			hostapi_name = sd.query_hostapis(device["hostapi"])["name"]
			return f"{device_index}: {device['name']} ({hostapi_name})"
		except Exception:
			return str(device_index)

	def _candidate_sample_rates(self, device_index: int) -> list[int]:
		rates = [int(max(8000, self.sample_rate))]
		try:
			device = sd.query_devices(device_index)
			default_rate = int(round(float(device.get("default_samplerate", 0.0))))
			if default_rate >= 8000 and default_rate not in rates:
				rates.append(default_rate)
		except Exception:
			pass
		return rates

	def _enqueue_command_if_any(self, recognized_text: str) -> None:
		normalized_text = " ".join(recognized_text.strip().lower().split())
		if not normalized_text:
			return

		command = self._parse_command(normalized_text)
		if command is None:
			return
		if normalized_text == self._last_enqueued_text:
			return

		self._last_enqueued_text = normalized_text
		self._command_queue.put((command, normalized_text))

	def _run_recognition_loop(self) -> None:
		while self._running.is_set():
			try:
				chunk = self._audio_queue.get(timeout=0.2)
			except queue.Empty:
				continue

			if self._recognizer is None:
				continue

			if self._recognizer.AcceptWaveform(chunk):
				final_text = self._extract_text(self._recognizer.Result(), "text")
				self._enqueue_command_if_any(final_text)
			else:
				partial_text = self._extract_text(self._recognizer.PartialResult(), "partial")
				self._enqueue_command_if_any(partial_text)

	def _audio_callback(self, indata, frames, time_info, status) -> None:
		if not self._running.is_set():
			return
		if status:
			return
		self._audio_queue.put(bytes(indata))

	def _run(self) -> None:
		last_error = None
		try:
			for device_index in self._device_candidates:
				if not self._running.is_set():
					break

				self.active_device = device_index
				self.active_device_name = self._describe_device(device_index)
				for stream_rate in self._candidate_sample_rates(device_index):
					if not self._running.is_set():
						break

					while True:
						try:
							self._audio_queue.get_nowait()
						except queue.Empty:
							break

					self.active_sample_rate = stream_rate
					try:
						if self._model is None:
							raise RuntimeError("Modelo Vosk no inicializado")
						self._recognizer = KaldiRecognizer(self._model, float(stream_rate))
						print(
							f"[Voice] Microfono seleccionado: {self.active_device_name} "
							f"@ {stream_rate} Hz"
						)
						with sd.RawInputStream(
							samplerate=stream_rate,
							blocksize=self.block_size,
							dtype="int16",
							channels=1,
							device=device_index,
							callback=self._audio_callback,
						):
							self._run_recognition_loop()
						return
					except Exception as exc:
						last_error = exc
						print(
							f"[Voice] No se pudo abrir {self.active_device_name} "
							f"a {stream_rate} Hz: {exc}"
						)

			if self._running.is_set():
				if last_error is None:
					self.error_message = "[Voice] No se pudo iniciar el microfono de voz."
				else:
					self.error_message = f"[Voice] Error en microfono/reconocimiento: {last_error}"
		except Exception as exc:
			self.error_message = f"[Voice] Error en el hilo de voz: {exc}"
		finally:
			self._running.clear()


class PhoneRotationListener(threading.Thread):
	def __init__(self, udp_ip: str, udp_port: int):
		super().__init__(daemon=True)
		self.udp_ip = str(udp_ip)
		self.udp_port = int(udp_port)
		self.error_message: str | None = None
		self._stop_event = threading.Event()
		self._sock: socket.socket | None = None
		self._queue: queue.Queue[str] = queue.Queue(maxsize=32)

	def start_listening(self) -> bool:
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
		self._stop_event.set()
		if self._sock is not None:
			try:
				self._sock.close()
			except OSError:
				pass
			self._sock = None

	def pop_latest_message(self) -> str | None:
		latest = None
		while True:
			try:
				latest = self._queue.get_nowait()
			except queue.Empty:
				break
		return latest


def init_audio_backend(
	saber_on_path: Path,
	saber_loop_path: Path,
	saber_off_path: Path,
	projectile_reflect_path: Path,
	life_lost_path: Path,
	cut_path: Path,
	parry_path: Path,
) -> str:
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


def set_audio_source_gain(source, gain: float) -> None:
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
	if AUDIO_BACKEND == "openal":
		return play_openal_source(OPENAL_ON_SOURCE)
	if AUDIO_BACKEND == "pygame":
		if PYGAME_ON_SOUND is None or PYGAME_ON_CHANNEL is None:
			return False
		try:
			PYGAME_ON_CHANNEL.play(PYGAME_ON_SOUND, loops=0)
			return True
		except pygame.error:
			return False
	return False


def play_saber_off_sound() -> bool:
	if AUDIO_BACKEND == "openal":
		return play_openal_source(OPENAL_OFF_SOURCE)
	if AUDIO_BACKEND == "pygame":
		if PYGAME_OFF_SOUND is None or PYGAME_ON_CHANNEL is None:
			return False
		try:
			PYGAME_ON_CHANNEL.play(PYGAME_OFF_SOUND, loops=0)
			return True
		except pygame.error:
			return False
	return False


def play_saber_loop_sound() -> bool:
	if AUDIO_BACKEND == "openal":
		return play_openal_source(OPENAL_LOOP_SOURCE)
	if AUDIO_BACKEND == "pygame":
		if PYGAME_LOOP_SOUND is None or PYGAME_LOOP_CHANNEL is None:
			return False
		try:
			PYGAME_LOOP_CHANNEL.play(PYGAME_LOOP_SOUND, loops=-1)
			return True
		except pygame.error:
			return False
	return False


def play_projectile_reflect_sound() -> bool:
	if AUDIO_BACKEND == "openal":
		return play_openal_source(OPENAL_PROJECTILE_REFLECT_SOURCE)
	if AUDIO_BACKEND == "pygame":
		if PYGAME_PROJECTILE_REFLECT_SOUND is None or PYGAME_EFFECT_CHANNEL is None:
			return False
		try:
			PYGAME_EFFECT_CHANNEL.play(PYGAME_PROJECTILE_REFLECT_SOUND, loops=0)
			return True
		except pygame.error:
			return False
	return False


def play_life_lost_sound() -> bool:
	if AUDIO_BACKEND == "openal":
		return play_openal_source(OPENAL_LIFE_LOST_SOURCE)
	if AUDIO_BACKEND == "pygame":
		if PYGAME_LIFE_LOST_SOUND is None or PYGAME_EFFECT_CHANNEL is None:
			return False
		try:
			PYGAME_EFFECT_CHANNEL.play(PYGAME_LIFE_LOST_SOUND, loops=0)
			return True
		except pygame.error:
			return False
	return False


def play_cut_sound() -> bool:
	if AUDIO_BACKEND == "openal":
		return play_openal_source(OPENAL_CUT_SOURCE)
	if AUDIO_BACKEND == "pygame":
		if PYGAME_CUT_SOUND is None or PYGAME_EFFECT_CHANNEL is None:
			return False
		try:
			PYGAME_EFFECT_CHANNEL.play(PYGAME_CUT_SOUND, loops=0)
			return True
		except pygame.error:
			return False
	return False


def play_parry_sound() -> bool:
	if AUDIO_BACKEND == "openal":
		return play_openal_source(OPENAL_PARRY_SOURCE)
	if AUDIO_BACKEND == "pygame":
		if PYGAME_PARRY_SOUND is None or PYGAME_EFFECT_CHANNEL is None:
			return False
		try:
			PYGAME_EFFECT_CHANNEL.play(PYGAME_PARRY_SOUND, loops=0)
			return True
		except pygame.error:
			return False
	return False


def shutdown_audio_backend() -> None:
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


@dataclass
class PoseSample:
	rotation_cv: np.ndarray
	translation_cv: np.ndarray
	timestamp_s: float


def create_combined_rotation(rotation_axis, additional_z_rotation=0.0):
	r1 = cv2.Rodrigues(np.array(rotation_axis, dtype=np.float32))[0]
	if additional_z_rotation != 0.0:
		r2 = cv2.Rodrigues(np.array([0.0, 0.0, additional_z_rotation], dtype=np.float32))[0]
		return r1 @ r2
	return r1


def build_cube_geometry(cube_size_m: float):
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
	data = np.load(npz_path)
	mtx_key = "mtx" if "mtx" in data.files else "camera_matrix"
	dist_key = "dist" if "dist" in data.files else "dist_coeffs"
	return data[mtx_key].astype(np.float32), data[dist_key].astype(np.float32)


def select_reference_face(corners, ids, cube_markers_3d):
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
	script_dir = Path(__file__).resolve().parent
	candidates = [
		Path(primary),
		script_dir / primary,
		Path(fallback),
		script_dir / fallback,
	]

	for candidate in candidates:
		if candidate.exists():
			return candidate

	searched = "\n".join(f"- {path}" for path in candidates)
	raise FileNotFoundError(
		"No se encontro archivo de calibracion de camara. Revisado:\n" + searched
	)


class ArucoPoseTracker(threading.Thread):
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
		self._running.clear()

	def get_latest_pose(self) -> Optional[PoseSample]:
		with self._lock:
			if self._latest_pose is None:
				return None
			return PoseSample(
				rotation_cv=self._latest_pose.rotation_cv.copy(),
				translation_cv=self._latest_pose.translation_cv.copy(),
				timestamp_s=self._latest_pose.timestamp_s,
			)

	def run(self):
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


def parse_resolution(resolution: str) -> tuple[int, int]:
	pieces = resolution.lower().split("x")
	if len(pieces) != 2:
		raise ValueError("Resolucion invalida. Usa WIDTHxHEIGHT, ejemplo 1280x720")
	width = int(pieces[0])
	height = int(pieces[1])
	if width <= 0 or height <= 0:
		raise ValueError("La resolucion debe tener valores positivos")
	return width, height


def clamp(value: float, min_value: float, max_value: float) -> float:
	return float(max(min_value, min(max_value, value)))


def orthonormalize_rotation(rotation: np.ndarray) -> np.ndarray:
	u, _, vt = np.linalg.svd(rotation.astype(np.float64))
	result = u @ vt
	if np.linalg.det(result) < 0.0:
		u[:, -1] *= -1.0
		result = u @ vt
	return result.astype(np.float32)


def rotation_matrix_from_euler_xyz_deg(x_deg: float, y_deg: float, z_deg: float) -> np.ndarray:
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
	rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
	rot, _ = cv2.Rodrigues(rvec)
	return orthonormalize_rotation(rot.astype(np.float32))


def parse_csv_floats(message: str) -> list[float]:
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


def draw_box(center: tuple[float, float, float], size: tuple[float, float, float], color_rgba: tuple[float, float, float, float]):
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

def normalize(vector: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
	length = float(np.linalg.norm(vector))
	if length <= 1e-6:
		if fallback is None:
			return np.zeros_like(vector, dtype=np.float32)
		return fallback.astype(np.float32)
	return (vector / length).astype(np.float32)


def distance_point_to_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
	segment = end - start
	length_sq = float(np.dot(segment, segment))
	if length_sq <= 1e-8:
		return float(np.linalg.norm(point - start))
	t = clamp(float(np.dot(point - start, segment) / length_sq), 0.0, 1.0)
	closest = start + (segment * t)
	return float(np.linalg.norm(point - closest))


def lerp_vec(start: np.ndarray, end: np.ndarray, t: float) -> np.ndarray:
	return (start + ((end - start) * float(t))).astype(np.float32)


def smoothstep01(value: float) -> float:
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
	blade_dir_n = normalize(blade_direction.astype(np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32))
	return rotation_matrix_from_forward(-blade_dir_n)


def draw_oriented_box(position: np.ndarray, rotation: np.ndarray, size: np.ndarray, color_rgba: tuple[float, float, float, float]):
	glPushMatrix()
	transform = np.eye(4, dtype=np.float32)
	transform[:3, :3] = rotation.astype(np.float32)
	transform[:3, 3] = position.astype(np.float32)
	glMultMatrixf(transform.T)
	draw_box((0.0, 0.0, 0.0), (float(size[0]), float(size[1]), float(size[2])), color_rgba)
	glPopMatrix()


def draw_oriented_diamond(position: np.ndarray, rotation: np.ndarray, size: np.ndarray, color_rgba: tuple[float, float, float, float]):
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
	text = str(max(0, int(value)))
	spacing = digit_w * 0.22
	cursor_x = x
	for ch in text:
		draw_digit_7seg(cursor_x, y, digit_w, digit_h, int(ch), color_rgba)
		cursor_x += digit_w + spacing


@dataclass
class BeatBlock:
	position: np.ndarray
	velocity: np.ndarray
	shape_key: str
	size: np.ndarray
	color_rgba: tuple[float, float, float, float]
	rotation: np.ndarray = None
	was_cut: bool = False
	passed_player: bool = False

	def __post_init__(self):
		if self.rotation is None:
			self.rotation = np.eye(3, dtype=np.float32)
		else:
			self.rotation = orthonormalize_rotation(self.rotation)


@dataclass
class BlockFragment:
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
	position: np.ndarray
	velocity: np.ndarray
	radius: float
	color_rgba: tuple[float, float, float, float]
	deflected: bool = False
	lifetime_s: float = 3.0


@dataclass
class EnemyCombatSaber:
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


class OpenGLSaberApp:
	def __init__(
		self,
		tracker: ArucoPoseTracker,
		width: int,
		height: int,
		pose_scale: float,
		pos_smoothing: float,
		lost_timeout_s: float,
		max_fps: int,
		vsync: bool,
	):
		self.tracker = tracker
		self.width = int(width)
		self.height = int(height)
		self.pose_scale = float(pose_scale)
		self.pos_smoothing = float(np.clip(pos_smoothing, 0.0, 1.0))
		self.lost_timeout_s = float(max(0.05, lost_timeout_s))
		self.max_fps = int(max(30, max_fps))
		self.vsync = bool(vsync)

		self.hand_anchor_local = np.array([0.0, -0.18, -1.15], dtype=np.float32)
		self.hand_motion_gain = 0.75
		self.control_mirror_x = True
		self.motion_bounds = np.array([0.82, 0.58, 0.72], dtype=np.float32)

		self.reference_translation_cv: Optional[np.ndarray] = None
		self.reference_rotation_cv: Optional[np.ndarray] = None
		self.filtered_pos: Optional[np.ndarray] = None
		self.last_local_offset: Optional[np.ndarray] = None
		self.local_velocity = np.zeros(3, dtype=np.float32)
		self.prediction_horizon_s = 0.08

		self.has_pose = False
		self.last_pose_time_s = 0.0
		self._last_pose_state = "waiting"

		self.current_pos = self.hand_anchor_local.copy()
		self.current_rot = np.eye(3, dtype=np.float32)
		self.prev_saber_base_pos = self.current_pos.copy()
		self.prev_saber_tip_pos = self.current_pos.copy()
		self.prev_saber_center_pos = self.current_pos.copy()
		self.saber_base_pos = self.current_pos.copy()
		self.saber_tip_pos = self.current_pos.copy()
		self.saber_center_pos = self.current_pos.copy()
		self.saber_tip_velocity = np.zeros(3, dtype=np.float32)

		self.saber_rest_rot = rotation_matrix_from_euler_xyz_deg(18.0, 0.0, 8.0)

		self.saber_color_key = "b"
		self.blade_target_on = False
		self.blade_power = 0.0
		self.saber_on_sound_path = Path(__file__).with_name(SABER_ON_SOUND_FILE)
		self.saber_loop_sound_path = Path(__file__).with_name(SABER_LOOP_SOUND_FILE)
		self.saber_off_sound_path = Path(__file__).with_name(SABER_OFF_SOUND_FILE)
		self.projectile_reflect_sound_path = Path(__file__).with_name(PROJECTILE_REFLECT_SOUND_FILE)
		self.life_lost_sound_path = Path(__file__).with_name(LIFE_LOST_SOUND_FILE)
		self.cut_sound_path = Path(__file__).with_name(CUT_SOUND_FILE)
		self.parry_sound_path = Path(__file__).with_name(PARRY_SOUND_FILE)
		self.audio_backend = init_audio_backend(
			self.saber_on_sound_path,
			self.saber_loop_sound_path,
			self.saber_off_sound_path,
			self.projectile_reflect_sound_path,
			self.life_lost_sound_path,
			self.cut_sound_path,
			self.parry_sound_path,
		)
		self.loop_start_delay_s = SABER_LOOP_START_DELAY_S
		self.loop_start_due_time: Optional[float] = None
		self.loop_playing = False
		self.voice_listener: Optional[VoiceCommandListener] = None
		self.voice_last_command_time = {
			"toggle_power": 0.0,
			"toggle_flip": 0.0,
			"color_r": 0.0,
			"color_g": 0.0,
			"color_b": 0.0,
			"start_blocks_mode": 0.0,
			"start_combat": 0.0,
			"restart_mode": 0.0,
			"toggle_vr": 0.0,
		}

		self.vr_mode = False
		self.vr_eye_separation = float(VR_EYE_SEPARATION)

		self.game_mode = GAME_MODE_BLOCKS
		self.block_spawn_interval_s = 0.82
		self.block_spawn_timer_s = 0.0
		self.block_speed = 2.45
		self.block_lanes_x = [-0.66, -0.22, 0.22, 0.66]
		self.block_lanes_y = [-0.20, 0.18, 0.56]
		self.block_palette = [
			(0.92, 0.24, 0.24, 0.96),
			(0.22, 0.84, 0.38, 0.96),
			(0.22, 0.56, 0.96, 0.96),
			(0.98, 0.78, 0.22, 0.96),
			(0.92, 0.34, 0.88, 0.96),
		]
		self.blocks: list[BeatBlock] = []
		self.fragments: list[BlockFragment] = []
		self.projectiles: list[Projectile] = []
		self.enemy_saber: Optional[EnemyCombatSaber] = None
		self.score = 0
		self.misses = 0
		self.max_lives = 10
		self.lives = self.max_lives
		self.damage_flash_until_s = 0.0
		self.damage_shake_until_s = 0.0
		self.blocks_started = False
		self.game_over = False
		self.projectile_spawn_interval_s = PROJECTILE_SPAWN_INTERVAL_S
		self.projectile_spawn_timer_s = 0.0

		self.running = True

		pygame.init()
		pygame.display.set_caption(WINDOW_TITLE)
		self._create_window(self.width, self.height)
		self._init_gl_state()

		self.clock = pygame.time.Clock()
		self._init_voice_control()
		self._init_phone_rotation_control()
		self._sync_audio_with_current_blade_state()

	def _create_window(self, width: int, height: int):
		flags = DOUBLEBUF | OPENGL | FULLSCREEN
		try:
			surface = pygame.display.set_mode((0, 0), flags, vsync=1 if self.vsync else 0)
		except TypeError:
			surface = pygame.display.set_mode((0, 0), flags)
		self.width, self.height = surface.get_size()

	def _init_gl_state(self):
		glEnable(GL_DEPTH_TEST)
		glDepthFunc(GL_LEQUAL)
		glEnable(GL_BLEND)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
		glClearColor(0.04, 0.05, 0.08, 1.0)
		self._set_projection(self.width, self.height)

	def _set_projection(self, width: int, height: int):
		safe_w = max(1, int(width))
		safe_h = max(1, int(height))
		self.width = safe_w
		self.height = safe_h

		glViewport(0, 0, safe_w, safe_h)
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		gluPerspective(88.0, safe_w / safe_h, 0.01, 100.0)
		glMatrixMode(GL_MODELVIEW)

	def _init_voice_control(self):
		if self.audio_backend == "none":
			print("[Audio] Sin backend. Instala PyOpenAL (pip install PyOpenAL) o pygame.")
		else:
			print(f"[Audio] Backend activo: {self.audio_backend}")

		if not VOICE_CONTROL_ENABLED:
			return

		voice_model_path = Path(__file__).with_name(VOICE_MODEL_DIR)
		self.voice_listener = VoiceCommandListener(
			model_path=voice_model_path,
			sample_rate=VOICE_SAMPLE_RATE,
			block_size=VOICE_BLOCK_SIZE,
			device_hint=VOICE_INPUT_DEVICE_HINT,
		)
		if self.voice_listener.start():
			time.sleep(0.15)
			if self.voice_listener.active_device_name:
				if self.voice_listener.active_sample_rate is not None:
					print(
						f"[Voice] Entrada activa: {self.voice_listener.active_device_name} "
						f"@ {self.voice_listener.active_sample_rate} Hz"
					)
				else:
					print(f"[Voice] Entrada activa: {self.voice_listener.active_device_name}")
			print("[Voice] Control activo: di 'sable' o 'saber', 'combate', 'cubos', 'reiniciar', 'rojo/verde/azul' o 'invertir'.")
		else:
			print(self.voice_listener.error_message)
			self.voice_listener = None

	def _init_phone_rotation_control(self):
		self.phone_listener: Optional[PhoneRotationListener] = None
		self.phone_reference_rot: Optional[np.ndarray] = None
		self.phone_camera_rot = np.eye(3, dtype=np.float32)
		self.phone_has_rotation = False
		self.phone_last_message: str | None = None
		self.phone_last_update_s = 0.0
		self.phone_last_print_s = 0.0

		if not PHONE_ROTATION_ENABLED:
			return

		listener = PhoneRotationListener(PHONE_ROTATION_UDP_IP, PHONE_ROTATION_UDP_PORT)
		if listener.start_listening():
			self.phone_listener = listener
			print(f"[Phone] Escuchando rotacion por UDP en {PHONE_ROTATION_UDP_IP}:{PHONE_ROTATION_UDP_PORT}")
		else:
			if listener.error_message:
				print(listener.error_message)

	def _process_phone_rotation(self):
		if self.phone_listener is None:
			return
		if self.phone_listener.error_message:
			print(self.phone_listener.error_message)
			self.phone_listener.stop()
			self.phone_listener = None
			return

		message = self.phone_listener.pop_latest_message()
		if not message:
			return

		values = parse_csv_floats(message)
		if len(values) < 3:
			return

		now_s = time.perf_counter()

		rot_raw: np.ndarray
		if len(values) >= 4:
			# Preferimos los ultimos 4/3 floats por si el mensaje trae timestamp/campos extra al principio.
			last4 = np.array(values[-4:], dtype=np.float64)
			first4 = np.array(values[:4], dtype=np.float64)

			def _looks_like_quat(q: np.ndarray) -> bool:
				q_abs_max = float(np.max(np.abs(q)))
				q_norm = float(np.linalg.norm(q))
				return q_abs_max <= 1.2 and 0.25 <= q_norm <= 2.5

			if _looks_like_quat(last4):
				rot_raw = rotation_matrix_from_quaternion_xyzw(float(last4[0]), float(last4[1]), float(last4[2]), float(last4[3]))
			elif _looks_like_quat(first4):
				rot_raw = rotation_matrix_from_quaternion_xyzw(float(first4[0]), float(first4[1]), float(first4[2]), float(first4[3]))
			else:
				v3 = values[-3:]
				if PHONE_ROTATION_3FLOAT_MODE == "euler_deg":
					rot_raw = rotation_matrix_from_euler_xyz_deg(float(v3[0]), float(v3[1]), float(v3[2]))
				else:
					rvec = np.array(v3, dtype=np.float32)
					rot_raw = rotation_matrix_from_rodrigues(rvec)
		else:
			v3 = values[-3:]
			if PHONE_ROTATION_3FLOAT_MODE == "euler_deg":
				rot_raw = rotation_matrix_from_euler_xyz_deg(float(v3[0]), float(v3[1]), float(v3[2]))
			else:
				rvec = np.array(v3, dtype=np.float32)
				rot_raw = rotation_matrix_from_rodrigues(rvec)

		axis_flip = np.diag(
			[float(PHONE_ROTATION_AXIS_FLIP[0]), float(PHONE_ROTATION_AXIS_FLIP[1]), float(PHONE_ROTATION_AXIS_FLIP[2])]
		).astype(np.float32)
		rot_raw = axis_flip @ rot_raw @ axis_flip
		if not np.isfinite(rot_raw).all():
			return

		if PHONE_VR_LANDSCAPE_MAPPING:
			# Matriz de transformacion flexible para asignar qué eje del movil mueve la camara
			swap_mat = np.zeros((3, 3), dtype=np.float32)
			swap_mat[0, PHONE_VR_MAP_X] = -1.0 if PHONE_VR_INVERT_X else 1.0
			swap_mat[1, PHONE_VR_MAP_Y] = -1.0 if PHONE_VR_INVERT_Y else 1.0
			swap_mat[2, PHONE_VR_MAP_Z] = -1.0 if PHONE_VR_INVERT_Z else 1.0
			
			rot_raw = swap_mat @ rot_raw @ swap_mat.T

		if self.phone_reference_rot is None:
			self.phone_reference_rot = rot_raw.astype(np.float32).copy()
			self.phone_camera_rot = np.eye(3, dtype=np.float32)
			self.phone_has_rotation = True
			print(f"[Phone] Referencia de rotacion fijada: {message}")
			self.phone_last_message = message
			self.phone_last_update_s = now_s
			self.phone_last_print_s = now_s
			return

		self.phone_camera_rot = orthonormalize_rotation(rot_raw @ self.phone_reference_rot.T)
		self.phone_has_rotation = True
		self.phone_last_message = message
		self.phone_last_update_s = now_s
		if PHONE_ROTATION_STATUS_PRINT_EVERY_S > 0.0 and (now_s - self.phone_last_print_s) >= float(PHONE_ROTATION_STATUS_PRINT_EVERY_S):
			forward = (self.phone_camera_rot @ np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)
			print(f"[Phone] {message} -> forward=({forward[0]:.2f},{forward[1]:.2f},{forward[2]:.2f})")
			self.phone_last_print_s = now_s

	def _set_blade_target(self, enabled: bool, source: str, spoken_text: Optional[str] = None):
		if self.blade_target_on == enabled:
			return

		now = time.perf_counter()
		self.blade_target_on = enabled
		stop_all_sounds()
		self.loop_playing = False

		if enabled:
			play_saber_on_sound()
			self.loop_start_due_time = now + max(0.0, self.loop_start_delay_s)
		else:
			play_saber_off_sound()
			self.loop_start_due_time = None

		if source == "voice" and spoken_text:
			print(f"[Voice] '{spoken_text}' -> sable {'ON' if enabled else 'OFF'}")
		else:
			print(f"[Saber] Hoja {'ON' if enabled else 'OFF'}")

	def _lose_life(self, amount: int = 1, reason: str = "impacto"):
		if self.game_over:
			return
		self.lives = max(0, self.lives - int(max(0, amount)))
		now_s = time.perf_counter()
		self.damage_flash_until_s = max(self.damage_flash_until_s, now_s + LIFE_LOST_FLASH_DURATION_S)
		self.damage_shake_until_s = max(self.damage_shake_until_s, now_s + LIFE_LOST_SHAKE_DURATION_S)
		play_life_lost_sound()
		print(f"[HUD] Vida -{amount} por {reason}. Quedan {self.lives}/{self.max_lives}")
		if self.lives <= 0:
			self.game_over = True
			self.blocks_started = False
			self.projectiles.clear()
			self.blocks.clear()
			print("[Game] Sin vidas. Fin de partida.")

	def _clear_gameplay_entities(self):
		self.blocks.clear()
		self.projectiles.clear()
		self.fragments.clear()
		self.enemy_saber = None

	def _reset_common_game_state(self):
		self.score = 0
		self.misses = 0
		self.lives = self.max_lives
		self.damage_flash_until_s = 0.0
		self.damage_shake_until_s = 0.0
		self.game_over = False
		self.block_spawn_timer_s = 0.0
		self.projectile_spawn_timer_s = 0.0
		self._clear_gameplay_entities()

	def _reset_block_mode(self):
		self._reset_common_game_state()
		self.game_mode = GAME_MODE_BLOCKS
		self.blocks_started = True
		self._spawn_block(initial_z=-4.2)
		self._spawn_block(initial_z=-5.6)
		print("[Game] Modo cubos reiniciado")

	def _reset_combat_mode(self):
		self._reset_common_game_state()
		self.game_mode = GAME_MODE_COMBAT
		self.blocks_started = False
		self.enemy_saber = self._create_enemy_saber()
		print("[Game] Modo combate reiniciado")

	def _restart_current_mode(self):
		if self.game_mode == GAME_MODE_COMBAT:
			self._reset_combat_mode()
		else:
			self._reset_block_mode()

	def _sync_audio_with_current_blade_state(self):
		stop_all_sounds()
		self.loop_playing = False
		if self.blade_target_on:
			play_saber_on_sound()
			self.loop_start_due_time = time.perf_counter() + max(0.0, self.loop_start_delay_s)
		else:
			self.loop_start_due_time = None

	def _update_audio_loop(self):
		now = time.perf_counter()
		if self.blade_target_on and not self.loop_playing and self.loop_start_due_time is not None and now >= self.loop_start_due_time:
			self.loop_playing = play_saber_loop_sound()

	def _process_voice_commands(self):
		if self.voice_listener is None:
			return
		if self.voice_listener.error_message:
			print(self.voice_listener.error_message)
			self.voice_listener.stop()
			self.voice_listener = None
			return

		for voice_command, spoken_text in self.voice_listener.pop_commands():
			command_now = time.perf_counter()
			if (command_now - self.voice_last_command_time.get(voice_command, 0.0)) < VOICE_COMMAND_COOLDOWN_S:
				continue
			self.voice_last_command_time[voice_command] = command_now

			if voice_command == "toggle_power":
				self._set_blade_target(not self.blade_target_on, source="voice", spoken_text=spoken_text)
			elif voice_command == "start_combat":
				self._reset_combat_mode()
				print(f"[Voice] '{spoken_text}' -> modo combate")
			elif voice_command == "start_blocks_mode":
				self._reset_block_mode()
				print(f"[Voice] '{spoken_text}' -> modo cubos")
			elif voice_command == "restart_mode":
				self._restart_current_mode()
				mode_name = "combate" if self.game_mode == GAME_MODE_COMBAT else "cubos"
				print(f"[Voice] '{spoken_text}' -> {mode_name} reiniciado")
			elif voice_command == "toggle_flip":
				self.control_mirror_x = not self.control_mirror_x
				print(
					f"[Voice] '{spoken_text}' -> inversion horizontal: "
					f"{'ON' if self.control_mirror_x else 'OFF'}"
				)
			elif voice_command == "toggle_vr":
				self.vr_mode = not self.vr_mode
				print(f"[Voice] '{spoken_text}' -> VR {'ON' if self.vr_mode else 'OFF'}")
			elif voice_command.startswith("color_"):
				color_key = voice_command[-1]
				if color_key in SABER_COLOR_PRESETS:
					self.saber_color_key = color_key
					print(f"[Voice] '{spoken_text}' -> color {SABER_COLOR_PRESETS[color_key]['name']}")

	def _toggle_vr_mode(self, source: str = "keyboard"):
		new_state = not self.vr_mode
		self.vr_mode = new_state
		if new_state:
			# Al entrar en VR, re-usamos la orientacion actual del movil como referencia.
			self.phone_reference_rot = None
			self.phone_camera_rot = np.eye(3, dtype=np.float32)
			self.phone_has_rotation = False
		print(f"[VR] {'ON' if self.vr_mode else 'OFF'} ({source})")

	def _shutdown_runtime_services(self):
		if self.phone_listener is not None:
			self.phone_listener.stop()
			self.phone_listener = None
		if self.voice_listener is not None:
			self.voice_listener.stop()
			self.voice_listener = None
		shutdown_audio_backend()

	def _cv_local_offset_from_translation(self, translation_cv: np.ndarray) -> np.ndarray:
		if self.reference_translation_cv is None:
			self.reference_translation_cv = translation_cv.astype(np.float32).copy()

		delta_cv = translation_cv.astype(np.float32) - self.reference_translation_cv
		x_sign = -1.0 if self.control_mirror_x else 1.0

		mapped = np.array(
			[
				x_sign * delta_cv[0],
				-delta_cv[1],
				-delta_cv[2],
			],
			dtype=np.float32,
		)
		mapped *= self.pose_scale * self.hand_motion_gain

		mapped[0] = clamp(mapped[0], -self.motion_bounds[0], self.motion_bounds[0])
		mapped[1] = clamp(mapped[1], -self.motion_bounds[1], self.motion_bounds[1])
		mapped[2] = clamp(mapped[2], -self.motion_bounds[2], self.motion_bounds[2])
		return mapped

	def _cv_relative_rotation_gl(self, rotation_cv: np.ndarray) -> np.ndarray:
		if self.reference_rotation_cv is None:
			self.reference_rotation_cv = rotation_cv.astype(np.float32).copy()
		
		rel_cv = rotation_cv @ self.reference_rotation_cv.T
		rel_cv = orthonormalize_rotation(rel_cv)

        # Conversión base de cámara OpenCV a OpenGL
		cv_to_gl = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float32,
        )
		rel_gl = cv_to_gl @ rel_cv @ cv_to_gl.T

        # --- CORRECCIÓN DE ORIENTACIÓN ---
        # Aplicamos una inversión en los ejes X e Y (una rotación de 180º en Z).
        # Esto invierte específicamente el Pitch (arriba/abajo) y el Yaw (izq/der),
        # manteniendo intacto el Roll (giro de la muñeca sobre sí misma).
		flip_pitch_yaw = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
		rel_gl = flip_pitch_yaw @ rel_gl @ flip_pitch_yaw.T

		return orthonormalize_rotation(rel_gl)

	def _update_pose(self):
		pose = self.tracker.get_latest_pose()
		now_s = time.perf_counter()

		if pose is not None:
			local_offset = self._cv_local_offset_from_translation(pose.translation_cv)
			if self.last_local_offset is not None:
				dt_pose = max(1e-4, pose.timestamp_s - self.last_pose_time_s)
				measured_velocity = (local_offset - self.last_local_offset) / dt_pose
				self.local_velocity = (
					(self.local_velocity * 0.45) + (measured_velocity.astype(np.float32) * 0.55)
				).astype(np.float32)
			self.last_local_offset = local_offset.astype(np.float32)
			target_pos = self.hand_anchor_local + local_offset

			rel_rot = self._cv_relative_rotation_gl(pose.rotation_cv)
			target_rot = orthonormalize_rotation(rel_rot @ self.saber_rest_rot)

			if self.filtered_pos is None:
				self.filtered_pos = target_pos
			else:
				alpha = self.pos_smoothing
				self.filtered_pos = (self.filtered_pos * (1.0 - alpha)) + (target_pos * alpha)

			self.current_pos = self.filtered_pos.astype(np.float32)
			self.current_rot = target_rot.astype(np.float32)
			self.last_pose_time_s = pose.timestamp_s
			self.has_pose = True

			if self._last_pose_state != "locked":
				print("[Tracking] Pose bloqueada")
				self._last_pose_state = "locked"
		elif not self.has_pose:
			if self._last_pose_state != "waiting":
				print("[Tracking] Esperando deteccion ArUco en camara local")
				self._last_pose_state = "waiting"
		elif (now_s - self.last_pose_time_s) <= self.prediction_horizon_s and self.filtered_pos is not None and self.last_local_offset is not None:
			extrapolated_offset = clamp(now_s - self.last_pose_time_s, 0.0, self.prediction_horizon_s) * self.local_velocity
			target_pos = self.hand_anchor_local + self.last_local_offset + extrapolated_offset
			self.filtered_pos = (self.filtered_pos * 0.35) + (target_pos.astype(np.float32) * 0.65)
			self.current_pos = self.filtered_pos.astype(np.float32)
		elif (now_s - self.last_pose_time_s) > self.lost_timeout_s:
			if self._last_pose_state != "lost":
				print("[Tracking] Pose perdida, manteniendo ultima transformacion")
				self._last_pose_state = "lost"

	def _compute_saber_points(self):
		base_local = np.array([0.0, 0.0, -0.36], dtype=np.float32)
		center_local = np.array([0.0, 0.0, -1.01], dtype=np.float32)
		tip_local = np.array([0.0, 0.0, -1.66], dtype=np.float32)
		self.saber_base_pos = self.current_pos + (self.current_rot @ base_local)
		self.saber_center_pos = self.current_pos + (self.current_rot @ center_local)
		self.saber_tip_pos = self.current_pos + (self.current_rot @ tip_local)

	def _draw_lightsaber_model(
		self,
		position: np.ndarray,
		rotation: np.ndarray,
		blade_power: float,
		outer_color: tuple[float, float, float],
		core_color: tuple[float, float, float],
	):
		glPushMatrix()

		transform = np.eye(4, dtype=np.float32)
		transform[:3, :3] = rotation.astype(np.float32)
		transform[:3, 3] = position.astype(np.float32)
		glMultMatrixf(transform.T)

		draw_box((0.0, 0.0, -0.12), (0.082, 0.082, 0.24), (0.20, 0.20, 0.24, 1.0))
		draw_box((0.0, 0.0, -0.275), (0.076, 0.076, 0.12), (0.28, 0.28, 0.33, 1.0))
		draw_box((0.0, 0.0, -0.355), (0.090, 0.090, 0.028), (0.44, 0.44, 0.50, 1.0))
		draw_box((0.031, 0.018, -0.19), (0.012, 0.012, 0.028), (0.95, 0.24, 0.24, 1.0))
		draw_box((0.030, -0.018, -0.24), (0.012, 0.012, 0.022), (0.28, 0.80, 1.0, 1.0))

		if blade_power > LASER_MIN_VISIBLE_POWER:
			blade_len = 1.34 * blade_power
			blade_center_z = -0.36 - (blade_len * 0.5)
			glow_alpha = 0.17 + (0.20 * blade_power)

			glDepthMask(False)
			glBlendFunc(GL_SRC_ALPHA, GL_ONE)
			draw_box(
				(0.0, 0.0, blade_center_z),
				(0.092, 0.092, blade_len),
				(*outer_color, glow_alpha),
			)

			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
			draw_box(
				(0.0, 0.0, blade_center_z),
				(0.036, 0.036, blade_len),
				(*core_color, 0.98),
			)
			glDepthMask(True)

		glPopMatrix()

	def _get_blade_segment_from_pose(self, position: np.ndarray, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		base_local = np.array([0.0, 0.0, -0.36], dtype=np.float32)
		tip_local = np.array([0.0, 0.0, -1.66], dtype=np.float32)
		base = position + (rotation @ base_local)
		tip = position + (rotation @ tip_local)
		return base.astype(np.float32), tip.astype(np.float32)

	def _create_enemy_saber(self) -> EnemyCombatSaber:
		guard_position = np.array([0.86, -0.18, -2.42], dtype=np.float32)
		guard_direction = normalize(
			np.array([-0.22, 0.96, 0.12], dtype=np.float32),
			np.array([0.0, 1.0, 0.1], dtype=np.float32),
		)
		return EnemyCombatSaber(
			guard_position=guard_position.copy(),
			guard_direction=guard_direction.copy(),
			position=guard_position.copy(),
			previous_position=guard_position.copy(),
			blade_direction=guard_direction.copy(),
			previous_blade_direction=guard_direction.copy(),
			pose_from_position=guard_position.copy(),
			pose_to_position=guard_position.copy(),
			pose_from_direction=guard_direction.copy(),
			pose_to_direction=guard_direction.copy(),
			strike_target_position=guard_position.copy(),
			strike_target_direction=guard_direction.copy(),
			idle_wait_s=COMBAT_ATTACK_INITIAL_DELAY_S,
		)

	def _combat_attack_patterns(self) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...]:
		return (
			(
				np.array([0.94, -0.06, -2.64], dtype=np.float32),
				np.array([-0.54, 0.80, 0.26], dtype=np.float32),
				np.array([0.22, -0.18, -2.02], dtype=np.float32),
				np.array([-0.12, 0.90, 0.42], dtype=np.float32),
			),
			(
				np.array([-0.16, -0.10, -2.58], dtype=np.float32),
				np.array([0.70, 0.68, 0.22], dtype=np.float32),
				np.array([0.36, -0.22, -2.00], dtype=np.float32),
				np.array([0.24, 0.90, 0.36], dtype=np.float32),
			),
			(
				np.array([0.10, 0.04, -2.60], dtype=np.float32),
				np.array([0.04, 0.98, 0.14], dtype=np.float32),
				np.array([0.04, -0.24, -1.94], dtype=np.float32),
				np.array([0.04, 0.86, 0.50], dtype=np.float32),
			),
			(
				np.array([0.72, -0.20, -2.50], dtype=np.float32),
				np.array([-0.12, 0.98, 0.10], dtype=np.float32),
				np.array([-0.10, -0.10, -2.06], dtype=np.float32),
				np.array([-0.42, 0.84, 0.34], dtype=np.float32),
			),
		)

	def _set_enemy_saber_pose(
		self,
		enemy_saber: EnemyCombatSaber,
		target_position: np.ndarray,
		target_direction: np.ndarray,
		phase: str,
		duration_s: float,
	):
		enemy_saber.pose_from_position = enemy_saber.position.copy()
		enemy_saber.pose_to_position = target_position.astype(np.float32).copy()
		enemy_saber.pose_from_direction = enemy_saber.blade_direction.copy()
		enemy_saber.pose_to_direction = normalize(
			target_direction.astype(np.float32),
			enemy_saber.guard_direction,
		)
		enemy_saber.phase = phase
		enemy_saber.phase_elapsed_s = 0.0
		enemy_saber.phase_duration_s = float(max(0.01, duration_s))

	def _begin_enemy_idle(self, enemy_saber: EnemyCombatSaber, delay_s: Optional[float] = None):
		enemy_saber.phase = "idle"
		enemy_saber.phase_elapsed_s = 0.0
		enemy_saber.phase_duration_s = 0.0
		enemy_saber.idle_wait_s = float(
			max(
				0.0,
				delay_s if delay_s is not None else random.uniform(COMBAT_IDLE_INTERVAL_MIN_S, COMBAT_IDLE_INTERVAL_MAX_S),
			)
		)
		enemy_saber.pose_from_position = enemy_saber.position.copy()
		enemy_saber.pose_to_position = enemy_saber.guard_position.copy()
		enemy_saber.pose_from_direction = enemy_saber.blade_direction.copy()
		enemy_saber.pose_to_direction = enemy_saber.guard_direction.copy()

	def _begin_enemy_attack(self, enemy_saber: EnemyCombatSaber):
		windup_position, windup_direction, strike_position, strike_direction = random.choice(self._combat_attack_patterns())
		enemy_saber.strike_target_position = strike_position.astype(np.float32).copy()
		enemy_saber.strike_target_direction = normalize(
			strike_direction.astype(np.float32),
			enemy_saber.guard_direction,
		)
		self._set_enemy_saber_pose(
			enemy_saber,
			windup_position,
			windup_direction,
			phase="windup",
			duration_s=COMBAT_WINDUP_DURATION_S * random.uniform(0.94, 1.08),
		)

	def _get_enemy_saber_segment(self, enemy_saber: EnemyCombatSaber, previous: bool = False) -> tuple[np.ndarray, np.ndarray]:
		position = enemy_saber.previous_position if previous else enemy_saber.position
		direction = enemy_saber.previous_blade_direction if previous else enemy_saber.blade_direction
		rotation = rotation_matrix_from_blade_direction(direction)
		return self._get_blade_segment_from_pose(position, rotation)

	def _try_parry_enemy_saber(self, enemy_saber: EnemyCombatSaber) -> bool:
		if self.blade_power <= 0.2:
			return False

		enemy_prev_base, enemy_prev_tip = self._get_enemy_saber_segment(enemy_saber, previous=True)
		enemy_base, enemy_tip = self._get_enemy_saber_segment(enemy_saber, previous=False)
		best_distance = float("inf")
		for point_start, point_end in (
			(enemy_prev_base, enemy_base),
			(enemy_saber.previous_position, enemy_saber.position),
			(enemy_prev_tip, enemy_tip),
		):
			best_distance = min(
				best_distance,
				min_distance_point_to_moving_segment(
					point_start,
					point_end,
					self.prev_saber_base_pos,
					self.prev_saber_tip_pos,
					self.saber_base_pos,
					self.saber_tip_pos,
				),
			)

		saber_speed = float(np.linalg.norm(self.saber_tip_velocity))
		if best_distance > COMBAT_PARRY_DISTANCE:
			return False

		play_parry_sound()
		self.score += 2
		return True

	def _spawn_block(self, initial_z: float = -7.5):
		lane_x = random.choice(self.block_lanes_x)
		lane_y = random.choice(self.block_lanes_y)
		shape_key = random.choice(["cube", "bar", "diamond"])
		if shape_key == "cube":
			size = np.array([0.24, 0.24, 0.24], dtype=np.float32)
		elif shape_key == "bar":
			size = np.array([0.18, 0.34, 0.18], dtype=np.float32)
		else:
			size = np.array([0.28, 0.32, 0.28], dtype=np.float32)
		position = np.array([lane_x, lane_y, initial_z], dtype=np.float32)
		velocity = np.array([0.0, 0.0, self.block_speed], dtype=np.float32)
		color_rgba = random.choice(self.block_palette)
		self.blocks.append(BeatBlock(position=position, velocity=velocity, shape_key=shape_key, size=size, color_rgba=color_rgba))

	def _spawn_projectile(self, initial_z: float = -7.8):
		lane_x = random.choice(self.block_lanes_x)
		lane_y = random.choice(self.block_lanes_y)
		position = np.array([lane_x, lane_y, initial_z], dtype=np.float32)
		velocity = np.array([0.0, 0.0, PROJECTILE_SPEED], dtype=np.float32)
		color = (1.0, 0.12, 0.10, 0.98)
		self.projectiles.append(
			Projectile(
				position=position,
				velocity=velocity,
				radius=PROJECTILE_RADIUS,
				color_rgba=color,
			)
		)

	def _spawn_cut_fragments(self, block: BeatBlock, cut_dir_2d: np.ndarray):
		cut_dir = normalize(np.array([cut_dir_2d[0], cut_dir_2d[1], 0.0], dtype=np.float32), np.array([1.0, 0.0, 0.0], dtype=np.float32))
		separation = normalize(np.array([cut_dir[1], -cut_dir[0], 0.0], dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32))
		size = block.size.astype(np.float32)
		if abs(separation[0]) >= abs(separation[1]):
			fragment_size = np.array([size[0] * 0.48, size[1], size[2]], dtype=np.float32)
		else:
			fragment_size = np.array([size[0], size[1] * 0.48, size[2]], dtype=np.float32)

		base_speed = 0.85 + min(1.8, float(np.linalg.norm(self.saber_tip_velocity)) * 0.25)
		for sign in (-1.0, 1.0):
			offset = separation * sign * (float(np.max(block.size)) * 0.16)
			velocity = (
				block.velocity
				+ (cut_dir * 0.35)
				+ (separation * sign * base_speed)
				+ np.array([0.0, 0.35, 0.0], dtype=np.float32)
			)
			angular_velocity = np.array(
				[
					random.uniform(-3.8, 3.8),
					random.uniform(-2.6, 2.6),
					random.uniform(-4.4, 4.4),
				],
				dtype=np.float32,
			)
			self.fragments.append(
				BlockFragment(
					position=(block.position + offset).astype(np.float32),
					velocity=velocity.astype(np.float32),
					size=fragment_size.astype(np.float32),
					shape_key=block.shape_key,
					rotation=block.rotation.copy(),
					angular_velocity=angular_velocity,
					lifetime_s=1.15,
					color_rgba=block.color_rgba,
				)
			)

	def _try_cut_block(self, block: BeatBlock, previous_position: np.ndarray) -> bool:
		if self.blade_power <= 0.35:
			return False

		block_radius = (float(np.linalg.norm(block.size)) * 0.5) + SABER_CUT_RADIUS_BONUS
		cut_distance = min_distance_point_to_moving_segment(
			previous_position,
			block.position,
			self.prev_saber_base_pos,
			self.prev_saber_tip_pos,
			self.saber_base_pos,
			self.saber_tip_pos,
		)
		center_distance = min_distance_point_to_moving_segment(
			previous_position,
			block.position,
			self.prev_saber_center_pos,
			self.prev_saber_tip_pos,
			self.saber_center_pos,
			self.saber_tip_pos,
		)
		threshold = block_radius + SABER_BLADE_BLOCK_RADIUS
		if min(cut_distance, center_distance) > threshold:
			return False

		swing_2d = self.saber_tip_velocity[:2]
		block.was_cut = True
		play_cut_sound()
		self.score += 1
		if float(np.linalg.norm(swing_2d)) <= 1e-6:
			cut_dir_2d = np.array([1.0, 0.0], dtype=np.float32)
		else:
			cut_dir_2d = normalize(swing_2d, np.array([1.0, 0.0], dtype=np.float32))
		self._spawn_cut_fragments(block, cut_dir_2d)
		return True

	def _update_blocks(self, dt: float):
		if self.game_mode != GAME_MODE_BLOCKS or not self.blocks_started:
			return

		self.block_spawn_timer_s += dt
		while self.block_spawn_timer_s >= self.block_spawn_interval_s:
			self.block_spawn_timer_s -= self.block_spawn_interval_s
			self._spawn_block()

		active_blocks = []
		for block in self.blocks:
			previous_position = block.position.copy()
			block.position = (block.position + (block.velocity * dt)).astype(np.float32)
			if not block.was_cut:
				self._try_cut_block(block, previous_position)

			if block.was_cut:
				continue
			if block.position[2] > 0.55:
				self.misses += 1
				block.passed_player = True
				continue
			active_blocks.append(block)
		self.blocks = active_blocks

	def _update_fragments(self, dt: float):
		gravity = np.array([0.0, -3.2, 0.0], dtype=np.float32)
		active_fragments = []
		for fragment in self.fragments:
			fragment.velocity = (fragment.velocity + (gravity * dt)).astype(np.float32)
			fragment.position = (fragment.position + (fragment.velocity * dt)).astype(np.float32)
			rotation_step = cv2.Rodrigues((fragment.angular_velocity * dt).astype(np.float32))[0].astype(np.float32)
			fragment.rotation = orthonormalize_rotation(rotation_step @ fragment.rotation)
			fragment.lifetime_s -= dt
			if fragment.lifetime_s > 0.0 and fragment.position[1] > -1.4:
				active_fragments.append(fragment)
		self.fragments = active_fragments

	def _try_block_projectile(self, projectile: Projectile, previous_position: np.ndarray) -> bool:
		if self.blade_power <= 0.2:
			return False

		distance = min_distance_point_to_moving_segment(
			previous_position,
			projectile.position,
			self.prev_saber_base_pos,
			self.prev_saber_tip_pos,
			self.saber_base_pos,
			self.saber_tip_pos,
		)
		if distance > (float(projectile.radius) + SABER_BLADE_BLOCK_RADIUS):
			return False

		swing_dir = normalize(
			self.saber_tip_velocity + np.array([0.0, 0.0, -0.6], dtype=np.float32),
			np.array([0.0, 0.0, -1.0], dtype=np.float32),
		)
		if swing_dir[2] > -0.2:
			swing_dir[2] = -0.6
			swing_dir = normalize(swing_dir, np.array([0.0, 0.0, -1.0], dtype=np.float32))

		projectile.velocity = (swing_dir * PROJECTILE_REBOUND_SPEED).astype(np.float32)
		projectile.position = (projectile.position + (swing_dir * 0.12)).astype(np.float32)
		projectile.deflected = True
		projectile.lifetime_s = 1.6
		play_projectile_reflect_sound()
		self.score += 1
		return True

	def _update_projectiles(self, dt: float):
		if self.game_mode != GAME_MODE_BLOCKS:
			return

		if self.blocks_started:
			self.projectile_spawn_timer_s += dt
			while self.projectile_spawn_timer_s >= self.projectile_spawn_interval_s:
				self.projectile_spawn_timer_s -= self.projectile_spawn_interval_s
				self._spawn_projectile()

		active_projectiles = []
		for projectile in self.projectiles:
			previous_position = projectile.position.copy()
			projectile.position = (projectile.position + (projectile.velocity * dt)).astype(np.float32)
			projectile.lifetime_s -= dt

			if not projectile.deflected:
				self._try_block_projectile(projectile, previous_position)

			if projectile.lifetime_s <= 0.0:
				continue
			if projectile.deflected:
				if projectile.position[2] > 0.9 or projectile.position[2] < -10.5 or abs(projectile.position[0]) > 3.0 or abs(projectile.position[1]) > 2.4:
					continue
			else:
				if projectile.position[2] > 0.38:
					self.misses += 1
					self._lose_life(1, "disparo")
					continue
			active_projectiles.append(projectile)
		self.projectiles = active_projectiles

	def _update_combat_attacks(self, dt: float):
		if self.game_mode != GAME_MODE_COMBAT or self.game_over or self.enemy_saber is None:
			return

		enemy_saber = self.enemy_saber
		enemy_saber.previous_position = enemy_saber.position.copy()
		enemy_saber.previous_blade_direction = enemy_saber.blade_direction.copy()

		if enemy_saber.phase == "idle":
			enemy_saber.phase_elapsed_s += dt
			sway = np.sin(enemy_saber.phase_elapsed_s * 2.8) * COMBAT_IDLE_SWAY_AMOUNT
			lift = np.sin(enemy_saber.phase_elapsed_s * 5.4) * COMBAT_IDLE_LIFT_AMOUNT
			enemy_saber.position = (
				enemy_saber.guard_position
				+ np.array([sway * 0.30, lift, 0.0], dtype=np.float32)
			).astype(np.float32)
			enemy_saber.blade_direction = normalize(
				enemy_saber.guard_direction + np.array([-sway * 0.18, lift * 0.20, -0.02], dtype=np.float32),
				enemy_saber.guard_direction,
			)
			if enemy_saber.phase_elapsed_s >= enemy_saber.idle_wait_s:
				self._begin_enemy_attack(enemy_saber)
			return

		enemy_saber.phase_elapsed_s += dt
		progress = clamp(enemy_saber.phase_elapsed_s / max(enemy_saber.phase_duration_s, 1e-6), 0.0, 1.0)
		shaped_progress = smoothstep01(progress)
		enemy_saber.position = lerp_vec(enemy_saber.pose_from_position, enemy_saber.pose_to_position, shaped_progress)
		enemy_saber.blade_direction = normalize(
			lerp_vec(enemy_saber.pose_from_direction, enemy_saber.pose_to_direction, shaped_progress),
			enemy_saber.pose_to_direction,
		)

		if enemy_saber.phase == "strike" and self._try_parry_enemy_saber(enemy_saber):
			self._set_enemy_saber_pose(
				enemy_saber,
				enemy_saber.guard_position.copy(),
				enemy_saber.guard_direction.copy(),
				phase="recover",
				duration_s=COMBAT_RECOVER_DURATION_S * random.uniform(0.94, 1.06),
			)
			return

		if enemy_saber.phase == "strike" and (progress >= 1.0 or enemy_saber.position[2] >= COMBAT_ATTACK_HIT_Z):
			self.misses += 1
			self._lose_life(1, "parry fallido")
			if self.game_over:
				return
			self._set_enemy_saber_pose(
				enemy_saber,
				enemy_saber.guard_position.copy(),
				enemy_saber.guard_direction.copy(),
				phase="recover",
				duration_s=COMBAT_RECOVER_DURATION_S * random.uniform(0.94, 1.08),
			)
			return

		if progress < 1.0:
			return

		if enemy_saber.phase == "windup":
			self._set_enemy_saber_pose(
				enemy_saber,
				enemy_saber.strike_target_position.copy(),
				enemy_saber.strike_target_direction.copy(),
				phase="strike",
				duration_s=COMBAT_STRIKE_DURATION_S * random.uniform(0.96, 1.06),
			)
		elif enemy_saber.phase == "recover":
			self._begin_enemy_idle(enemy_saber)

	def _draw_block(self, block: BeatBlock):
		if block.shape_key == "diamond":
			draw_oriented_diamond(block.position, block.rotation, block.size, block.color_rgba)
			return
		draw_oriented_box(block.position, block.rotation, block.size, block.color_rgba)

	def _draw_fragments(self):
		for fragment in self.fragments:
			alpha = clamp(fragment.lifetime_s / 1.15, 0.0, 1.0)
			color = (
				fragment.color_rgba[0],
				fragment.color_rgba[1],
				fragment.color_rgba[2],
				fragment.color_rgba[3] * alpha,
			)
			if fragment.shape_key == "diamond":
				draw_oriented_diamond(fragment.position, fragment.rotation, fragment.size, color)
			else:
				draw_oriented_box(fragment.position, fragment.rotation, fragment.size, color)

	def _draw_projectiles(self):
		for projectile in self.projectiles:
			alpha = clamp(projectile.lifetime_s / (1.6 if projectile.deflected else 3.0), 0.2, 1.0)
			forward = normalize(projectile.velocity, np.array([0.0, 0.0, 1.0], dtype=np.float32))
			rotation = rotation_matrix_from_forward(forward)
			laser_len = PROJECTILE_LASER_LENGTH * (1.0 if not projectile.deflected else 0.82)
			glow_radius = float(projectile.radius) * (2.0 if not projectile.deflected else 1.55)
			core_radius = float(projectile.radius) * 0.72

			draw_oriented_box(
				projectile.position,
				rotation,
				np.array([glow_radius, glow_radius, laser_len], dtype=np.float32),
				(1.0, 0.08, 0.06, 0.24 * alpha),
			)
			draw_oriented_box(
				projectile.position,
				rotation,
				np.array([core_radius, core_radius, laser_len * 0.92], dtype=np.float32),
				(1.0, 0.22, 0.18, 0.98 * alpha),
			)
			draw_oriented_box(
				projectile.position + (forward * (laser_len * 0.22)).astype(np.float32),
				rotation,
				np.array([core_radius * 0.72, core_radius * 0.72, laser_len * 0.24], dtype=np.float32),
				(1.0, 0.88, 0.84, 0.95 * alpha),
			)

	def _draw_combat_attacks(self):
		if self.game_mode != GAME_MODE_COMBAT or self.enemy_saber is None:
			return

		enemy_saber = self.enemy_saber
		rotation = rotation_matrix_from_blade_direction(enemy_saber.blade_direction)
		self._draw_lightsaber_model(
			enemy_saber.position,
			rotation,
			0.94 if enemy_saber.phase == "idle" else 1.0,
			SABER_COLOR_PRESETS["r"]["outer"],
			SABER_COLOR_PRESETS["r"]["core"],
		)

	def _draw_hud(self, view_width: Optional[int] = None, view_height: Optional[int] = None):
		safe_w = int(view_width) if view_width is not None else int(self.width)
		safe_h = int(view_height) if view_height is not None else int(self.height)
		safe_w = max(1, safe_w)
		safe_h = max(1, safe_h)
		glMatrixMode(GL_PROJECTION)
		glPushMatrix()
		glLoadIdentity()
		glOrtho(0.0, float(safe_w), float(safe_h), 0.0, -1.0, 1.0)
		glMatrixMode(GL_MODELVIEW)
		glPushMatrix()
		glLoadIdentity()
		glDisable(GL_DEPTH_TEST)

		bar_x = 28.0
		bar_y = 24.0
		segment_w = 22.0
		segment_h = 18.0
		segment_gap = 6.0
		bar_w = (self.max_lives * segment_w) + ((self.max_lives - 1) * segment_gap) + 16.0
		bar_h = segment_h + 16.0

		draw_screen_rect(bar_x - 8.0, bar_y - 8.0, bar_w, bar_h, (0.02, 0.03, 0.05, 0.42))
		draw_screen_rect(bar_x - 8.0, bar_y - 8.0, bar_w, bar_h, (0.72, 0.18, 0.18, 0.85), outline=True)
		for idx in range(self.max_lives):
			seg_x = bar_x + (idx * (segment_w + segment_gap))
			is_full = idx < self.lives
			fill_color = (0.96, 0.18, 0.18, 0.95) if is_full else (0.20, 0.06, 0.06, 0.55)
			outline_color = (1.0, 0.42, 0.42, 0.92) if is_full else (0.42, 0.16, 0.16, 0.78)
			draw_screen_rect(seg_x, bar_y, segment_w, segment_h, fill_color)
			draw_screen_rect(seg_x, bar_y, segment_w, segment_h, outline_color, outline=True)

		score_text = max(0, self.score)
		digit_w = 20.0
		digit_h = 34.0
		score_digits = len(str(score_text))
		total_score_w = (score_digits * digit_w) + (max(0, score_digits - 1) * digit_w * 0.22)
		score_x = float(safe_w) - total_score_w - 32.0
		score_y = 18.0
		draw_screen_rect(score_x - 14.0, score_y - 8.0, total_score_w + 28.0, digit_h + 16.0, (0.02, 0.03, 0.05, 0.42))
		draw_number_7seg(score_x, score_y, digit_w, digit_h, score_text, (0.95, 0.86, 0.30, 0.98))

		if self.game_over:
			panel_w = 220.0
			panel_h = 44.0
			panel_x = (float(safe_w) - panel_w) * 0.5
			panel_y = 26.0
			draw_screen_rect(panel_x, panel_y, panel_w, panel_h, (0.18, 0.02, 0.02, 0.72))
			draw_screen_rect(panel_x, panel_y, panel_w, panel_h, (0.95, 0.24, 0.24, 0.9), outline=True)
			draw_number_7seg(panel_x + 82.0, panel_y + 6.0, 18.0, 30.0, 0, (1.0, 0.24, 0.20, 0.98))

		damage_flash_remaining = self.damage_flash_until_s - time.perf_counter()
		if damage_flash_remaining > 0.0:
			flash_alpha = clamp(damage_flash_remaining / LIFE_LOST_FLASH_DURATION_S, 0.0, 1.0)
			draw_screen_rect(
				0.0,
				0.0,
				float(safe_w),
				float(safe_h),
				(0.92, 0.05, 0.05, 0.26 * flash_alpha),
			)

		glEnable(GL_DEPTH_TEST)
		glPopMatrix()
		glMatrixMode(GL_PROJECTION)
		glPopMatrix()
		glMatrixMode(GL_MODELVIEW)

	def _update_blade_power(self, dt: float):
		if self.blade_target_on:
			self.blade_power = min(1.0, self.blade_power + (dt / max(LASER_ON_DURATION_S, 1e-6)))
		else:
			self.blade_power = max(0.0, self.blade_power - (dt / max(LASER_OFF_DURATION_S, 1e-6)))

	def _draw_saber(self):
		color_data = SABER_COLOR_PRESETS[self.saber_color_key]
		self._draw_lightsaber_model(
			self.current_pos,
			self.current_rot,
			self.blade_power,
			color_data["outer"],
			color_data["core"],
		)

	def _render_eye(self, camera_x: float, camera_y: float, camera_z: float, view_width: int, view_height: int):
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()

		forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
		up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
		if self.vr_mode and self.phone_has_rotation and self.phone_camera_rot is not None:
			forward = (self.phone_camera_rot @ forward).astype(np.float32)
			up = (self.phone_camera_rot @ up).astype(np.float32)

		# Camara de primera persona
		gluLookAt(
			camera_x,
			camera_y,
			camera_z,
			camera_x + float(forward[0]),
			camera_y + float(forward[1]),
			camera_z + float(forward[2]),
			float(up[0]),
			float(up[1]),
			float(up[2]),
		)

		draw_floor_grid()
		draw_rgb_corridor(time.perf_counter())
		for block in self.blocks:
			self._draw_block(block)
		self._draw_projectiles()
		self._draw_combat_attacks()
		self._draw_fragments()
		self._draw_saber()
		self._draw_hud(view_width=view_width, view_height=view_height)

	def _set_projection_for_view(self, view_width: int, view_height: int):
		safe_w = max(1, int(view_width))
		safe_h = max(1, int(view_height))
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		gluPerspective(88.0, safe_w / safe_h, 0.01, 100.0)
		glMatrixMode(GL_MODELVIEW)

	def _render(self):
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

		base_camera_x = 0.0
		base_camera_y = -0.08
		base_camera_z = 0.18
		now_s = time.perf_counter()
		damage_shake_remaining = self.damage_shake_until_s - now_s
		if damage_shake_remaining > 0.0:
			shake_alpha = clamp(damage_shake_remaining / LIFE_LOST_SHAKE_DURATION_S, 0.0, 1.0)
			base_camera_x += random.uniform(-LIFE_LOST_SHAKE_OFFSET, LIFE_LOST_SHAKE_OFFSET) * shake_alpha
			base_camera_y += random.uniform(-LIFE_LOST_SHAKE_OFFSET, LIFE_LOST_SHAKE_OFFSET) * shake_alpha
			base_camera_z += random.uniform(-LIFE_LOST_SHAKE_OFFSET * 0.35, LIFE_LOST_SHAKE_OFFSET * 0.35) * shake_alpha

		if not self.vr_mode:
			glViewport(0, 0, int(self.width), int(self.height))
			self._set_projection_for_view(int(self.width), int(self.height))
			self._render_eye(base_camera_x, base_camera_y, base_camera_z, int(self.width), int(self.height))
			pygame.display.flip()
			return

		# VR: dos ojos en pantalla partida (lado a lado)
		full_w = max(2, int(self.width))
		full_h = max(1, int(self.height))
		eye_w = max(1, full_w // 2)
		eye_h = full_h
		eye_offset = float(self.vr_eye_separation) * 0.5

		# Ojo izquierdo
		glViewport(0, 0, eye_w, eye_h)
		self._set_projection_for_view(eye_w, eye_h)
		self._render_eye(base_camera_x - eye_offset, base_camera_y, base_camera_z, eye_w, eye_h)

		# Ojo derecho
		glViewport(eye_w, 0, eye_w, eye_h)
		self._set_projection_for_view(eye_w, eye_h)
		self._render_eye(base_camera_x + eye_offset, base_camera_y, base_camera_z, eye_w, eye_h)

		pygame.display.flip()

	def _on_keydown(self, key: int):
		if key == K_ESCAPE:
			self.running = False
			return

		if key == K_SPACE:
			self._set_blade_target(not self.blade_target_on, source="keyboard")
			return

		if key == K_i:
			self._restart_current_mode()
			mode_name = "combate" if self.game_mode == GAME_MODE_COMBAT else "cubos"
			print(f"[Game] Inicio/reinicio por teclado -> modo {mode_name}")
			return

		if key == K_c:
			self._reset_combat_mode()
			print("[Game] Modo combate por teclado")
			return

		if key == K_m:
			self.control_mirror_x = not self.control_mirror_x
			print(f"[Control] Mirror X {'ON' if self.control_mirror_x else 'OFF'}")
			return

		if key == K_v:
			self._toggle_vr_mode(source="keyboard")
			return

		if key == K_z:
			self.reference_translation_cv = None
			self.reference_rotation_cv = None
			self.filtered_pos = None
			self.last_local_offset = None
			self.local_velocity[:] = 0.0
			self.has_pose = False
			self.last_pose_time_s = 0.0
			self._last_pose_state = "waiting"
			self.phone_reference_rot = None
			self.phone_camera_rot = np.eye(3, dtype=np.float32)
			self.phone_has_rotation = False
			print("[Tracking] Recentrado solicitado")
			return

		if key in (K_r, K_g, K_b):
			pressed = chr(key)
			if pressed in SABER_COLOR_PRESETS:
				self.saber_color_key = pressed
				print(f"[Saber] Color: {SABER_COLOR_PRESETS[pressed]['name']}")

	def run(self):
		print("[OpenGL] ESC: salir | Espacio: hoja on/off | R/G/B: color | I: iniciar/reiniciar | C: combate | M: mirror X | V: VR | Z: recentrar")
		print("[OpenGL] Voz: 'sable' o 'saber', 'combate', 'cubos', 'reiniciar', 'rojo', 'verde', 'azul', 'invertir' si Vosk esta disponible")
		print("[OpenGL] Voz extra: di 'realidad virtual' para VR")
		print("[Gameplay] Di 'cubos' para el modo de bloques o 'combate' para el modo parry con sable enemigo")
		print("[Tracking] Modo autonomo: lectura ArUco directa desde esta camara")

		try:
			while self.running:
				dt = float(np.clip(self.clock.tick(self.max_fps) / 1000.0, 0.0, 0.1))
				self.prev_saber_base_pos = self.saber_base_pos.copy()
				self.prev_saber_tip_pos = self.saber_tip_pos.copy()
				self.prev_saber_center_pos = self.saber_center_pos.copy()

				for event in pygame.event.get():
					if event.type == QUIT:
						self.running = False
					elif event.type == VIDEORESIZE:
						self._create_window(event.w, event.h)
						self._set_projection(event.w, event.h)
					elif event.type == KEYDOWN:
						self._on_keydown(event.key)

				self._process_voice_commands()
				self._process_phone_rotation()
				self._update_audio_loop()
				self._update_pose()
				self._update_blade_power(dt)
				self._compute_saber_points()
				self.saber_tip_velocity = ((self.saber_tip_pos - self.prev_saber_tip_pos) / max(dt, 1e-4)).astype(np.float32)
				self._update_blocks(dt)
				self._update_projectiles(dt)
				self._update_combat_attacks(dt)
				self._update_fragments(dt)
				self._render()
		finally:
			self._shutdown_runtime_services()
			pygame.quit()


def parse_args():
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


if __name__ == "__main__":
	sys.exit(main())
