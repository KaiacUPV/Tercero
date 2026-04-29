# Auto-extracted from openglsable.py without modifying the original file.
"""Aplicacion principal del juego 3D.

Coordina tracking ArUco, entrada de voz, orientacion del movil, audio, gameplay
(bloques/combate) y render OpenGL.
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

from .audio import *
from .constants import *
from .math_utils import *
from .models import *
from .phone import PhoneRotationListener
from .rendering import *
from .tracking import *
from .voice import VoiceCommandListener

class OpenGLSaberApp:
	"""Runtime del juego tipo Beat Saber con sable controlado por ArUco."""

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
		# Configuracion base recibida desde CLI.
		self.tracker = tracker
		self.width = int(width)
		self.height = int(height)
		self.pose_scale = float(pose_scale)
		self.pos_smoothing = float(np.clip(pos_smoothing, 0.0, 1.0))
		self.lost_timeout_s = float(max(0.05, lost_timeout_s))
		self.max_fps = int(max(30, max_fps))
		self.vsync = bool(vsync)

		# Estado de tracking/control: convierte la pose del cubo en movimiento local.
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

		# Estado visual/sonoro de la hoja.
		self.saber_color_key = "b"
		self.blade_target_on = False
		self.blade_power = 0.0
		project_root = Path(__file__).resolve().parent.parent
		audio_dir = project_root / "audios"
		self.saber_on_sound_path = audio_dir / SABER_ON_SOUND_FILE
		self.saber_loop_sound_path = audio_dir / SABER_LOOP_SOUND_FILE
		self.saber_off_sound_path = audio_dir / SABER_OFF_SOUND_FILE
		self.projectile_reflect_sound_path = audio_dir / PROJECTILE_REFLECT_SOUND_FILE
		self.life_lost_sound_path = audio_dir / LIFE_LOST_SOUND_FILE
		self.cut_sound_path = audio_dir / CUT_SOUND_FILE
		self.parry_sound_path = audio_dir / PARRY_SOUND_FILE
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
			"recenter": 0.0,
			"start_blocks_mode": 0.0,
			"start_combat": 0.0,
			"restart_mode": 0.0,
			"toggle_vr": 0.0,
		}

		# VR renderiza dos ojos lado a lado; la rotacion del movil es opcional.
		self.vr_mode = False
		self.vr_eye_separation = float(VR_EYE_SEPARATION)

		# Estado de gameplay: modo bloques por defecto, combate bajo demanda.
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
		"""Crea/recrea la ventana pygame en modo OpenGL."""
		flags = DOUBLEBUF | OPENGL | FULLSCREEN
		try:
			surface = pygame.display.set_mode((0, 0), flags, vsync=1 if self.vsync else 0)
		except TypeError:
			surface = pygame.display.set_mode((0, 0), flags)
		self.width, self.height = surface.get_size()

	def _init_gl_state(self):
		"""Estado OpenGL global: profundidad, blending y color de fondo."""
		glEnable(GL_DEPTH_TEST)
		glDepthFunc(GL_LEQUAL)
		glEnable(GL_BLEND)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
		glClearColor(0.04, 0.05, 0.08, 1.0)
		self._set_projection(self.width, self.height)

	def _set_projection(self, width: int, height: int):
		"""Configura la proyeccion perspectiva para el tamano de ventana."""
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
		"""Arranca Vosk si esta activado y disponible."""
		if self.audio_backend == "none":
			print("[Audio] Sin backend. Instala PyOpenAL (pip install PyOpenAL) o pygame.")
		else:
			print(f"[Audio] Backend activo: {self.audio_backend}")

		if not VOICE_CONTROL_ENABLED:
			return

		project_root = Path(__file__).resolve().parent.parent
		voice_model_path = project_root / VOICE_MODEL_DIR
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
			print("[Voice] Control activo: di 'sable', 'combate', 'cubos', 'reset', 'rojo/verde/azul' o 'invertir'.")
		else:
			print(self.voice_listener.error_message)
			self.voice_listener = None

	def _init_phone_rotation_control(self):
		"""Abre el receptor UDP para usar orientacion del movil en modo VR."""
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
		"""Actualiza la rotacion de camara a partir del ultimo mensaje UDP."""
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
		"""Cambia el estado objetivo de la hoja y sincroniza efectos de audio."""
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
		"""Resta vidas y dispara feedback visual/sonoro de impacto."""
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
		"""Elimina entidades temporales al reiniciar/cambiar de modo."""
		self.blocks.clear()
		self.projectiles.clear()
		self.fragments.clear()
		self.enemy_saber = None

	def _reset_common_game_state(self):
		"""Reinicia puntuacion, vidas y efectos comunes a todos los modos."""
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
		"""Activa el modo de bloques/cortes."""
		self._reset_common_game_state()
		self.game_mode = GAME_MODE_BLOCKS
		self.blocks_started = True
		self._spawn_block(initial_z=-4.2)
		self._spawn_block(initial_z=-5.6)
		print("[Game] Modo cubos reiniciado")

	def _reset_combat_mode(self):
		"""Activa el modo combate/parry contra sable enemigo."""
		self._reset_common_game_state()
		self.game_mode = GAME_MODE_COMBAT
		self.blocks_started = False
		self.enemy_saber = self._create_enemy_saber()
		print("[Game] Modo combate reiniciado")

	def _restart_current_mode(self):
		"""Reinicia el modo que este activo sin cambiar de tipo de juego."""
		if self.game_mode == GAME_MODE_COMBAT:
			self._reset_combat_mode()
		else:
			self._reset_block_mode()

	def _sync_audio_with_current_blade_state(self):
		"""Alinea audio inicial con el estado actual de la hoja."""
		stop_all_sounds()
		self.loop_playing = False
		if self.blade_target_on:
			play_saber_on_sound()
			self.loop_start_due_time = time.perf_counter() + max(0.0, self.loop_start_delay_s)
		else:
			self.loop_start_due_time = None

	def _update_audio_loop(self):
		"""Arranca el loop de la hoja cuando termina el sonido de encendido."""
		now = time.perf_counter()
		if self.blade_target_on and not self.loop_playing and self.loop_start_due_time is not None and now >= self.loop_start_due_time:
			self.loop_playing = play_saber_loop_sound()

	def _process_voice_commands(self):
		"""Consume comandos de voz y los traduce a acciones de juego."""
		if self.voice_listener is None:
			return
		if self.voice_listener.error_message:
			print(self.voice_listener.error_message)
			self.voice_listener.stop()
			self.voice_listener = None
			return

		for voice_command, spoken_text in self.voice_listener.pop_commands():
			command_now = time.perf_counter()
			cooldown_s = 0.0 if voice_command == "recenter" else VOICE_COMMAND_COOLDOWN_S
			if (command_now - self.voice_last_command_time.get(voice_command, 0.0)) < cooldown_s:
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
			elif voice_command == "recenter":
				# Exactamente lo mismo que pulsar la tecla Z.
				self._on_keydown(K_z)
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
		"""Alterna render estereoscopico lado a lado."""
		new_state = not self.vr_mode
		self.vr_mode = new_state
		if new_state:
			# Al entrar en VR, re-usamos la orientacion actual del movil como referencia.
			self.phone_reference_rot = None
			self.phone_camera_rot = np.eye(3, dtype=np.float32)
			self.phone_has_rotation = False
		print(f"[VR] {'ON' if self.vr_mode else 'OFF'} ({source})")

	def _reset_tracking(self):
		"""Marca la pose actual como nueva referencia de movimiento."""
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
		print("[Tracking] Recentrado manual activado (Pose CV & Camara Movil)")

	def _shutdown_runtime_services(self):
		"""Cierra hilos y backends externos antes de salir."""
		if self.phone_listener is not None:
			self.phone_listener.stop()
			self.phone_listener = None
		if self.voice_listener is not None:
			self.voice_listener.stop()
			self.voice_listener = None
		shutdown_audio_backend()

	def _cv_local_offset_from_translation(self, translation_cv: np.ndarray) -> np.ndarray:
		"""Convierte traslacion OpenCV de camara a offset local de mano."""
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
		"""Convierte rotacion relativa OpenCV al sistema de coordenadas OpenGL."""
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
		"""Lee la ultima pose del tracker y actualiza posicion/rotacion del sable."""
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
		"""Calcula base, punta y centro de la hoja para render y colisiones."""
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
		"""Dibuja mango y hoja como primitivas OpenGL orientadas."""
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
		"""Devuelve segmento 3D de la hoja para una pose cualquiera."""
		base_local = np.array([0.0, 0.0, -0.36], dtype=np.float32)
		tip_local = np.array([0.0, 0.0, -1.66], dtype=np.float32)
		base = position + (rotation @ base_local)
		tip = position + (rotation @ tip_local)
		return base.astype(np.float32), tip.astype(np.float32)

	def _create_enemy_saber(self) -> EnemyCombatSaber:
		"""Crea el sable enemigo en posicion de guardia inicial."""
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
		"""Define poses de ataque para variar el modo combate."""
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
		"""Actualiza posicion/direccion del sable enemigo conservando pose previa."""
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
		"""Pasa el enemigo a espera antes del siguiente ataque."""
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
		"""Elige un patron y arranca la fase de carga del ataque."""
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
		"""Segmento de colision del sable enemigo, actual o previo."""
		position = enemy_saber.previous_position if previous else enemy_saber.position
		direction = enemy_saber.previous_blade_direction if previous else enemy_saber.blade_direction
		rotation = rotation_matrix_from_blade_direction(direction)
		return self._get_blade_segment_from_pose(position, rotation)

	def _try_parry_enemy_saber(self, enemy_saber: EnemyCombatSaber) -> bool:
		"""Detecta choque/parry entre la hoja del jugador y la enemiga."""
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
		"""Genera un bloque en un carril aleatorio."""
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
		"""Genera un proyectil que avanza hacia el jugador."""
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
		"""Crea fragmentos visuales cuando un bloque se corta."""
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
		"""Comprueba si el movimiento reciente de la hoja corta un bloque."""
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
		"""Avanza bloques, spawnea nuevos y cuenta cortes/fallos."""
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
		"""Actualiza fragmentos despues de cortes y elimina los caducados."""
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
		"""Intenta desviar un proyectil con el segmento barrido del sable."""
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
		"""Avanza proyectiles, rebotes e impactos al jugador."""
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
		"""Maquina de estados del sable enemigo en modo combate."""
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
		"""Dibuja un bloque segun su forma."""
		if block.shape_key == "diamond":
			draw_oriented_diamond(block.position, block.rotation, block.size, block.color_rgba)
			return
		draw_oriented_box(block.position, block.rotation, block.size, block.color_rgba)

	def _draw_fragments(self):
		"""Dibuja fragmentos activos de bloques cortados."""
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
		"""Dibuja proyectiles como haces luminosos."""
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
		"""Dibuja el sable enemigo si el modo combate esta activo."""
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
		"""Dibuja vidas, puntuacion, game over y flash de dano."""
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
		"""Interpola visualmente encendido/apagado de la hoja."""
		if self.blade_target_on:
			self.blade_power = min(1.0, self.blade_power + (dt / max(LASER_ON_DURATION_S, 1e-6)))
		else:
			self.blade_power = max(0.0, self.blade_power - (dt / max(LASER_OFF_DURATION_S, 1e-6)))

	def _draw_saber(self):
		"""Dibuja el sable del jugador en su pose actual."""
		color_data = SABER_COLOR_PRESETS[self.saber_color_key]
		self._draw_lightsaber_model(
			self.current_pos,
			self.current_rot,
			self.blade_power,
			color_data["outer"],
			color_data["core"],
		)

	def _render_eye(self, camera_x: float, camera_y: float, camera_z: float, view_width: int, view_height: int):
		"""Renderiza una vista completa desde una posicion de camara."""
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
		"""Configura perspectiva por ojo/vista."""
		safe_w = max(1, int(view_width))
		safe_h = max(1, int(view_height))
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		gluPerspective(88.0, safe_w / safe_h, 0.01, 100.0)
		glMatrixMode(GL_MODELVIEW)

	def _render(self):
		"""Render principal: una vista normal o dos vistas en modo VR."""
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
		"""Mapa de controles de teclado."""
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
			self._reset_tracking()
			return

		if key in (K_r, K_g, K_b):
			pressed = chr(key)
			if pressed in SABER_COLOR_PRESETS:
				self.saber_color_key = pressed
				print(f"[Saber] Color: {SABER_COLOR_PRESETS[pressed]['name']}")

	def run(self):
		"""Bucle principal de eventos, updates de gameplay y render."""
		print("[OpenGL] ESC: salir | Espacio: hoja on/off | R/G/B: color | I: iniciar/reiniciar | C: combate | M: mirror X | V: VR | Z: recentrar")
		print("[OpenGL] Voz: 'sable' o 'saber', 'combate', 'cubos', 'reset', 'rojo', 'verde', 'azul', 'invertir' si Vosk esta disponible")
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
