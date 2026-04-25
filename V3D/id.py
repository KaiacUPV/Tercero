import cv2
import cv2.aruco as aruco
import json
import numpy as np
import os
import queue
import random
import socket
import struct
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

# ===== Constantes =====
PRIMARY_MARKER = 17
CUBE_SIZE_M = 0.065
MARKER_SIZE = CUBE_SIZE_M
HALF_SIZE = CUBE_SIZE_M / 2.0
PRISM_LENGTH = 0.62
PARALLEL_SMOOTH_ZONE_DEG = 30.0
FRONTAL_LOCK_ZONE_DEG = 10.0
SMOOTH_TRANSLATION_ALPHA = 0.16
SMOOTH_ROTATION_ALPHA = 0.14
FRONTAL_LOCK_TRANSLATION_ALPHA = 0.07
FRONTAL_LOCK_ROTATION_ALPHA = 0.03
LASER_CYLINDER_SIDES = 24
LASER_OUTER_RADIUS_SCALE = 0.92
LASER_INNER_RADIUS_SCALE = 0.30
HANDLE_LENGTH_M = 0.145
HANDLE_RADIUS_SCALE = 1.06
HANDLE_CAP_LENGTH_M = 0.024
HANDLE_CAP_RADIUS_SCALE = 1.34
EMITTER_LENGTH_M = 0.022
EMITTER_RADIUS_SCALE = 0.94
HANDLE_PROFILE_SPLITS = (0.48, 0.76)
HANDLE_PROFILE_RADIUS_SCALES = (1.16, 1.00, 0.80, 0.92)
HANDLE_DETAIL_RING_COUNT = 5
HANDLE_DETAIL_RING_RADIUS_SCALE = 1.03
HANDLE_DETAIL_RING_LENGTH_M = 0.005
HANDLE_DETAIL_RING_START_RATIO = 0.16
HANDLE_DETAIL_RING_STEP_RATIO = 0.15
HANDLE_BUTTON_RADIUS_SCALE = 0.18
HANDLE_BUTTON_HEIGHT_M = 0.012
HANDLE_BUTTON_POSITIONS = (0.70, 0.56)
HANDLE_FIN_COUNT = 3
HANDLE_FIN_RADIUS_SCALE = 0.14
HANDLE_FIN_HEIGHT_M = 0.018
HANDLE_FIN_POSITIONS = (0.82, 0.90)
LASER_ON_DURATION_S = 0.34
LASER_OFF_DURATION_S = 0.26
LASER_MIN_VISIBLE_POWER = 0.001
LASER_MIN_DRAW_DEPTH_M = 0.06
MAX_PROJECTED_COORD_ABS_PX = 20000.0
KEY_TOGGLE_COOLDOWN_S = 0.18
SABER_ON_SOUND_FILE = "sable-on.wav"
SABER_LOOP_SOUND_FILE = "loop.wav"
SABER_OFF_SOUND_FILE = "sable-off.wav"
SABER_LOOP_START_DELAY_S = 0.09
CAMERA_HORIZONTAL_FLIP_DEFAULT = False
DEFAULT_SABER_COLOR_KEY = "r"
SABER_COLOR_PRESETS = {
    "r": {
        "name": "red",
        "outer_fill": (40, 45, 255),
        "outer_edge": (20, 30, 220),
        "inner_fill": (215, 225, 255),
        "hilt_section_edges": ((98, 98, 150), (78, 78, 130), (120, 120, 170)),
        "hilt_cap_edge": (145, 145, 188),
        "hilt_emitter_edge": (175, 175, 205),
        "ring_fill": (24, 24, 92),
        "ring_edge": (78, 78, 185),
        "button_colors": ((26, 26, 200), (24, 120, 215)),
        "fin_edge": (70, 70, 165),
    },
    "g": {
        "name": "green",
        "outer_fill": (65, 255, 70),
        "outer_edge": (35, 220, 35),
        "inner_fill": (220, 255, 225),
        "hilt_section_edges": ((92, 150, 92), (72, 128, 72), (112, 172, 112)),
        "hilt_cap_edge": (130, 188, 130),
        "hilt_emitter_edge": (155, 208, 155),
        "ring_fill": (24, 84, 24),
        "ring_edge": (78, 185, 78),
        "button_colors": ((28, 170, 42), (45, 215, 90)),
        "fin_edge": (70, 165, 70),
    },
    "b": {
        "name": "blue",
        "outer_fill": (255, 130, 40),
        "outer_edge": (225, 85, 20),
        "inner_fill": (255, 235, 220),
        "hilt_section_edges": ((150, 110, 92), (130, 92, 74), (172, 130, 112)),
        "hilt_cap_edge": (190, 145, 125),
        "hilt_emitter_edge": (210, 170, 148),
        "ring_fill": (96, 42, 24),
        "ring_edge": (190, 110, 78),
        "button_colors": ((210, 125, 45), (240, 168, 82)),
        "fin_edge": (170, 100, 70),
    },
}
BEAT_MODE_ENABLED = True
BEAT_SPAWN_INTERVAL_S = 0.85
BEAT_BLOCK_SPEED_MPS = 0.34
BEAT_BLOCK_SIZE_M = 0.055
BEAT_BLOCK_MAX_ACTIVE = 8
BEAT_BLOCK_MIN_DISTANCE_M = 0.30
BEAT_BLOCK_MAX_DISTANCE_M = 0.60
BEAT_BLOCK_LANE_OFFSET_M = 0.080
BEAT_BLOCK_LANES = (
    (-1.0, 0.0),
    (1.0, 0.0),
    (0.0, -1.0),
    (0.0, 1.0),
    (-1.0, -1.0),
    (1.0, 1.0),
    (-1.0, 1.0),
    (1.0, -1.0),
)
BEAT_CUT_EXTRA_RADIUS_M = 0.040
BEAT_CUT_MIN_LASER_POWER = 0.30
SHOW_FACE_INFO = False
UDP_POSE_ENABLED = True
UDP_POSE_HOST = "127.0.0.1"
UDP_POSE_PORT = 5005
POSE_PACKET_STRUCT = struct.Struct("<4sI d 9f 3f")
POSE_MAGIC = b"SABR"
POSE_VERSION = 1
WINDOW_NAME = "Detectar ArUco"
FULLSCREEN_PREVIEW = True
VOICE_CONTROL_ENABLED = True
VOICE_MODEL_DIR = "vosk-model-small-es-0.42"
VOICE_SAMPLE_RATE = 16000
VOICE_BLOCK_SIZE = 8000
VOICE_COMMAND_COOLDOWN_S = 0.75
VOICE_INPUT_DEVICE_HINT = "C-Media"  # Cambia a "NexiGo" si prefieres el micro de webcam.


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

        if recognized_text in {"r", "g", "b"}:
            return f"color_{recognized_text}"

        if tokens & red_words:
            return "color_r"
        if tokens & green_words:
            return "color_g"
        if tokens & blue_words:
            return "color_b"

        if tokens & flip_words:
            return "toggle_flip"

        toggle_phrases = (
            "espacio",
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

            ranked.append((score, idx, name, hostapi_name))

        ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return [idx for _, idx, _, _ in ranked]

    @staticmethod
    def _describe_device(device_index: int) -> str:
        try:
            device = sd.query_devices(device_index)
            hostapi_name = sd.query_hostapis(device["hostapi"])["name"]
            return f"{device_index}: {device['name']} ({hostapi_name})"
        except Exception:
            return str(device_index)

    def _candidate_sample_rates(self, device_index: int) -> list[int]:
        rates = []

        preferred = int(max(8000, self.sample_rate))
        rates.append(preferred)

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


class PoseUdpSender:
    def __init__(self, host: str, port: int):
        self.target = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_pose(self, rotation_cv: np.ndarray, translation_cv: np.ndarray):
        payload = POSE_PACKET_STRUCT.pack(
            POSE_MAGIC,
            POSE_VERSION,
            time.perf_counter(),
            *rotation_cv.astype(np.float32).reshape(9),
            *translation_cv.astype(np.float32).reshape(3),
        )
        try:
            self.sock.sendto(payload, self.target)
        except OSError:
            # Si falla la red local, no se interrumpe el tracking visual.
            pass

    def close(self):
        self.sock.close()


class PoseSmoother:
    def __init__(self, translation_alpha: float, rotation_alpha: float):
        self.translation_alpha = float(np.clip(translation_alpha, 0.0, 1.0))
        self.rotation_alpha = float(np.clip(rotation_alpha, 0.0, 1.0))
        self._R_smooth = None
        self._t_smooth = None

    def update(
        self,
        rvec: np.ndarray,
        tvec: np.ndarray,
        translation_alpha: float | None = None,
        rotation_alpha: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if translation_alpha is None:
            translation_alpha = self.translation_alpha
        else:
            translation_alpha = float(np.clip(translation_alpha, 0.0, 1.0))

        if rotation_alpha is None:
            rotation_alpha = self.rotation_alpha
        else:
            rotation_alpha = float(np.clip(rotation_alpha, 0.0, 1.0))

        R_measured, _ = cv2.Rodrigues(rvec.astype(np.float32).reshape(3, 1))
        t_measured = tvec.astype(np.float32).reshape(3)

        if self._R_smooth is None:
            self._R_smooth = R_measured
            self._t_smooth = t_measured
        else:
            self._t_smooth = ((1.0 - translation_alpha) * self._t_smooth) + (translation_alpha * t_measured)

            # Suavizado incremental de rotacion para evitar jitter angular.
            R_delta = self._R_smooth.T @ R_measured
            delta_rvec, _ = cv2.Rodrigues(R_delta)
            step_rvec = (delta_rvec.reshape(3) * rotation_alpha).astype(np.float32)
            R_step, _ = cv2.Rodrigues(step_rvec)
            self._R_smooth = self._R_smooth @ R_step

        smooth_rvec, _ = cv2.Rodrigues(self._R_smooth)
        smooth_tvec = self._t_smooth.reshape(3, 1)
        return smooth_rvec.astype(np.float32), smooth_tvec.astype(np.float32)


def get_adaptive_smoothing_alphas(angle_deg: float) -> tuple[float, float]:
    """
    Activa suavizado solo cuando la normal de la cara esta cerca de 0 o 180 grados
    respecto al eje optico (marcador paralelo o casi paralelo a la lente).
    """
    edge_distance = min(angle_deg, 180.0 - angle_deg)
    if edge_distance >= PARALLEL_SMOOTH_ZONE_DEG:
        return 1.0, 1.0

    # Zona frontal extrema: aplicar el suavizado mas fuerte.
    if edge_distance <= FRONTAL_LOCK_ZONE_DEG:
        return FRONTAL_LOCK_TRANSLATION_ALPHA, FRONTAL_LOCK_ROTATION_ALPHA

    # Transicion suave entre lock frontal y zona sin suavizado.
    blend = (edge_distance - FRONTAL_LOCK_ZONE_DEG) / (PARALLEL_SMOOTH_ZONE_DEG - FRONTAL_LOCK_ZONE_DEG)
    blend = np.clip(blend, 0.0, 1.0)
    blend = blend * blend * (3.0 - 2.0 * blend)  # smoothstep

    t_alpha = SMOOTH_TRANSLATION_ALPHA + (1.0 - SMOOTH_TRANSLATION_ALPHA) * blend
    r_alpha = SMOOTH_ROTATION_ALPHA + (1.0 - SMOOTH_ROTATION_ALPHA) * blend
    return float(np.clip(t_alpha, 0.0, 1.0)), float(np.clip(r_alpha, 0.0, 1.0))


def create_combined_rotation(rotation_axis, additional_z_rotation=0):
    R1 = cv2.Rodrigues(np.array(rotation_axis))[0]
    if additional_z_rotation != 0:
        R2 = cv2.Rodrigues(np.array([0, 0, additional_z_rotation]))[0]
        return R1 @ R2
    return R1


# Geometría del cubo (rotaciones relativas respecto a la cara 17)
CUBE_GEOMETRY = {
    17: {"relative_rotation": np.eye(3)},
    3:  {"relative_rotation": create_combined_rotation([np.pi/2, 0, 0], -np.pi/2)},
    7:  {"relative_rotation": create_combined_rotation([-np.pi/2, 0, 0], 0)},
    15: {"relative_rotation": create_combined_rotation([0, np.pi/2, 0], 0)},
    22: {"relative_rotation": create_combined_rotation([0, -np.pi/2, 0], -np.pi)},
}

# Precalcular las coordenadas 3D de todas las esquinas del cubo referenciadas al centro de la cara 17
CUBE_MARKERS_3D = {}
base_corners = np.array([
    [-HALF_SIZE,  HALF_SIZE, 0],
    [ HALF_SIZE,  HALF_SIZE, 0],
    [ HALF_SIZE, -HALF_SIZE, 0],
    [-HALF_SIZE, -HALF_SIZE, 0]
], dtype=np.float32)

for marker_id, geo in CUBE_GEOMETRY.items():
    R_rel = geo["relative_rotation"]
    # El vector offset del origen (cara 17) hacia el origen de la cara relativa en 3D
    t_adj = np.array([0, 0, -HALF_SIZE]) + R_rel[:, 2] * HALF_SIZE
    geo["relative_translation"] = t_adj.astype(np.float32)
    corners_3d = (R_rel @ base_corners.T).T + t_adj
    CUBE_MARKERS_3D[marker_id] = corners_3d.astype(np.float32)

# Prisma fijo anclado a la cara 17
_primary_base = CUBE_MARKERS_3D[PRIMARY_MARKER]
_primary_normal = CUBE_GEOMETRY[PRIMARY_MARKER]["relative_rotation"][:, 2].astype(np.float32)
_primary_top = _primary_base + _primary_normal * PRISM_LENGTH
PRISM_POINTS_3D = np.vstack((_primary_base, _primary_top)).astype(np.float32)

# Offset en metros para compensar el borde blanco del ArUco (5mm) cuando se usa cara lateral
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
    lane_x, lane_y = random.choice(BEAT_BLOCK_LANES)
    color_key = random.choice(tuple(SABER_COLOR_PRESETS.keys()))
    return {
        "distance": float(random.uniform(BEAT_BLOCK_MIN_DISTANCE_M, BEAT_BLOCK_MAX_DISTANCE_M)),
        "lane_x": float(lane_x),
        "lane_y": float(lane_y),
        "color_key": color_key,
    }


def advance_beat_blocks(blocks: list[dict], dt: float) -> tuple[list[dict], int]:
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
    pts = points_2d.reshape(-1, 2)
    if pts.size == 0:
        return False
    if not np.all(np.isfinite(pts)):
        return False
    return float(np.max(np.abs(pts))) <= max_abs_px


AUDIO_BACKEND = "none"
OPENAL_ON_SOURCE = None
OPENAL_LOOP_SOURCE = None
OPENAL_OFF_SOURCE = None
PYGAME_ON_SOUND = None
PYGAME_LOOP_SOUND = None
PYGAME_OFF_SOUND = None
PYGAME_ON_CHANNEL = None
PYGAME_LOOP_CHANNEL = None


def init_audio_backend(saber_on_path: Path, saber_loop_path: Path, saber_off_path: Path) -> str:
    global AUDIO_BACKEND
    global OPENAL_ON_SOURCE, OPENAL_LOOP_SOURCE, OPENAL_OFF_SOURCE
    global PYGAME_ON_SOUND, PYGAME_LOOP_SOUND, PYGAME_OFF_SOUND, PYGAME_ON_CHANNEL, PYGAME_LOOP_CHANNEL

    AUDIO_BACKEND = "none"
    OPENAL_ON_SOURCE = None
    OPENAL_LOOP_SOURCE = None
    OPENAL_OFF_SOURCE = None
    PYGAME_ON_SOUND = None
    PYGAME_LOOP_SOUND = None
    PYGAME_OFF_SOUND = None
    PYGAME_ON_CHANNEL = None
    PYGAME_LOOP_CHANNEL = None

    if oalOpen is not None:
        try:
            if saber_on_path.exists():
                OPENAL_ON_SOURCE = oalOpen(str(saber_on_path))
            if saber_loop_path.exists():
                OPENAL_LOOP_SOURCE = oalOpen(str(saber_loop_path))
                OPENAL_LOOP_SOURCE.set_looping(True)
            if saber_off_path.exists():
                OPENAL_OFF_SOURCE = oalOpen(str(saber_off_path))

            AUDIO_BACKEND = "openal"
            return AUDIO_BACKEND
        except Exception:
            OPENAL_ON_SOURCE = None
            OPENAL_LOOP_SOURCE = None
            OPENAL_OFF_SOURCE = None
            if oalQuit is not None:
                try:
                    oalQuit()
                except Exception:
                    pass

    if pygame is not None:
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()

            if saber_on_path.exists():
                PYGAME_ON_SOUND = pygame.mixer.Sound(str(saber_on_path))
            if saber_loop_path.exists():
                PYGAME_LOOP_SOUND = pygame.mixer.Sound(str(saber_loop_path))
            if saber_off_path.exists():
                PYGAME_OFF_SOUND = pygame.mixer.Sound(str(saber_off_path))

            PYGAME_ON_CHANNEL = pygame.mixer.Channel(0)
            PYGAME_LOOP_CHANNEL = pygame.mixer.Channel(1)
            AUDIO_BACKEND = "pygame"
            return AUDIO_BACKEND
        except Exception:
            PYGAME_ON_SOUND = None
            PYGAME_LOOP_SOUND = None
            PYGAME_OFF_SOUND = None
            PYGAME_ON_CHANNEL = None
            PYGAME_LOOP_CHANNEL = None

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


def stop_all_sounds() -> None:
    if AUDIO_BACKEND == "openal":
        for source in (OPENAL_ON_SOURCE, OPENAL_LOOP_SOURCE, OPENAL_OFF_SOURCE):
            if source is None:
                continue
            try:
                source.stop()
            except Exception:
                pass

    if AUDIO_BACKEND == "pygame" and pygame is not None:
        try:
            if PYGAME_ON_CHANNEL is not None:
                PYGAME_ON_CHANNEL.stop()
            if PYGAME_LOOP_CHANNEL is not None:
                PYGAME_LOOP_CHANNEL.stop()
        except pygame.error:
            pass


def play_saber_on_sound(sound_path: Path) -> bool:
    if AUDIO_BACKEND == "openal":
        return play_openal_source(OPENAL_ON_SOURCE)

    if AUDIO_BACKEND == "pygame" and pygame is not None:
        if PYGAME_ON_SOUND is None or PYGAME_ON_CHANNEL is None:
            return False
        try:
            PYGAME_ON_CHANNEL.play(PYGAME_ON_SOUND, loops=0)
            return True
        except pygame.error:
            return False

    return False


def play_saber_off_sound(sound_path: Path) -> bool:
    if AUDIO_BACKEND == "openal":
        return play_openal_source(OPENAL_OFF_SOURCE)

    if AUDIO_BACKEND == "pygame" and pygame is not None:
        if PYGAME_OFF_SOUND is None or PYGAME_ON_CHANNEL is None:
            return False
        try:
            PYGAME_ON_CHANNEL.play(PYGAME_OFF_SOUND, loops=0)
            return True
        except pygame.error:
            return False

    return False


def play_saber_loop_sound(sound_path: Path) -> bool:
    if AUDIO_BACKEND == "openal":
        return play_openal_source(OPENAL_LOOP_SOURCE)

    if AUDIO_BACKEND == "pygame" and pygame is not None:
        if PYGAME_LOOP_SOUND is None or PYGAME_LOOP_CHANNEL is None:
            return False
        try:
            PYGAME_LOOP_CHANNEL.play(PYGAME_LOOP_SOUND, loops=-1)
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

    if AUDIO_BACKEND == "pygame" and pygame is not None:
        try:
            if pygame.mixer.get_init():
                pygame.mixer.quit()
        except pygame.error:
            pass


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[Camera] No se pudo abrir la camara 0. Cierra otras apps que usen la webcam y reintenta.")
    raise SystemExit(1)

# Buscamos el archivo subiendo un nivel y entrando en Proyecto_VR3 o en la misma carpeta
calibration_path = Path(__file__).parent.parent / "Proyecto_VR3" / "camera_calibration.npz"
if not calibration_path.exists():
    calibration_path = Path(__file__).with_name("camera_calibration.npz")
camera_matrix, dist_coeffs = load_calibration(calibration_path)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())
pose_sender = PoseUdpSender(UDP_POSE_HOST, UDP_POSE_PORT) if UDP_POSE_ENABLED else None
pose_smoother = PoseSmoother(SMOOTH_TRANSLATION_ALPHA, SMOOTH_ROTATION_ALPHA)
laser_target_on = False
laser_power = 0.0
last_frame_time = time.perf_counter()
last_toggle_time = 0.0
saber_on_sound_path = Path(__file__).with_name(SABER_ON_SOUND_FILE)
saber_loop_sound_path = Path(__file__).with_name(SABER_LOOP_SOUND_FILE)
saber_off_sound_path = Path(__file__).with_name(SABER_OFF_SOUND_FILE)
audio_backend = init_audio_backend(saber_on_sound_path, saber_loop_sound_path, saber_off_sound_path)
loop_start_delay_s = SABER_LOOP_START_DELAY_S
print(f"[Audio] Backend activo: {audio_backend}")
if audio_backend == "none":
    print("[Audio] Sin backend. Instala PyOpenAL (pip install PyOpenAL) o pygame.")
print(f"[App] PID: {os.getpid()} | ESC: salir | Espacio/RGB/I: teclado | Voz activa si modelo disponible")

voice_listener = None
voice_last_command_time = {
    "toggle_power": 0.0,
    "toggle_flip": 0.0,
    "color_r": 0.0,
    "color_g": 0.0,
    "color_b": 0.0,
}

if VOICE_CONTROL_ENABLED:
    voice_model_path = Path(__file__).with_name(VOICE_MODEL_DIR)
    voice_listener = VoiceCommandListener(
        model_path=voice_model_path,
        sample_rate=VOICE_SAMPLE_RATE,
        block_size=VOICE_BLOCK_SIZE,
        device_hint=VOICE_INPUT_DEVICE_HINT,
    )
    if voice_listener.start():
        time.sleep(0.15)
        if voice_listener.active_device_name:
            if voice_listener.active_sample_rate is not None:
                print(
                    f"[Voice] Entrada activa: {voice_listener.active_device_name} "
                    f"@ {voice_listener.active_sample_rate} Hz"
                )
            else:
                print(f"[Voice] Entrada activa: {voice_listener.active_device_name}")
        print("[Voice] Control activo: di 'espacio', 'rojo/verde/azul' o 'invertir'.")
    else:
        print(voice_listener.error_message)
        voice_listener = None

loop_start_due_time = None
loop_playing = False
camera_horizontal_flip = CAMERA_HORIZONTAL_FLIP_DEFAULT
current_saber_color_key = DEFAULT_SABER_COLOR_KEY
last_flip_toggle_time = 0.0
beat_blocks = []
beat_score = 0
beat_misses = 0
beat_last_spawn_time = time.perf_counter()
camera_read_fail_count = 0

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
if FULLSCREEN_PREVIEW:
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        camera_read_fail_count += 1
        if camera_read_fail_count == 1:
            print("[Camera] No se pudo leer frame de la camara. Reintentando...")
        if camera_read_fail_count >= 45:
            print("[Camera] Lectura fallida continua. Cerrando la app para liberar recursos.")
            break
        time.sleep(0.01)
        continue
    camera_read_fail_count = 0

    now = time.perf_counter()
    dt = float(np.clip(now - last_frame_time, 0.0, 0.10))
    last_frame_time = now

    # Animacion de encendido/apagado del sable controlada por estado objetivo.
    if laser_target_on:
        laser_power = min(1.0, laser_power + (dt / max(LASER_ON_DURATION_S, 1e-6)))
    else:
        laser_power = max(0.0, laser_power - (dt / max(LASER_OFF_DURATION_S, 1e-6)))

    # Audio: sonido de encendido una vez y loop continuo mientras el sable este activo.
    if laser_target_on and not loop_playing and loop_start_due_time is not None and now >= loop_start_due_time:
        loop_playing = play_saber_loop_sound(saber_loop_sound_path)

    if BEAT_MODE_ENABLED:
        beat_blocks, expired_misses = advance_beat_blocks(beat_blocks, dt)
        beat_misses += expired_misses

        while (now - beat_last_spawn_time) >= BEAT_SPAWN_INTERVAL_S and len(beat_blocks) < BEAT_BLOCK_MAX_ACTIVE:
            beat_blocks.append(create_beat_block())
            beat_last_spawn_time += BEAT_SPAWN_INTERVAL_S

    if voice_listener is not None:
        if voice_listener.error_message:
            print(voice_listener.error_message)
            voice_listener.stop()
            voice_listener = None
        else:
            for voice_command, spoken_text in voice_listener.pop_commands():
                command_now = time.perf_counter()
                if (command_now - voice_last_command_time.get(voice_command, 0.0)) < VOICE_COMMAND_COOLDOWN_S:
                    continue
                voice_last_command_time[voice_command] = command_now

                if voice_command == "toggle_power":
                    laser_target_on = not laser_target_on
                    if laser_target_on:
                        stop_all_sounds()
                        loop_playing = False
                        play_saber_on_sound(saber_on_sound_path)
                        loop_start_due_time = command_now + max(0.0, loop_start_delay_s)
                        print(f"[Voice] '{spoken_text}' -> sable ON")
                    else:
                        stop_all_sounds()
                        loop_playing = False
                        play_saber_off_sound(saber_off_sound_path)
                        loop_start_due_time = None
                        print(f"[Voice] '{spoken_text}' -> sable OFF")
                    last_toggle_time = command_now

                elif voice_command == "toggle_flip":
                    camera_horizontal_flip = not camera_horizontal_flip
                    print(
                        f"[Voice] '{spoken_text}' -> inversion horizontal: "
                        f"{'ON' if camera_horizontal_flip else 'OFF'}"
                    )
                    last_flip_toggle_time = command_now

                elif voice_command.startswith("color_"):
                    color_key = voice_command[-1]
                    if color_key in SABER_COLOR_PRESETS:
                        current_saber_color_key = color_key
                        color_name = SABER_COLOR_PRESETS[color_key]["name"]
                        print(f"[Voice] '{spoken_text}' -> color {color_name}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    marker_labels = []

    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners)
        
        # Pose de los marcadores (se usa para estimar la pose del cubo)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, camera_matrix, dist_coeffs)
        # Etiquetas opcionales de depuracion (sin mostrar IDs de marcadores).
        for i, marker_corners in enumerate(corners):
            cx, cy = int(marker_corners[0][:, 0].mean()), int(marker_corners[0][:, 1].mean())
            if SHOW_FACE_INFO:
                x, y, z = tvecs[i][0]
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                marker_labels.append({
                    "cx": cx,
                    "cy": cy,
                    "text": f"C({cx},{cy}) X:{x:+.3f} Y:{y:+.3f} Z:{z:+.3f}m",
                })

        # Elegir la cara de referencia: 17 si está visible, si no, la más visible
        best_marker_id, best_idx = select_reference_face(corners, ids)

        # Estimar pose del cubo usando SOLO esa cara
        prism_rvec, prism_tvec = None, None
        control_tvec = None
        
        if best_marker_id is not None and best_idx is not None:
            prism_rvec, prism_tvec = estimate_cube_pose_from_marker_pose(
                best_marker_id,
                rvecs[best_idx][0],
                tvecs[best_idx][0]
            )
            
            if prism_rvec is not None and prism_tvec is not None:
                R_measured, _ = cv2.Rodrigues(prism_rvec)
                z_normal = R_measured[:, 2]
                angle_deg = np.degrees(np.arccos(np.clip(z_normal[2], -1.0, 1.0)))
                adaptive_t_alpha, adaptive_r_alpha = get_adaptive_smoothing_alphas(angle_deg)

                prism_rvec, prism_tvec = pose_smoother.update(
                    prism_rvec,
                    prism_tvec,
                    adaptive_t_alpha,
                    adaptive_r_alpha,
                )
                control_tvec = get_control_anchor_tvec(prism_rvec, prism_tvec)

        if prism_rvec is not None and prism_tvec is not None and control_tvec is not None:
            R, _ = cv2.Rodrigues(prism_rvec)
            if pose_sender is not None:
                pose_sender.send_pose(R, control_tvec.reshape(3))

            # Sable láser anclado a la cara 17; la pose se obtiene de la cara de referencia visible
            # Aplicar offset de 5mm hacia adentro si se está usando una cara lateral
            prism_points_adjusted = get_prism_points_with_offset(best_marker_id)
            saber_color = SABER_COLOR_PRESETS.get(
                current_saber_color_key,
                SABER_COLOR_PRESETS[DEFAULT_SABER_COLOR_KEY],
            )
            
            # Geometria del sable: mango estilizado segmentado + hoja luminosa.
            center_base = np.mean(prism_points_adjusted[:4], axis=0)
            center_top = np.mean(prism_points_adjusted[4:], axis=0)
            
            # Obtener ejes perpendiculares a la base para dibujar círculos
            face_R = CUBE_GEOMETRY[PRIMARY_MARKER]["relative_rotation"]
            axis_x = face_R[:, 0]
            axis_y = face_R[:, 1]

            blade_vec = center_top - center_base
            blade_len = float(np.linalg.norm(blade_vec))
            if blade_len > 1e-6:
                blade_dir = blade_vec / blade_len

                # Evita que la hoja cruce el plano de la camara (z ~ 0), lo cual puede congelar el render.
                center_base_cam = (R @ center_base.reshape(3, 1)) + prism_tvec.reshape(3, 1)
                blade_dir_cam = R @ blade_dir.reshape(3, 1)
                safe_blade_len = blade_len
                dir_z = float(blade_dir_cam[2, 0])
                base_z = float(center_base_cam[2, 0])
                if dir_z < -1e-6:
                    max_len_before_camera = (base_z - LASER_MIN_DRAW_DEPTH_M) / (-dir_z)
                    safe_blade_len = min(safe_blade_len, max(0.0, max_len_before_camera))

                radius_outer = HALF_SIZE * LASER_OUTER_RADIUS_SCALE
                radius_inner = HALF_SIZE * LASER_INNER_RADIUS_SCALE
                handle_radius = HALF_SIZE * HANDLE_RADIUS_SCALE
                handle_cap_radius = HALF_SIZE * HANDLE_CAP_RADIUS_SCALE
                emitter_radius = HALF_SIZE * EMITTER_RADIUS_SCALE

                handle_end = center_base - blade_dir * HANDLE_LENGTH_M
                split_1, split_2 = HANDLE_PROFILE_SPLITS
                r0, r1, r2, r3 = HANDLE_PROFILE_RADIUS_SCALES
                section_1_end = handle_end + blade_dir * (HANDLE_LENGTH_M * split_1)
                section_2_end = handle_end + blade_dir * (HANDLE_LENGTH_M * split_2)

                hilt_section_points = [
                    build_frustum_points(
                        handle_end,
                        section_1_end,
                        axis_x,
                        axis_y,
                        handle_radius * r0,
                        handle_radius * r1,
                        LASER_CYLINDER_SIDES,
                    ),
                    build_frustum_points(
                        section_1_end,
                        section_2_end,
                        axis_x,
                        axis_y,
                        handle_radius * r1,
                        handle_radius * r2,
                        LASER_CYLINDER_SIDES,
                    ),
                    build_frustum_points(
                        section_2_end,
                        center_base,
                        axis_x,
                        axis_y,
                        handle_radius * r2,
                        handle_radius * r3,
                        LASER_CYLINDER_SIDES,
                    ),
                ]

                handle_cap_end = handle_end - blade_dir * HANDLE_CAP_LENGTH_M
                handle_cap_points = build_frustum_points(
                    handle_cap_end,
                    handle_end,
                    axis_x,
                    axis_y,
                    handle_cap_radius,
                    handle_radius * 1.05,
                    LASER_CYLINDER_SIDES,
                )

                emitter_end = center_base + blade_dir * EMITTER_LENGTH_M
                emitter_points = build_frustum_points(
                    center_base,
                    emitter_end,
                    axis_x,
                    axis_y,
                    emitter_radius * 1.06,
                    emitter_radius * 0.88,
                    LASER_CYLINDER_SIDES,
                )

                detail_ring_points = []
                ring_radius = handle_radius * HANDLE_DETAIL_RING_RADIUS_SCALE
                ring_half_len = HANDLE_DETAIL_RING_LENGTH_M * 0.5
                for ring_idx in range(HANDLE_DETAIL_RING_COUNT):
                    ring_ratio = HANDLE_DETAIL_RING_START_RATIO + (ring_idx * HANDLE_DETAIL_RING_STEP_RATIO)
                    if ring_ratio <= 0.0 or ring_ratio >= 1.0:
                        continue

                    ring_center = handle_end + blade_dir * (HANDLE_LENGTH_M * ring_ratio)
                    ring_start = ring_center - blade_dir * ring_half_len
                    ring_end = ring_center + blade_dir * ring_half_len
                    detail_ring_points.append(
                        build_cylinder_points(
                            ring_start,
                            ring_end,
                            axis_x,
                            axis_y,
                            ring_radius,
                            LASER_CYLINDER_SIDES,
                        )
                    )

                button_points = []
                button_axis = axis_x.astype(np.float32)
                button_radius = HALF_SIZE * HANDLE_BUTTON_RADIUS_SCALE
                for button_ratio in HANDLE_BUTTON_POSITIONS:
                    button_center = handle_end + blade_dir * (HANDLE_LENGTH_M * float(button_ratio))
                    button_start = button_center + button_axis * (handle_radius * 0.92)
                    button_end = button_start + button_axis * HANDLE_BUTTON_HEIGHT_M
                    button_points.append(
                        build_cylinder_points(
                            button_start,
                            button_end,
                            blade_dir,
                            axis_y,
                            button_radius,
                            LASER_CYLINDER_SIDES,
                        )
                    )

                fin_points = []
                fin_radius = HALF_SIZE * HANDLE_FIN_RADIUS_SCALE
                for fin_ratio in HANDLE_FIN_POSITIONS:
                    fin_center = handle_end + blade_dir * (HANDLE_LENGTH_M * float(fin_ratio))
                    for fin_idx in range(HANDLE_FIN_COUNT):
                        angle = (2.0 * np.pi * fin_idx) / HANDLE_FIN_COUNT
                        fin_normal = (axis_x * np.cos(angle)) + (axis_y * np.sin(angle))
                        fin_normal = fin_normal.astype(np.float32)
                        fin_normal_norm = float(np.linalg.norm(fin_normal))
                        if fin_normal_norm <= 1e-6:
                            continue
                        fin_normal /= fin_normal_norm

                        fin_start = fin_center + fin_normal * (handle_radius * 0.95)
                        fin_end = fin_start + fin_normal * HANDLE_FIN_HEIGHT_M
                        fin_tangent = np.cross(blade_dir, fin_normal).astype(np.float32)
                        fin_tangent_norm = float(np.linalg.norm(fin_tangent))
                        if fin_tangent_norm <= 1e-6:
                            fin_tangent = axis_y.astype(np.float32)
                        else:
                            fin_tangent /= fin_tangent_norm

                        fin_points.append(
                            build_cylinder_points(
                                fin_start,
                                fin_end,
                                blade_dir,
                                fin_tangent,
                                fin_radius,
                                LASER_CYLINDER_SIDES,
                            )
                        )

                pts_hilt_sections = []
                for hilt_points in hilt_section_points:
                    pts_hilt, _ = cv2.projectPoints(hilt_points, prism_rvec, prism_tvec, camera_matrix, dist_coeffs)
                    pts_hilt_sections.append(pts_hilt)

                pts_handle_cap, _ = cv2.projectPoints(handle_cap_points, prism_rvec, prism_tvec, camera_matrix, dist_coeffs)
                pts_emitter, _ = cv2.projectPoints(emitter_points, prism_rvec, prism_tvec, camera_matrix, dist_coeffs)

                pts_detail_rings = []
                for ring_points in detail_ring_points:
                    pts_ring, _ = cv2.projectPoints(ring_points, prism_rvec, prism_tvec, camera_matrix, dist_coeffs)
                    pts_detail_rings.append(pts_ring)

                pts_buttons = []
                for button_pts in button_points:
                    pts_button, _ = cv2.projectPoints(button_pts, prism_rvec, prism_tvec, camera_matrix, dist_coeffs)
                    pts_buttons.append(pts_button)

                pts_fins = []
                for fin_pts in fin_points:
                    pts_fin, _ = cv2.projectPoints(fin_pts, prism_rvec, prism_tvec, camera_matrix, dist_coeffs)
                    pts_fins.append(pts_fin)

                hilt_hulls = []
                hilt_safe = True
                for pts_hilt in pts_hilt_sections:
                    if not projected_points_are_safe(pts_hilt, MAX_PROJECTED_COORD_ABS_PX):
                        hilt_safe = False
                        break
                    hilt_hulls.append(cv2.convexHull(pts_hilt.reshape(-1, 2).astype(np.int32)))

                cap_safe = projected_points_are_safe(pts_handle_cap, MAX_PROJECTED_COORD_ABS_PX)
                emitter_safe = projected_points_are_safe(pts_emitter, MAX_PROJECTED_COORD_ABS_PX)

                ring_hulls = []
                rings_safe = True
                for pts_ring in pts_detail_rings:
                    if not projected_points_are_safe(pts_ring, MAX_PROJECTED_COORD_ABS_PX):
                        rings_safe = False
                        break
                    ring_hulls.append(cv2.convexHull(pts_ring.reshape(-1, 2).astype(np.int32)))

                button_hulls = []
                buttons_safe = True
                for pts_button in pts_buttons:
                    if not projected_points_are_safe(pts_button, MAX_PROJECTED_COORD_ABS_PX):
                        buttons_safe = False
                        break
                    button_hulls.append(cv2.convexHull(pts_button.reshape(-1, 2).astype(np.int32)))

                fin_hulls = []
                fins_safe = True
                for pts_fin in pts_fins:
                    if not projected_points_are_safe(pts_fin, MAX_PROJECTED_COORD_ABS_PX):
                        fins_safe = False
                        break
                    fin_hulls.append(cv2.convexHull(pts_fin.reshape(-1, 2).astype(np.int32)))

                if hilt_safe and cap_safe and emitter_safe and rings_safe and buttons_safe and fins_safe:
                    hull_handle_cap = cv2.convexHull(pts_handle_cap.reshape(-1, 2).astype(np.int32))
                    hull_emitter = cv2.convexHull(pts_emitter.reshape(-1, 2).astype(np.int32))

                    hilt_fill_palette = [
                        (30, 30, 42),
                        (20, 20, 28),
                        (36, 36, 50),
                    ]
                    hilt_edge_palette = saber_color["hilt_section_edges"]
                    for section_idx, section_hull in enumerate(hilt_hulls):
                        fill_color = hilt_fill_palette[section_idx % len(hilt_fill_palette)]
                        edge_color = hilt_edge_palette[section_idx % len(hilt_edge_palette)]
                        cv2.fillConvexPoly(frame, section_hull, fill_color)
                        cv2.polylines(frame, [section_hull], True, edge_color, 2)

                    cv2.fillConvexPoly(frame, hull_handle_cap, (18, 18, 24))
                    cv2.fillConvexPoly(frame, hull_emitter, (62, 62, 78))
                    cv2.polylines(frame, [hull_handle_cap], True, saber_color["hilt_cap_edge"], 2)
                    cv2.polylines(frame, [hull_emitter], True, saber_color["hilt_emitter_edge"], 2)

                    for ring_hull in ring_hulls:
                        cv2.fillConvexPoly(frame, ring_hull, saber_color["ring_fill"])
                        cv2.polylines(frame, [ring_hull], True, saber_color["ring_edge"], 1)

                    button_colors = saber_color["button_colors"]
                    for button_idx, button_hull in enumerate(button_hulls):
                        base_color = button_colors[button_idx % len(button_colors)]
                        cv2.fillConvexPoly(frame, button_hull, base_color)
                        cv2.polylines(frame, [button_hull], True, (235, 240, 245), 1)

                    for fin_hull in fin_hulls:
                        cv2.fillConvexPoly(frame, fin_hull, (16, 16, 22))
                        cv2.polylines(frame, [fin_hull], True, saber_color["fin_edge"], 1)

                if laser_power > LASER_MIN_VISIBLE_POWER and safe_blade_len > 1e-5:
                    animated_blade_len = min(blade_len * laser_power, safe_blade_len)
                    if animated_blade_len > 1e-5:
                        blade_tip = center_base + (blade_dir * animated_blade_len)

                        layer_outer = build_cylinder_points(
                            center_base,
                            blade_tip,
                            axis_x,
                            axis_y,
                            radius_outer,
                            LASER_CYLINDER_SIDES,
                        )
                        layer_inner = build_cylinder_points(
                            center_base,
                            blade_tip,
                            axis_x,
                            axis_y,
                            radius_inner,
                            LASER_CYLINDER_SIDES,
                        )

                        pts_outer, _ = cv2.projectPoints(layer_outer, prism_rvec, prism_tvec, camera_matrix, dist_coeffs)
                        pts_inner, _ = cv2.projectPoints(layer_inner, prism_rvec, prism_tvec, camera_matrix, dist_coeffs)

                        outer_safe = projected_points_are_safe(pts_outer, MAX_PROJECTED_COORD_ABS_PX)
                        inner_safe = projected_points_are_safe(pts_inner, MAX_PROJECTED_COORD_ABS_PX)
                        if outer_safe and inner_safe:
                            hull_outer = cv2.convexHull(pts_outer.reshape(-1, 2).astype(np.int32))
                            hull_inner = cv2.convexHull(pts_inner.reshape(-1, 2).astype(np.int32))

                            glow_alpha = 0.08 + (0.20 * laser_power)
                            glow_thickness = max(1, int(round(1 + (3 * laser_power))))

                            # Halo de color segun tecla seleccionada (R/G/B).
                            overlay = frame.copy()
                            cv2.fillConvexPoly(overlay, hull_outer, saber_color["outer_fill"])
                            cv2.polylines(overlay, [hull_outer], True, saber_color["outer_edge"], glow_thickness)
                            cv2.addWeighted(overlay, glow_alpha, frame, 1.0 - glow_alpha, 0, frame)

                            # Núcleo claro del láser.
                            cv2.fillConvexPoly(frame, hull_inner, saber_color["inner_fill"])
                            cv2.polylines(frame, [hull_inner], True, (255, 255, 255), 2)

    # Mostrar frame
    mirrored_frame = cv2.flip(frame, 1) if camera_horizontal_flip else frame.copy()
    h, w = mirrored_frame.shape[:2]
    
    if marker_labels:
        max_text_width = 550 if SHOW_FACE_INFO else 160
        for label in marker_labels:
            text_x = max(10, min(label["cx"] + 10, w - max_text_width))
            text_y = max(20, min(label["cy"] - 10, h - 10))
            cv2.putText(mirrored_frame, label["text"], (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    try:
        _, _, window_w, window_h = cv2.getWindowImageRect(WINDOW_NAME)
    except cv2.error:
        window_w, window_h = mirrored_frame.shape[1], mirrored_frame.shape[0]

    display_frame = fit_frame_to_window(mirrored_frame, window_w, window_h)
    cv2.imshow(WINDOW_NAME, display_frame)

    try:
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            print("[App] Ventana cerrada por el usuario.")
            break
    except cv2.error:
        print("[App] Ventana no disponible. Cerrando la app.")
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        toggle_now = time.perf_counter()
        if (toggle_now - last_toggle_time) > KEY_TOGGLE_COOLDOWN_S:
            laser_target_on = not laser_target_on
            if laser_target_on:
                stop_all_sounds()
                loop_playing = False
                play_saber_on_sound(saber_on_sound_path)
                loop_start_due_time = toggle_now + max(0.0, loop_start_delay_s)
            else:
                stop_all_sounds()
                loop_playing = False
                play_saber_off_sound(saber_off_sound_path)
                loop_start_due_time = None
            last_toggle_time = toggle_now
    if key == ord("i"):
        flip_now = time.perf_counter()
        if (flip_now - last_flip_toggle_time) > KEY_TOGGLE_COOLDOWN_S:
            camera_horizontal_flip = not camera_horizontal_flip
            print(f"[View] Inversion horizontal: {'ON' if camera_horizontal_flip else 'OFF'}")
            last_flip_toggle_time = flip_now
    if key in (ord("r"), ord("g"), ord("b")):
        selected_color_key = chr(key)
        if selected_color_key in SABER_COLOR_PRESETS:
            current_saber_color_key = selected_color_key
            print(f"[Saber] Color seleccionado: {SABER_COLOR_PRESETS[selected_color_key]['name']}")
    if key == 27:
        break

cap.release()
if pose_sender is not None:
    pose_sender.close()
if voice_listener is not None:
    voice_listener.stop()
shutdown_audio_backend()
cv2.destroyAllWindows()
