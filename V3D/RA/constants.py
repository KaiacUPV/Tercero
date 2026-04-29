# Auto-extracted from id.py without modifying the original file.
"""Constantes del prototipo RA.

Agrupa medidas del cubo/sable, colores, audio, camara, voz y el modo simple de
bloques para ajustar el prototipo sin tocar la logica principal.
"""

# Geometria fisica del cubo ArUco y longitud visual de la hoja.
PRIMARY_MARKER = 17
CUBE_SIZE_M = 0.065
MARKER_SIZE = CUBE_SIZE_M
HALF_SIZE = CUBE_SIZE_M / 2.0
PRISM_LENGTH = 0.62

# Zonas angulares (grados) usadas para ajustar el suavizado.
PARALLEL_SMOOTH_ZONE_DEG = 30.0
FRONTAL_LOCK_ZONE_DEG = 10.0

# Suavizado base (alpha) de traslación/rotación.
# Valores más pequeños = más estable pero más "lento".
SMOOTH_TRANSLATION_ALPHA = 0.16
SMOOTH_ROTATION_ALPHA = 0.14
FRONTAL_LOCK_TRANSLATION_ALPHA = 0.07
FRONTAL_LOCK_ROTATION_ALPHA = 0.03

# Parametrización del modelo del sable (solo para el render 2D/preview).
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

# Audio
SABER_ON_SOUND_FILE = "sable-on.wav"
SABER_LOOP_SOUND_FILE = "loop.wav"
SABER_OFF_SOUND_FILE = "sable-off.wav"
SABER_LOOP_START_DELAY_S = 0.09

# Cámara / UI
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

# Ventana OpenCV
WINDOW_NAME = "Detectar ArUco"
FULLSCREEN_PREVIEW = True

# Voz (Vosk)
VOICE_CONTROL_ENABLED = True
VOICE_MODEL_DIR = "vosk-model-small-es-0.42"
VOICE_SAMPLE_RATE = 16000
VOICE_BLOCK_SIZE = 8000
VOICE_COMMAND_COOLDOWN_S = 0.75
VOICE_INPUT_DEVICE_HINT = "C-Media"  # Cambia a "NexiGo" si prefieres el micro de webcam.
