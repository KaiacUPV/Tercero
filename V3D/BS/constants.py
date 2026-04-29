# Auto-extracted from openglsable.py without modifying the original file.
"""Constantes de configuracion del juego tipo Beat Saber.

Aqui se concentran medidas, velocidades, colores, audio, voz, VR y parametros de
combate. Tenerlas juntas facilita ajustar el prototipo sin entrar en BS.app.
"""
PRIMARY_MARKER = 17
DEFAULT_CUBE_SIZE_M = 0.065

# Animacion y colisiones del sable del jugador.
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

# Reconocimiento de voz con Vosk/sounddevice.
VOICE_CONTROL_ENABLED = True
VOICE_MODEL_DIR = "vosk-model-small-es-0.42"
VOICE_SAMPLE_RATE = 16000
VOICE_BLOCK_SIZE = 8000
VOICE_COMMAND_COOLDOWN_S = 0.75
VOICE_INPUT_DEVICE_HINT = "C-Media"

# Orientacion del movil por UDP para modo VR.
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

# Modo bloques/proyectiles.
PROJECTILE_RADIUS = 0.055
PROJECTILE_SPEED = 3.35
PROJECTILE_SPAWN_INTERVAL_S = 1.15
PROJECTILE_REBOUND_SPEED = 4.15
PROJECTILE_LASER_LENGTH = 0.30
GAME_MODE_BLOCKS = "blocks"
GAME_MODE_COMBAT = "combat"

# Modo combate contra sable enemigo.
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

# Separacion entre ojos para render estereoscopico lado a lado.
VR_EYE_SEPARATION = 0.065

# Ventana y colores del sable.
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
