# Auto-extracted from id.py without modifying the original file.
"""Audio del prototipo de realidad aumentada.

Este modulo intenta usar OpenAL primero y pygame.mixer despues. La app puede
seguir funcionando sin sonido: en ese caso AUDIO_BACKEND queda en "none" y las
funciones de reproduccion devuelven False.
"""
import cv2
import cv2.aruco as aruco
import json
import numpy as np
import os
import queue
import random
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

from .constants import *

AUDIO_BACKEND = "none"
# Fuentes/caches globales para no recargar los wav en cada pulsacion.
OPENAL_ON_SOURCE = None
OPENAL_LOOP_SOURCE = None
OPENAL_OFF_SOURCE = None
PYGAME_ON_SOUND = None
PYGAME_LOOP_SOUND = None
PYGAME_OFF_SOUND = None
PYGAME_ON_CHANNEL = None
PYGAME_LOOP_CHANNEL = None


def init_audio_backend(saber_on_path: Path, saber_loop_path: Path, saber_off_path: Path) -> str:
    """Inicializa el backend de audio.

    Prioridad:
    1) OpenAL (si está instalado y puede cargar sonidos)
    2) pygame.mixer (si está instalado)
    3) none (silencioso)
    """
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

    # OpenAL permite fuentes con looping nativo, ideal para el zumbido continuo.
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
            # Si OpenAL falla a medias, limpiamos sus referencias y probamos pygame.
            OPENAL_ON_SOURCE = None
            OPENAL_LOOP_SOURCE = None
            OPENAL_OFF_SOURCE = None
            if oalQuit is not None:
                try:
                    oalQuit()
                except Exception:
                    pass

    # Fallback: pygame.mixer esta disponible en mas instalaciones Windows/Python.
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
    """Reinicia y reproduce una fuente OpenAL de forma defensiva."""
    if source is None:
        return False

    # Stop/rewind evita que dos toggles rapidos solapen el mismo sonido.
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
    """Corta cualquier reproducción en curso en el backend activo."""
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
    """Reproduce el sonido corto de encendido del sable."""
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
    """Reproduce el sonido corto de apagado del sable."""
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
    """Arranca el bucle de sonido mientras la hoja esta encendida."""
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
    """Libera recursos del backend de audio."""
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
