# Auto-extracted from id.py without modifying the original file.
"""Suavizado y rotaciones para tracking RA.

OpenCV entrega poses con ruido; este modulo suaviza traslacion/rotacion y crea
rotaciones auxiliares para describir la geometria del cubo.
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

class PoseSmoother:
    """Filtro de suavizado (EMA) para traslación y rotación.

    - La traslación se filtra con una media exponencial (EMA).
    - La rotación se filtra de forma incremental: calcula delta entre la rotación
      previa y la medida y aplica una fracción (rotation_alpha) del delta.

    Esto evita jitter sin romper ortonormalidad de la matriz de rotación.
    """

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
        """Actualiza el filtro con una medición nueva.

        Args:
            rvec: vector de Rodrigues (3,) o (3,1) en radianes.
            tvec: traslación (3,) o (3,1) en metros.
            translation_alpha: alpha puntual para traslación (si None usa el default).
            rotation_alpha: alpha puntual para rotación (si None usa el default).

        Returns:
            (rvec_suavizado, tvec_suavizado)
        """
        if translation_alpha is None:
            translation_alpha = self.translation_alpha
        else:
            translation_alpha = float(np.clip(translation_alpha, 0.0, 1.0))

        if rotation_alpha is None:
            rotation_alpha = self.rotation_alpha
        else:
            rotation_alpha = float(np.clip(rotation_alpha, 0.0, 1.0))

        # Normalizamos tipos/shape para evitar sorpresas con OpenCV.
        R_measured, _ = cv2.Rodrigues(rvec.astype(np.float32).reshape(3, 1))
        t_measured = tvec.astype(np.float32).reshape(3)

        if self._R_smooth is None:
            # Primera muestra: inicializa el estado del filtro.
            self._R_smooth = R_measured
            self._t_smooth = t_measured
        else:
            # EMA de traslación: mezcla exponencial (1-a)*prev + a*meas.
            self._t_smooth = ((1.0 - translation_alpha) * self._t_smooth) + (translation_alpha * t_measured)

            # Suavizado incremental de rotacion para evitar jitter angular.
            # Queremos acercar la rotación suave a la medida sin interpolaciones que
            # pierdan ortonormalidad. Para ello calculamos un delta en SO(3):
            #   R_measured = R_smooth * R_delta
            # y aplicamos solo una fracción del delta.
            R_delta = self._R_smooth.T @ R_measured
            delta_rvec, _ = cv2.Rodrigues(R_delta)
            step_rvec = (delta_rvec.reshape(3) * rotation_alpha).astype(np.float32)
            R_step, _ = cv2.Rodrigues(step_rvec)
            self._R_smooth = self._R_smooth @ R_step

        smooth_rvec, _ = cv2.Rodrigues(self._R_smooth)
        smooth_tvec = self._t_smooth.reshape(3, 1)
        return smooth_rvec.astype(np.float32), smooth_tvec.astype(np.float32)

def get_adaptive_smoothing_alphas(angle_deg: float) -> tuple[float, float]:
    """Devuelve (translation_alpha, rotation_alpha) según estabilidad geométrica.

    Interpretación:
    - `angle_deg` es el ángulo entre el eje Z del cubo (normal) y el eje Z de cámara.
    - Cuando el marcador está casi paralelo a la lente (angle ~ 90°), suele ser más estable.
    - Cuando está casi frontal o casi de espaldas (angle ~ 0° o ~180°), suelen aparecer
      saltos por ambigüedad/perspectiva y conviene suavizar más.
    """
    # Usamos la distancia al extremo más cercano (0° u 180°).
    edge_distance = min(angle_deg, 180.0 - angle_deg)
    if edge_distance >= PARALLEL_SMOOTH_ZONE_DEG:
        # Zona estable: sin suavizado extra (alpha=1 => seguir medida).
        return 1.0, 1.0

    # Zona frontal extrema: aplicar el suavizado mas fuerte.
    if edge_distance <= FRONTAL_LOCK_ZONE_DEG:
        return FRONTAL_LOCK_TRANSLATION_ALPHA, FRONTAL_LOCK_ROTATION_ALPHA

    # Transicion suave entre lock frontal y zona sin suavizado.
    blend = (edge_distance - FRONTAL_LOCK_ZONE_DEG) / (PARALLEL_SMOOTH_ZONE_DEG - FRONTAL_LOCK_ZONE_DEG)
    blend = np.clip(blend, 0.0, 1.0)
    # Smoothstep para evitar cortes bruscos en la mezcla.
    blend = blend * blend * (3.0 - 2.0 * blend)  # smoothstep

    t_alpha = SMOOTH_TRANSLATION_ALPHA + (1.0 - SMOOTH_TRANSLATION_ALPHA) * blend
    r_alpha = SMOOTH_ROTATION_ALPHA + (1.0 - SMOOTH_ROTATION_ALPHA) * blend
    return float(np.clip(t_alpha, 0.0, 1.0)), float(np.clip(r_alpha, 0.0, 1.0))


def create_combined_rotation(rotation_axis, additional_z_rotation=0):
    """Crea una rotación combinada.

    `rotation_axis` se interpreta como un rvec (Rodrigues) en radianes.
    Si `additional_z_rotation` != 0, compone una rotación extra en Z.
    """
    R1 = cv2.Rodrigues(np.array(rotation_axis))[0]
    if additional_z_rotation != 0:
        R2 = cv2.Rodrigues(np.array([0, 0, additional_z_rotation]))[0]
        return R1 @ R2
    return R1
