# Auto-extracted from id.py without modifying the original file.
"""Control por voz del prototipo RA.

Ejecuta Vosk en un hilo, parsea frases simples (espacio, colores, invertir) y
entrega comandos al bucle principal sin bloquear la camara.
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

class VoiceCommandListener:
    """Reconocimiento de comandos de voz (Vosk) en un hilo.

    Diseño:
    - Abre un `RawInputStream` con sounddevice.
    - Alimenta a Vosk con audio en bruto y extrae texto final/parcial.
    - Convierte texto a comandos discretos (`toggle_power`, `color_r`, etc.).
    - Encola comandos en una `Queue` para que el hilo principal los procese.

    Motivo:
    - Aislar audio/voz del hilo principal evita congelamientos cuando el micrófono
      o el backend de audio falla o tarda en inicializar.
    """

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
        """Mapa simple de palabras -> comando.

        Mantiene el reconocimiento tolerante a variantes (por ejemplo "erre" -> rojo).
        Se usa un set de tokens para robustez ante frases más largas.
        """
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
        # Estrategia de arranque:
        # - Probar varios dispositivos de entrada (ranked por heurística).
        # - Para cada dispositivo, intentar sample rates razonables (preferido + default).
        # - Cuando uno funciona, mantener ese stream y salir.
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
