"""Punto de entrada del juego tipo Beat Saber.

La logica real vive en BS.cli/BS.app; este archivo solo permite lanzar el juego
con `python main_beatsaber.py`.
"""
from BS.cli import main


if __name__ == "__main__":
	# Propaga el codigo de salida del CLI al proceso.
	raise SystemExit(main())
