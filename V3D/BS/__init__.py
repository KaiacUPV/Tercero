"""Modular version of the OpenGL Saber application."""

from .app import OpenGLSaberApp
from .cli import main, parse_args

__all__ = ["OpenGLSaberApp", "main", "parse_args"]
