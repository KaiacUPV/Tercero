"""Punto de entrada del sable en realidad aumentada.

Mantiene el arranque separado del codigo de la app para poder importar RA.app
sin ejecutar automaticamente la camara, el audio ni la ventana OpenCV.
"""
from RA.app import main


if __name__ == "__main__":
    # `main()` devuelve un codigo de salida; SystemExit lo pasa al sistema operativo.
    raise SystemExit(main())
