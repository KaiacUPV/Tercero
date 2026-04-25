import argparse
import sys

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibra una camara con un patron chessboard usando webcam."
    )
    parser.add_argument("--camera", type=int, default=0, help="Indice de la camara")
    parser.add_argument(
        "--cols",
        type=int,
        default=9,
        help="Numero de esquinas internas por columna del chessboard",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=6,
        help="Numero de esquinas internas por fila del chessboard",
    )
    parser.add_argument(
        "--square",
        type=float,
        default=0.025,
        help="Tamano real del lado de un cuadro (metros, mm, etc)",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=15,
        help="Minimo de capturas validas para calibrar",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="calibracion_camara.npz",
        help="Archivo de salida .npz",
    )
    return parser.parse_args()


def build_object_points(cols: int, rows: int, square_size: float) -> np.ndarray:
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid * square_size
    return objp


def compute_reprojection_error(
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    total_error = 0.0
    total_points = 0

    for i, objp in enumerate(objpoints):
        projected, _ = cv2.projectPoints(
            objp, rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv2.norm(imgpoints[i], projected, cv2.NORM_L2)
        total_error += error * error
        total_points += len(objp)

    if total_points == 0:
        return float("nan")

    return float(np.sqrt(total_error / total_points))


def main() -> int:
    args = parse_args()

    pattern_size = (args.cols, args.rows)
    objp = build_object_points(args.cols, args.rows, args.square)

    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"No se pudo abrir la camara con indice {args.camera}")
        return 1

    print("Instrucciones:")
    print("  - Coloca el chessboard en distintos angulos y distancias")
    print("  - Pulsa 'c' para capturar una imagen valida")
    print("  - Pulsa 'q' para terminar y calibrar")

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    image_size = None

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer un frame de la camara")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK,
        )

        vis = frame.copy()

        if found:
            corners_subpix = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria,
            )
            cv2.drawChessboardCorners(vis, pattern_size, corners_subpix, found)
            cv2.putText(
                vis,
                "Patron detectado - pulsa 'c' para guardar",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            corners_subpix = None
            cv2.putText(
                vis,
                "Patron NO detectado",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            vis,
            f"Capturas validas: {len(imgpoints)} / {args.min_frames}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Calibracion Chessboard", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            if found and corners_subpix is not None:
                objpoints.append(objp.copy())
                imgpoints.append(corners_subpix)
                print(f"Captura guardada: {len(imgpoints)}")
            else:
                print("No se guardo captura: patron no detectado")

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if image_size is None:
        print("No se obtuvo tamano de imagen. Abortando.")
        return 1

    if len(imgpoints) < args.min_frames:
        print(
            f"Muy pocas capturas validas ({len(imgpoints)}). "
            f"Necesitas al menos {args.min_frames}."
        )
        return 1

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )

    reproj_error = compute_reprojection_error(
        objpoints,
        imgpoints,
        rvecs,
        tvecs,
        camera_matrix,
        dist_coeffs,
    )

    np.savez(
        args.output,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvecs=np.array(rvecs, dtype=object),
        tvecs=np.array(tvecs, dtype=object),
        image_size=np.array(image_size),
        board_size=np.array(pattern_size),
        square_size=np.array([args.square], dtype=np.float32),
        rms=np.array([rms], dtype=np.float32),
        reprojection_error=np.array([reproj_error], dtype=np.float32),
    )

    print("\nCalibracion completada")
    print(f"RMS: {rms:.6f}")
    print(f"Error reproyeccion medio: {reproj_error:.6f} px")
    print(f"Matriz de camara:\n{camera_matrix}")
    print(f"Distorsion:\n{dist_coeffs.ravel()}")
    print(f"Guardado en: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
