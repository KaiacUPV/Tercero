import cv2
import numpy as np
import argparse


# -----------------------------
# Crear puntos 3D del patrón
# -----------------------------
def create_object_points(pattern_size, square_size=1):

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp = objp * square_size

    return objp


# -----------------------------
# Detectar patrón
# -----------------------------
def detect_pattern(gray, pattern_size, pattern_type):

    if pattern_type == "chessboard":

        ret, corners = cv2.findChessboardCorners(gray, pattern_size)

    elif pattern_type == "circle":

        ret, corners = cv2.findCirclesGrid(gray, pattern_size)

    else:
        raise ValueError("Tipo de patrón no soportado")

    return ret, corners


# -----------------------------
# Captura de imágenes y calibración
# -----------------------------
def calibrate_camera(num_images, pattern_size, pattern_type):

    cap = cv2.VideoCapture(0)

    objpoints = []
    imgpoints = []

    objp = create_object_points(pattern_size)

    captured = 0

    print("\nPresiona 's' para guardar imagen")
    print("Presiona 'q' para salir\n")

    while captured < num_images:

        ret, frame = cap.read()

        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, corners = detect_pattern(gray, pattern_size, pattern_type)

        display = frame.copy()

        if found:

            if pattern_type == "chessboard":
                cv2.drawChessboardCorners(display, pattern_size, corners, found)

            if pattern_type == "circle":
                cv2.drawChessboardCorners(display, pattern_size, corners, found)

        cv2.putText(display,
                    f"Images: {captured}/{num_images}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2)

        cv2.imshow("Calibration", display)

        key = cv2.waitKey(1)

        if key == ord('s') and found:

            objpoints.append(objp)
            imgpoints.append(corners)

            captured += 1

            print(f"Imagen {captured} capturada")

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\nCalculando calibración...")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        None,
        None
    )

    print("\nCamera Matrix:\n", mtx)
    print("\nDistortion Coefficients:\n", dist)

    np.savez("camera_calibration.npz",
             mtx=mtx,
             dist=dist,
             rvecs=rvecs,
             tvecs=tvecs)

    print("\nCalibración guardada en camera_calibration.npz")

    return mtx, dist


# -----------------------------
# Proyección de punto virtual
# -----------------------------
def project_virtual_point(pattern_size):

    data = np.load("camera_calibration.npz")

    mtx = data["mtx"]
    dist = data["dist"]

    cap = cv2.VideoCapture(0)

    objp = create_object_points(pattern_size)

    print("\nPresiona ESC para salir")

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret_corners, corners = cv2.findChessboardCorners(gray, pattern_size)

        if ret_corners:

            ret, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)

            virtual_point = np.array([(3,3,0)], dtype=np.float32)

            imgpts, _ = cv2.projectPoints(
                virtual_point,
                rvec,
                tvec,
                mtx,
                dist
            )

            x, y = imgpts[0][0]

            cv2.circle(frame,
                       (int(x), int(y)),
                       10,
                       (0,0,255),
                       -1)

        cv2.imshow("Virtual Point Projection", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# -----------------------------
# Error de reproyección
# -----------------------------
def compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):

    total_error = 0

    for i in range(len(objpoints)):

        imgpoints2, _ = cv2.projectPoints(
            objpoints[i],
            rvecs[i],
            tvecs[i],
            mtx,
            dist
        )

        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)

        total_error += error

    print("Mean reprojection error:", total_error/len(objpoints))


# -----------------------------
# MAIN
# -----------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pattern",
        type=str,
        default="chessboard",
        help="chessboard or circle"
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=6
    )

    parser.add_argument(
        "--cols",
        type=int,
        default=9
    )

    parser.add_argument(
        "--images",
        type=int,
        default=15
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="calibrate",
        help="calibrate or project"
    )

    args = parser.parse_args()

    pattern_size = (args.rows, args.cols)

    if args.mode == "calibrate":

        calibrate_camera(
            args.images,
            pattern_size,
            args.pattern
        )

    elif args.mode == "project":

        project_virtual_point(pattern_size)


if __name__ == "__main__":
    main()