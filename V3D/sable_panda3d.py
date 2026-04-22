import argparse
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import cv2.aruco as aruco
import numpy as np
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight,
    DirectionalLight,
    LineSegs,
    LMatrix3f,
    PointLight,
    Quat,
    TextNode,
    Vec3,
    Vec4,
    WindowProperties,
)


PRIMARY_MARKER = 17
DEFAULT_CUBE_SIZE_M = 0.065
POSE_PACKET_STRUCT = struct.Struct("<4sI d 9f 3f")
POSE_MAGIC = b"SABR"
POSE_VERSION = 1
CV_TO_PANDA = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)
CV_TO_PANDA_MIRROR = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)


def create_combined_rotation(rotation_axis, additional_z_rotation=0.0):
    r1 = cv2.Rodrigues(np.array(rotation_axis, dtype=np.float32))[0]
    if additional_z_rotation != 0.0:
        r2 = cv2.Rodrigues(np.array([0.0, 0.0, additional_z_rotation], dtype=np.float32))[0]
        return r1 @ r2
    return r1


def build_cube_geometry(cube_size_m: float):
    half_size = cube_size_m / 2.0

    cube_geometry = {
        17: {"relative_rotation": np.eye(3, dtype=np.float32)},
        3: {"relative_rotation": create_combined_rotation([np.pi / 2.0, 0.0, 0.0], -np.pi / 2.0)},
        7: {"relative_rotation": create_combined_rotation([-np.pi / 2.0, 0.0, 0.0], 0.0)},
        15: {"relative_rotation": create_combined_rotation([0.0, np.pi / 2.0, 0.0], 0.0)},
        22: {"relative_rotation": create_combined_rotation([0.0, -np.pi / 2.0, 0.0], -np.pi)},
    }

    cube_markers_3d = {}
    base_corners = np.array(
        [
            [-half_size, half_size, 0.0],
            [half_size, half_size, 0.0],
            [half_size, -half_size, 0.0],
            [-half_size, -half_size, 0.0],
        ],
        dtype=np.float32,
    )

    for marker_id, geo in cube_geometry.items():
        r_rel = geo["relative_rotation"].astype(np.float32)
        t_adj = np.array([0.0, 0.0, -half_size], dtype=np.float32) + r_rel[:, 2] * half_size
        geo["relative_translation"] = t_adj.astype(np.float32)
        corners_3d = (r_rel @ base_corners.T).T + t_adj
        cube_markers_3d[marker_id] = corners_3d.astype(np.float32)

    return cube_geometry, cube_markers_3d


def load_calibration(npz_path: Path):
    data = np.load(npz_path)
    mtx_key = "mtx" if "mtx" in data.files else "camera_matrix"
    dist_key = "dist" if "dist" in data.files else "dist_coeffs"
    return data[mtx_key].astype(np.float32), data[dist_key].astype(np.float32)


def select_reference_face(corners, ids, cube_markers_3d):
    ids_flat = ids.flatten()
    if PRIMARY_MARKER in ids_flat:
        idx = int(np.where(ids_flat == PRIMARY_MARKER)[0][0])
        return PRIMARY_MARKER, idx

    best_marker_id = None
    best_idx = None
    max_area = 0.0

    for idx, (marker_corners, marker_id) in enumerate(zip(corners, ids_flat)):
        if marker_id not in cube_markers_3d:
            continue
        area = cv2.contourArea(marker_corners[0].astype(np.float32))
        if area > max_area:
            max_area = area
            best_marker_id = int(marker_id)
            best_idx = idx

    return best_marker_id, best_idx


def estimate_cube_pose_from_marker_pose(marker_id, marker_rvec, marker_tvec, cube_geometry):
    if marker_id not in cube_geometry:
        return None, None

    face_geo = cube_geometry[marker_id]
    r_cm = face_geo["relative_rotation"].astype(np.float32)
    t_cm = face_geo["relative_translation"].reshape(3, 1).astype(np.float32)

    r_cam_marker, _ = cv2.Rodrigues(marker_rvec)
    t_cam_marker = marker_tvec.reshape(3, 1).astype(np.float32)

    r_cam_cube = r_cam_marker @ r_cm.T
    t_cam_cube = t_cam_marker - (r_cam_cube @ t_cm)

    cube_rvec, _ = cv2.Rodrigues(r_cam_cube)
    return cube_rvec, t_cam_cube.astype(np.float32)


def resolve_calibration_path(primary: str, fallback: str) -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path(primary),
        script_dir / primary,
        Path(fallback),
        script_dir / fallback,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        "Could not find a camera calibration file. Checked:\n" + searched
    )


def parse_resolution(resolution: str):
    pieces = resolution.lower().split("x")
    if len(pieces) != 2:
        raise ValueError("Resolution must use WIDTHxHEIGHT format, e.g. 640x480")
    width = int(pieces[0])
    height = int(pieces[1])
    if width <= 0 or height <= 0:
        raise ValueError("Resolution values must be positive integers")
    return width, height


@dataclass
class PoseSample:
    rotation_cv: np.ndarray
    translation_cv: np.ndarray
    timestamp_s: float


class UdpPoseReceiver(threading.Thread):
    def __init__(self, listen_host: str, listen_port: int):
        super().__init__(daemon=True)
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.error_message: Optional[str] = None

        self._running = threading.Event()
        self._running.set()
        self._lock = threading.Lock()
        self._latest_pose: Optional[PoseSample] = None
        self._sock: Optional[socket.socket] = None

    def stop(self):
        self._running.clear()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass

    def get_latest_pose(self) -> Optional[PoseSample]:
        with self._lock:
            if self._latest_pose is None:
                return None
            return PoseSample(
                rotation_cv=self._latest_pose.rotation_cv.copy(),
                translation_cv=self._latest_pose.translation_cv.copy(),
                timestamp_s=self._latest_pose.timestamp_s,
            )

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock = sock
        sock.settimeout(0.2)

        try:
            sock.bind((self.listen_host, self.listen_port))
        except OSError as exc:
            self.error_message = (
                f"Could not bind UDP pose receiver at {self.listen_host}:{self.listen_port}: {exc}"
            )
            return

        while self._running.is_set():
            try:
                data, _ = sock.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                break

            if len(data) < POSE_PACKET_STRUCT.size:
                continue

            unpacked = POSE_PACKET_STRUCT.unpack_from(data)
            magic = unpacked[0]
            version = unpacked[1]
            values = unpacked[3:]

            if magic != POSE_MAGIC or version != POSE_VERSION:
                continue

            rotation_cv = np.array(values[:9], dtype=np.float32).reshape(3, 3)
            translation_cv = np.array(values[9:12], dtype=np.float32)

            sample = PoseSample(
                rotation_cv=rotation_cv,
                translation_cv=translation_cv,
                timestamp_s=time.perf_counter(),
            )
            with self._lock:
                self._latest_pose = sample


class ArucoPoseTracker(threading.Thread):
    def __init__(
        self,
        camera_index: int,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        cube_size_m: float,
        marker_size_m: float,
        cube_geometry,
        cube_markers_3d,
        width: int,
        height: int,
        capture_fps: float,
        show_camera: bool,
    ):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.cube_size_m = float(cube_size_m)
        self.marker_size_m = marker_size_m
        self.cube_geometry = cube_geometry
        self.cube_markers_3d = cube_markers_3d
        self.width = width
        self.height = height
        self.capture_fps = capture_fps
        self.show_camera = show_camera

        self.error_message: Optional[str] = None

        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
        self.detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())

        self._running = threading.Event()
        self._running.set()
        self._lock = threading.Lock()
        self._latest_pose: Optional[PoseSample] = None

    def stop(self):
        self._running.clear()

    def get_latest_pose(self) -> Optional[PoseSample]:
        with self._lock:
            if self._latest_pose is None:
                return None
            return PoseSample(
                rotation_cv=self._latest_pose.rotation_cv.copy(),
                translation_cv=self._latest_pose.translation_cv.copy(),
                timestamp_s=self._latest_pose.timestamp_s,
            )

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.error_message = f"Could not open camera index {self.camera_index}"
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        cap.set(cv2.CAP_PROP_FPS, float(self.capture_fps))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while self._running.is_set():
            ok, frame = cap.read()
            if not ok:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.detector.detectMarkers(gray)

            cube_rvec = None
            cube_tvec = None

            if ids is not None and len(ids) > 0:
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    corners,
                    self.marker_size_m,
                    self.camera_matrix,
                    self.dist_coeffs,
                )

                marker_id, marker_idx = select_reference_face(
                    corners,
                    ids,
                    self.cube_markers_3d,
                )

                if marker_id is not None and marker_idx is not None:
                    cube_rvec, cube_tvec = estimate_cube_pose_from_marker_pose(
                        marker_id,
                        rvecs[marker_idx][0],
                        tvecs[marker_idx][0],
                        self.cube_geometry,
                    )

                    if cube_rvec is not None and cube_tvec is not None:
                        r_cam_cube, _ = cv2.Rodrigues(cube_rvec)
                        t_cam_cube = cube_tvec.reshape(3, 1).astype(np.float32)
                        anchor_local = np.array(
                            [0.0, 0.0, -0.5 * self.cube_size_m],
                            dtype=np.float32,
                        ).reshape(3, 1)
                        t_cam_anchor = (r_cam_cube @ anchor_local) + t_cam_cube
                        
                        rotation_cv = r_cam_cube
                        sample = PoseSample(
                            rotation_cv=rotation_cv.astype(np.float32),
                            translation_cv=t_cam_anchor.reshape(3).astype(np.float32),
                            timestamp_s=time.perf_counter(),
                        )
                        with self._lock:
                            self._latest_pose = sample

            if self.show_camera:
                display = frame.copy()
                if ids is not None and len(ids) > 0:
                    aruco.drawDetectedMarkers(display, corners)
                if cube_rvec is not None and cube_tvec is not None:
                    cv2.drawFrameAxes(
                        display,
                        self.camera_matrix,
                        self.dist_coeffs,
                        cube_rvec,
                        cube_tvec,
                        0.06,
                    )

                cv2.imshow("Aruco tracker", display)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    self.stop()
                    break

        cap.release()
        if self.show_camera:
            cv2.destroyWindow("Aruco tracker")


class SaberGame(ShowBase):
    def __init__(
        self,
        tracker,
        pose_scale: float,
        pos_smoothing: float,
        lost_timeout_s: float,
        tracking_label: str,
    ):
        super().__init__()

        self.tracker = tracker
        self.tracking_label = tracking_label
        self.pose_scale = float(pose_scale)
        self.pos_smoothing = float(np.clip(pos_smoothing, 0.0, 1.0))
        self.lost_timeout_s = float(lost_timeout_s)
        # Pose base del sable unico centrado en espacio local de camara (first person).
        self.hand_anchor_local = Vec3(0.0, 1.05, -0.42)
        self.hand_motion_gain = 0.75
        self.saber_rest_hpr = Vec3(0.0, -10.0, 0.0)
        self.saber_rest_quat = Quat()
        self.saber_rest_quat.setHpr(self.saber_rest_hpr)

        self.filtered_pos: Optional[Vec3] = None
        self.reference_translation_cv: Optional[np.ndarray] = None
        self.reference_rotation_cv: Optional[np.ndarray] = None
        self.last_pose_time_s = 0.0
        self.has_pose = False

        self.disableMouse()
        self._setup_window()
        self._setup_scene()

        self.accept("escape", self._request_exit)
        self.accept("r", self._recenter_pose_reference)
        self.taskMgr.add(self._update_pose_task, "update-pose-task")

    def _setup_window(self):
        props = WindowProperties()
        props.setTitle("Aruco Lightsaber - Panda3D")
        props.setSize(1280, 720)
        self.win.requestProperties(props)

        self.setFrameRateMeter(True)
        self.camLens.setNearFar(0.01, 100.0)
        self.camLens.setFov(100.0)

        # Camara de primera persona (origen del jugador mirando al frente).
        self.camera.setPos(0.0, 0.0, 1.65)
        self.camera.setHpr(0.0, 0.0, 0.0)

    def _setup_scene(self):
        self.setBackgroundColor(0.06, 0.08, 0.12, 1.0)

        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.35, 0.35, 0.38, 1.0))
        ambient_np = self.render.attachNewNode(ambient)
        self.render.setLight(ambient_np)

        key_light = DirectionalLight("key-light")
        key_light.setColor(Vec4(0.72, 0.72, 0.74, 1.0))
        key_light_np = self.render.attachNewNode(key_light)
        key_light_np.setHpr(-30.0, -35.0, 0.0)
        self.render.setLight(key_light_np)

        floor = self.loader.loadModel("models/misc/rgbCube")
        floor.reparentTo(self.render)
        floor.setScale(6.0, 45.0, 0.008)
        floor.setPos(0.0, 18.0, -0.62)
        floor.setColor(0.22, 0.24, 0.28, 1.0)
        floor.setLightOff()

        axis = LineSegs("world-axis")
        axis.setThickness(4.0)
        axis.setColor(1.0, 0.2, 0.2, 1.0)
        axis.moveTo(0.0, 0.0, 0.0)
        axis.drawTo(0.9, 0.0, 0.0)
        axis.setColor(0.2, 1.0, 0.2, 1.0)
        axis.moveTo(0.0, 0.0, 0.0)
        axis.drawTo(0.0, 0.9, 0.0)
        axis.setColor(0.2, 0.6, 1.0, 1.0)
        axis.moveTo(0.0, 0.0, 0.0)
        axis.drawTo(0.0, 0.0, 0.9)
        axis_np = self.render.attachNewNode(axis.create())
        axis_np.setPos(0.0, 3.0, -0.55)

        # El sable cuelga de la camara para sentirse como mano en primera persona.
        self.saber_root = self.camera.attachNewNode("saber-root")

        hilt = self.loader.loadModel("models/misc/rgbCube")
        hilt.reparentTo(self.saber_root)
        hilt.setScale(0.07, 0.34, 0.07)
        hilt.setPos(0.0, 0.21, 0.0)
        hilt.setColor(0.22, 0.22, 0.22, 1.0)

        blade_glow = self.loader.loadModel("models/misc/rgbCube")
        blade_glow.reparentTo(self.saber_root)
        blade_glow.setScale(0.07, 1.38, 0.07)
        blade_glow.setPos(0.0, 1.74, 0.0)
        blade_glow.setColor(0.22, 0.86, 1.0, 0.26)
        blade_glow.setTransparency(True)
        blade_glow.setLightOff()
        blade_glow.setDepthWrite(False)

        blade_core = self.loader.loadModel("models/misc/rgbCube")
        blade_core.reparentTo(self.saber_root)
        blade_core.setScale(0.03, 1.28, 0.03)
        blade_core.setPos(0.0, 1.74, 0.0)
        blade_core.setColor(0.74, 0.95, 1.0, 0.98)
        blade_core.setTransparency(True)
        blade_core.setLightOff()
        self.blade_core = blade_core
        self.blade_glow = blade_glow

        blade_tip = self.saber_root.attachNewNode("blade-tip")
        blade_tip.setPos(0.0, 3.00, 0.0)

        blade_mid = self.saber_root.attachNewNode("blade-mid")
        blade_mid.setPos(0.0, 1.65, 0.0)

        blade_light = PointLight("blade-light")
        blade_light.setColor(Vec4(2.2, 2.6, 3.0, 1.0))
        blade_light.setAttenuation(Vec3(1.0, 0.0, 0.05))
        blade_light_np = blade_tip.attachNewNode(blade_light)
        self.render.setLight(blade_light_np)

        blade_fill = PointLight("blade-fill")
        blade_fill.setColor(Vec4(0.9, 1.4, 1.7, 1.0))
        blade_fill.setAttenuation(Vec3(1.0, 0.0, 0.09))
        blade_fill_np = blade_mid.attachNewNode(blade_fill)
        self.render.setLight(blade_fill_np)

        self.saber_root.setPos(self.hand_anchor_local)
        self.saber_root.setHpr(self.saber_rest_hpr)
        self.saber_root.show()

        self.status_text = OnscreenText(
            text=f"TRACKING: waiting for {self.tracking_label}",
            pos=(-1.30, 0.92),
            scale=0.055,
            fg=(0.95, 0.97, 1.0, 1.0),
            align=TextNode.ALeft,
            mayChange=True,
        )
        self.help_text = OnscreenText(
            text="ESC: quit | R: recenter hand | Start id.py in UDP mode",
            pos=(-1.30, 0.85),
            scale=0.04,
            fg=(0.78, 0.84, 0.92, 1.0),
            align=TextNode.ALeft,
        )

    def _cv_local_offset_from_translation(self, translation_cv: np.ndarray) -> Vec3:
        if self.reference_translation_cv is None:
            self.reference_translation_cv = translation_cv.astype(np.float32).copy()

        delta_cv = translation_cv.astype(np.float32) - self.reference_translation_cv
        mapped = CV_TO_PANDA @ delta_cv
        # La vista del tracking se muestra espejada al usuario en id.py;
        # se invierte X para que el control en primera persona sea intuitivo.
        mapped[0] = -mapped[0]
        mapped = mapped * self.pose_scale * self.hand_motion_gain
        x = float(np.clip(mapped[0], -0.55, 0.55))
        y = float(np.clip(mapped[1], -0.35, 0.75))
        z = float(np.clip(mapped[2], -0.45, 0.45))
        return Vec3(x, y, z)

    def _cv_relative_quat(self, rotation_cv: np.ndarray) -> Quat:
        if self.reference_rotation_cv is None:
            self.reference_rotation_cv = rotation_cv.astype(np.float32).copy()

        rotation_rel_cv = rotation_cv @ self.reference_rotation_cv.T
        return self._cv_rotation_to_panda_quat(rotation_rel_cv)

    def _cv_rotation_to_panda_quat(self, rotation_cv: np.ndarray) -> Quat:
        # Usar la misma base espejada que en traslacion para mantener coherencia de ejes.
        rotation_panda = CV_TO_PANDA_MIRROR @ rotation_cv @ CV_TO_PANDA_MIRROR.T
        matrix = LMatrix3f(
            float(rotation_panda[0, 0]),
            float(rotation_panda[0, 1]),
            float(rotation_panda[0, 2]),
            float(rotation_panda[1, 0]),
            float(rotation_panda[1, 1]),
            float(rotation_panda[1, 2]),
            float(rotation_panda[2, 0]),
            float(rotation_panda[2, 1]),
            float(rotation_panda[2, 2]),
        )
        quat = Quat()
        quat.setFromMatrix(matrix)
        return quat

    def _update_pose_task(self, task):
        pose = self.tracker.get_latest_pose()
        now_s = time.perf_counter()

        if pose is not None:
            target_pos = self.hand_anchor_local + self._cv_local_offset_from_translation(
                pose.translation_cv
            )
            relative_quat = self._cv_relative_quat(pose.rotation_cv)
            # Componer en cuaterniones evita inversiones/intercambios de ejes por Euler.
            target_quat = relative_quat * self.saber_rest_quat

            if self.filtered_pos is None:
                self.filtered_pos = target_pos
            else:
                alpha = self.pos_smoothing
                self.filtered_pos = (self.filtered_pos * (1.0 - alpha)) + (target_pos * alpha)

            self.saber_root.setPos(self.filtered_pos)
            self.saber_root.setQuat(target_quat)
            self.saber_root.show()
            self.saber_root.setColorScale(1.0, 1.0, 1.0, 1.0)
            self.blade_core.setColorScale(1.8, 1.8, 1.9, 1.0)
            self.blade_glow.setColorScale(2.4, 2.4, 2.6, 1.0)
            self.status_text.setText("TRACKING: pose locked")
            self.last_pose_time_s = pose.timestamp_s
            self.has_pose = True
        elif not self.has_pose:
            self.status_text.setText(f"TRACKING: no {self.tracking_label} yet")
        elif (now_s - self.last_pose_time_s) > self.lost_timeout_s:
            self.status_text.setText("TRACKING: pose lost, holding last transform")
            self.saber_root.setColorScale(0.65, 0.65, 0.80, 1.0)
            self.blade_core.setColorScale(0.9, 0.9, 1.05, 1.0)
            self.blade_glow.setColorScale(1.1, 1.1, 1.3, 1.0)

        return task.cont

    def _recenter_pose_reference(self):
        # Solo se recentra la rotación; la traslación permanece absoluta desde el primer tracking
        self.reference_rotation_cv = None
        self.filtered_pos = None
        self.status_text.setText("TRACKING: recenter requested, hold marker steady")

    def _request_exit(self):
        self.tracker.stop()
        self.userExit()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Panda3D lightsaber controlled by ArUco cube pose"
    )
    parser.add_argument(
        "--pose-source",
        type=str,
        choices=("udp", "camera"),
        default="udp",
        help="Pose source for the saber: udp (recommended) or camera",
    )
    parser.add_argument(
        "--listen-host",
        type=str,
        default="127.0.0.1",
        help="UDP host/IP to bind when pose-source=udp",
    )
    parser.add_argument(
        "--listen-port",
        type=int,
        default=5005,
        help="UDP port to bind when pose-source=udp",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument(
        "--calibration",
        type=str,
        default="camera_calibration.npz",
        help="Primary calibration file",
    )
    parser.add_argument(
        "--fallback-calibration",
        type=str,
        default="calibracion_camara.npz",
        help="Fallback calibration file",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="640x480",
        help="Capture resolution in WIDTHxHEIGHT format",
    )
    parser.add_argument(
        "--capture-fps",
        type=float,
        default=60.0,
        help="Requested camera fps",
    )
    parser.add_argument(
        "--cube-size-m",
        type=float,
        default=DEFAULT_CUBE_SIZE_M,
        help="Cube side size in meters",
    )
    parser.add_argument(
        "--marker-size-m",
        type=float,
        default=DEFAULT_CUBE_SIZE_M,
        help="Detected marker side size in meters",
    )
    parser.add_argument(
        "--pose-scale",
        type=float,
        default=4.0,
        help="Scale from real meters to Panda world units",
    )
    parser.add_argument(
        "--pos-smoothing",
        type=float,
        default=0.35,
        help="Position smoothing alpha in [0,1]",
    )
    parser.add_argument(
        "--lost-timeout",
        type=float,
        default=0.25,
        help="Hide the saber if pose is lost for this duration (seconds)",
    )
    parser.add_argument(
        "--show-camera",
        action="store_true",
        help="Show OpenCV debug window",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.pose_source == "udp":
        tracker = UdpPoseReceiver(args.listen_host, args.listen_port)
        tracking_label = f"UDP pose {args.listen_host}:{args.listen_port}"
    else:
        width, height = parse_resolution(args.resolution)
        calibration_path = resolve_calibration_path(args.calibration, args.fallback_calibration)
        camera_matrix, dist_coeffs = load_calibration(calibration_path)
        cube_geometry, cube_markers_3d = build_cube_geometry(args.cube_size_m)

        tracker = ArucoPoseTracker(
            camera_index=args.camera,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            cube_size_m=args.cube_size_m,
            marker_size_m=args.marker_size_m,
            cube_geometry=cube_geometry,
            cube_markers_3d=cube_markers_3d,
            width=width,
            height=height,
            capture_fps=args.capture_fps,
            show_camera=args.show_camera,
        )
        tracking_label = f"camera pose (camera {args.camera})"

    tracker.start()

    time.sleep(0.2)
    if tracker.error_message:
        print(tracker.error_message)
        tracker.stop()
        tracker.join(timeout=1.0)
        return 1

    app = SaberGame(
        tracker=tracker,
        pose_scale=args.pose_scale,
        pos_smoothing=args.pos_smoothing,
        lost_timeout_s=args.lost_timeout,
        tracking_label=tracking_label,
    )

    try:
        app.run()
    finally:
        tracker.stop()
        tracker.join(timeout=1.0)

    return 0


if __name__ == "__main__":
    sys.exit(main())