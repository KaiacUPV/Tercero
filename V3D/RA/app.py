# Auto-extracted from id.py without modifying the original file.
"""Aplicacion OpenCV del sable en realidad aumentada.

Abre la camara, detecta el cubo ArUco, proyecta el sable 2D sobre el frame y
coordina teclado, voz y audio.
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

from .audio import *
from .constants import *
from .geometry import *
from .pose import *
from .voice import VoiceCommandListener


def run_app() -> int:
    project_root = Path(__file__).resolve().parent.parent
    calibration_dir = project_root / "calibracion"
    calibration_dir_alt = project_root / "calibración"
    audio_dir = project_root / "audios"

    def _first_existing_path(candidates: list[Path], *, required: bool = False, label: str = "archivo") -> Path:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        if required:
            print(f"[Init] No se encontro {label}. Rutas probadas:")
            for candidate in candidates:
                print(f"  - {candidate}")
            raise SystemExit(1)
        return candidates[0]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Camera] No se pudo abrir la camara 0. Cierra otras apps que usen la webcam y reintenta.")
        raise SystemExit(1)

    # Calibración: intenta una ruta "habitual" y si no, busca local junto al script.
    calibration_path = _first_existing_path(
        [
            calibration_dir / "camera_calibration.npz",
            calibration_dir / "calibracion_camara.npz",
            calibration_dir_alt / "camera_calibration.npz",
            calibration_dir_alt / "calibracion_camara.npz",
            project_root / "camera_calibration.npz",
            project_root / "calibracion_camara.npz",
        ],
        required=True,
        label="archivo de calibracion (.npz)",
    )
    camera_matrix, dist_coeffs = load_calibration(calibration_path)

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
    detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())
    pose_smoother = PoseSmoother(SMOOTH_TRANSLATION_ALPHA, SMOOTH_ROTATION_ALPHA)
    laser_target_on = False
    laser_power = 0.0
    last_frame_time = time.perf_counter()
    last_toggle_time = 0.0
    saber_on_sound_path = _first_existing_path(
        [audio_dir / SABER_ON_SOUND_FILE, project_root / SABER_ON_SOUND_FILE],
        required=True,
        label=SABER_ON_SOUND_FILE,
    )
    saber_loop_sound_path = _first_existing_path(
        [audio_dir / SABER_LOOP_SOUND_FILE, project_root / SABER_LOOP_SOUND_FILE],
        required=True,
        label=SABER_LOOP_SOUND_FILE,
    )
    saber_off_sound_path = _first_existing_path(
        [audio_dir / SABER_OFF_SOUND_FILE, project_root / SABER_OFF_SOUND_FILE],
        required=True,
        label=SABER_OFF_SOUND_FILE,
    )
    audio_backend = init_audio_backend(saber_on_sound_path, saber_loop_sound_path, saber_off_sound_path)
    loop_start_delay_s = SABER_LOOP_START_DELAY_S
    print(f"[Audio] Backend activo: {audio_backend}")
    if audio_backend == "none":
        print("[Audio] Sin backend. Instala PyOpenAL (pip install PyOpenAL) o pygame.")
    print(f"[App] PID: {os.getpid()} | ESC: salir | Espacio/RGB/I: teclado | Voz activa si modelo disponible")

    voice_listener = None
    # Cooldown por comando: evita que Vosk repita el mismo comando muchas veces seguidas
    # por resultados parciales o por eco/ruido.
    voice_last_command_time = {
        "toggle_power": 0.0,
        "toggle_flip": 0.0,
        "color_r": 0.0,
        "color_g": 0.0,
        "color_b": 0.0,
    }

    if VOICE_CONTROL_ENABLED:
        voice_model_path = project_root / VOICE_MODEL_DIR
        voice_listener = VoiceCommandListener(
            model_path=voice_model_path,
            sample_rate=VOICE_SAMPLE_RATE,
            block_size=VOICE_BLOCK_SIZE,
            device_hint=VOICE_INPUT_DEVICE_HINT,
        )
        if voice_listener.start():
            time.sleep(0.15)
            if voice_listener.active_device_name:
                if voice_listener.active_sample_rate is not None:
                    print(
                        f"[Voice] Entrada activa: {voice_listener.active_device_name} "
                        f"@ {voice_listener.active_sample_rate} Hz"
                    )
                else:
                    print(f"[Voice] Entrada activa: {voice_listener.active_device_name}")
            print("[Voice] Control activo: di 'espacio', 'rojo/verde/azul' o 'invertir'.")
        else:
            print(voice_listener.error_message)
            voice_listener = None

    loop_start_due_time = None
    loop_playing = False
    camera_horizontal_flip = CAMERA_HORIZONTAL_FLIP_DEFAULT
    current_saber_color_key = DEFAULT_SABER_COLOR_KEY
    last_flip_toggle_time = 0.0
    beat_blocks = []
    beat_score = 0
    beat_misses = 0
    beat_last_spawn_time = time.perf_counter()
    camera_read_fail_count = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if FULLSCREEN_PREVIEW:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Bucle principal: procesa input (voz/teclado), tracking y render de preview.
    while True:
        ret, frame = cap.read()
        if not ret:
            camera_read_fail_count += 1
            if camera_read_fail_count == 1:
                print("[Camera] No se pudo leer frame de la camara. Reintentando...")
            if camera_read_fail_count >= 45:
                print("[Camera] Lectura fallida continua. Cerrando la app para liberar recursos.")
                break
            time.sleep(0.01)
            continue
        camera_read_fail_count = 0

        now = time.perf_counter()
        dt = float(np.clip(now - last_frame_time, 0.0, 0.10))
        last_frame_time = now

        # Animacion de encendido/apagado del sable controlada por estado objetivo.
        if laser_target_on:
            laser_power = min(1.0, laser_power + (dt / max(LASER_ON_DURATION_S, 1e-6)))
        else:
            laser_power = max(0.0, laser_power - (dt / max(LASER_OFF_DURATION_S, 1e-6)))

        # Audio: sonido de encendido una vez y loop continuo mientras el sable este activo.
        if laser_target_on and not loop_playing and loop_start_due_time is not None and now >= loop_start_due_time:
            loop_playing = play_saber_loop_sound(saber_loop_sound_path)

        if BEAT_MODE_ENABLED:
            # Avanza y depura la lista de bloques activos.
            beat_blocks, expired_misses = advance_beat_blocks(beat_blocks, dt)
            beat_misses += expired_misses

            # Spawnea nuevos bloques a ritmo fijo. Usamos un while para “ponernos al día”
            # si hubo un frame lento.
            while (now - beat_last_spawn_time) >= BEAT_SPAWN_INTERVAL_S and len(beat_blocks) < BEAT_BLOCK_MAX_ACTIVE:
                beat_blocks.append(create_beat_block())
                beat_last_spawn_time += BEAT_SPAWN_INTERVAL_S

        if voice_listener is not None:
            if voice_listener.error_message:
                # Si el hilo de voz falló, lo apagamos para que el resto del tracking siga.
                print(voice_listener.error_message)
                voice_listener.stop()
                voice_listener = None
            else:
                # Consumimos todos los comandos ya reconocidos en esta iteración.
                for voice_command, spoken_text in voice_listener.pop_commands():
                    command_now = time.perf_counter()
                    # Cooldown por tipo de comando (evita "spam" si Vosk repite la palabra).
                    if (command_now - voice_last_command_time.get(voice_command, 0.0)) < VOICE_COMMAND_COOLDOWN_S:
                        continue
                    voice_last_command_time[voice_command] = command_now

                    if voice_command == "toggle_power":
                        # Encendido/apagado del sable (afecta animación y audio).
                        laser_target_on = not laser_target_on
                        if laser_target_on:
                            stop_all_sounds()
                            loop_playing = False
                            play_saber_on_sound(saber_on_sound_path)
                            loop_start_due_time = command_now + max(0.0, loop_start_delay_s)
                            print(f"[Voice] '{spoken_text}' -> sable ON")
                        else:
                            stop_all_sounds()
                            loop_playing = False
                            play_saber_off_sound(saber_off_sound_path)
                            loop_start_due_time = None
                            print(f"[Voice] '{spoken_text}' -> sable OFF")
                        last_toggle_time = command_now

                    elif voice_command == "toggle_flip":
                        # Espejo horizontal: útil si la cámara está montada invertida.
                        camera_horizontal_flip = not camera_horizontal_flip
                        print(
                            f"[Voice] '{spoken_text}' -> inversion horizontal: "
                            f"{'ON' if camera_horizontal_flip else 'OFF'}"
                        )
                        last_flip_toggle_time = command_now

                    elif voice_command.startswith("color_"):
                        # Cambia preset de color (r/g/b).
                        color_key = voice_command[-1]
                        if color_key in SABER_COLOR_PRESETS:
                            current_saber_color_key = color_key
                            color_name = SABER_COLOR_PRESETS[color_key]["name"]
                            print(f"[Voice] '{spoken_text}' -> color {color_name}")

        # --- Detección ArUco ---
        # Pasamos a grayscale por rendimiento y robustez.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        marker_labels = []

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners)
            
            # Pose (rvec/tvec) de cada marcador detectado.
            # OpenCV devuelve rvec/tvec por marcador respecto a la cámara.
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, camera_matrix, dist_coeffs)
            # Etiquetas opcionales de depuracion (sin mostrar IDs de marcadores).
            for i, marker_corners in enumerate(corners):
                cx, cy = int(marker_corners[0][:, 0].mean()), int(marker_corners[0][:, 1].mean())
                if SHOW_FACE_INFO:
                    x, y, z = tvecs[i][0]
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    marker_labels.append({
                        "cx": cx,
                        "cy": cy,
                        "text": f"C({cx},{cy}) X:{x:+.3f} Y:{y:+.3f} Z:{z:+.3f}m",
                    })

            # Elegir la cara de referencia: 17 si está visible, si no, la más visible.
            # Esto reduce saltos cuando la cara principal aparece/desaparece.
            best_marker_id, best_idx = select_reference_face(corners, ids)

            # Estimar pose del cubo usando SOLO esa cara.
            # Salida: rvec/tvec del cubo (no del marcador) en marco de cámara.
            prism_rvec, prism_tvec = None, None
            control_tvec = None
            
            if best_marker_id is not None and best_idx is not None:
                # Convertimos pose de la cara elegida a pose del cubo completo.
                prism_rvec, prism_tvec = estimate_cube_pose_from_marker_pose(
                    best_marker_id,
                    rvecs[best_idx][0],
                    tvecs[best_idx][0]
                )
                
                if prism_rvec is not None and prism_tvec is not None:
                    # Calcula ángulo de la normal del cubo respecto a la cámara.
                    # Cuando el marcador está "de canto" (paralelo a la lente), el jitter sube.
                    R_measured, _ = cv2.Rodrigues(prism_rvec)
                    z_normal = R_measured[:, 2]
                    angle_deg = np.degrees(np.arccos(np.clip(z_normal[2], -1.0, 1.0)))
                    adaptive_t_alpha, adaptive_r_alpha = get_adaptive_smoothing_alphas(angle_deg)

                    # Suavizado adaptativo: más fuerte en casos inestables, mínimo en el resto.
                    prism_rvec, prism_tvec = pose_smoother.update(
                        prism_rvec,
                        prism_tvec,
                        adaptive_t_alpha,
                        adaptive_r_alpha,
                    )
                    # Punto de control estable: centro del cubo.
                    control_tvec = get_control_anchor_tvec(prism_rvec, prism_tvec)

            if prism_rvec is not None and prism_tvec is not None and control_tvec is not None:
                # Matriz de rotación del cubo para proyectar geometría 3D al frame (cámara).
                R, _ = cv2.Rodrigues(prism_rvec)
                # (UDP eliminado) Si necesitas exportar la pose a otro proceso,
                # se recomienda hacerlo desde el consumidor final (por ejemplo, openglsable.py).

                # Sable láser anclado a la cara 17; la pose se obtiene de la cara de referencia visible.
                # `get_prism_points_with_offset` aplica un offset (borde blanco ArUco) cuando usamos
                # caras laterales para que la hoja “nazca” coherentemente.
                prism_points_adjusted = get_prism_points_with_offset(best_marker_id)
                saber_color = SABER_COLOR_PRESETS.get(
                    current_saber_color_key,
                    SABER_COLOR_PRESETS[DEFAULT_SABER_COLOR_KEY],
                )
                
                # Geometria del sable: mango estilizado segmentado + hoja luminosa.
                # center_base/center_top: extremos del prisma en el marco del cubo.
                center_base = np.mean(prism_points_adjusted[:4], axis=0)
                center_top = np.mean(prism_points_adjusted[4:], axis=0)
                
                # Ejes para generar círculos (cylinder/frustum) alrededor del mango.
                # Importante: estos ejes viven en el marco del cubo (antes de la proyección).
                face_R = CUBE_GEOMETRY[PRIMARY_MARKER]["relative_rotation"]
                axis_x = face_R[:, 0]
                axis_y = face_R[:, 1]

                # Vector de dirección de la hoja (en marco del cubo).
                blade_vec = center_top - center_base
                blade_len = float(np.linalg.norm(blade_vec))
                if blade_len > 1e-6:
                    blade_dir = blade_vec / blade_len

                    # Seguridad: evita que la hoja cruce el plano de la cámara (z ~ 0).
                    # Si el modelo se proyecta “detrás” de la cámara o muy cerca, OpenCV puede
                    # devolver proyecciones enormes o degeneradas que parecen un cuelgue.
                    center_base_cam = (R @ center_base.reshape(3, 1)) + prism_tvec.reshape(3, 1)
                    blade_dir_cam = R @ blade_dir.reshape(3, 1)
                    safe_blade_len = blade_len
                    dir_z = float(blade_dir_cam[2, 0])
                    base_z = float(center_base_cam[2, 0])
                    if dir_z < -1e-6:
                        max_len_before_camera = (base_z - LASER_MIN_DRAW_DEPTH_M) / (-dir_z)
                        safe_blade_len = min(safe_blade_len, max(0.0, max_len_before_camera))

                    radius_outer = HALF_SIZE * LASER_OUTER_RADIUS_SCALE
                    radius_inner = HALF_SIZE * LASER_INNER_RADIUS_SCALE
                    handle_radius = HALF_SIZE * HANDLE_RADIUS_SCALE
                    handle_cap_radius = HALF_SIZE * HANDLE_CAP_RADIUS_SCALE
                    emitter_radius = HALF_SIZE * EMITTER_RADIUS_SCALE

                    handle_end = center_base - blade_dir * HANDLE_LENGTH_M
                    split_1, split_2 = HANDLE_PROFILE_SPLITS
                    r0, r1, r2, r3 = HANDLE_PROFILE_RADIUS_SCALES
                    section_1_end = handle_end + blade_dir * (HANDLE_LENGTH_M * split_1)
                    section_2_end = handle_end + blade_dir * (HANDLE_LENGTH_M * split_2)

                    hilt_section_points = [
                        build_frustum_points(
                            handle_end,
                            section_1_end,
                            axis_x,
                            axis_y,
                            handle_radius * r0,
                            handle_radius * r1,
                            LASER_CYLINDER_SIDES,
                        ),
                        build_frustum_points(
                            section_1_end,
                            section_2_end,
                            axis_x,
                            axis_y,
                            handle_radius * r1,
                            handle_radius * r2,
                            LASER_CYLINDER_SIDES,
                        ),
                        build_frustum_points(
                            section_2_end,
                            center_base,
                            axis_x,
                            axis_y,
                            handle_radius * r2,
                            handle_radius * r3,
                            LASER_CYLINDER_SIDES,
                        ),
                    ]

                    handle_cap_end = handle_end - blade_dir * HANDLE_CAP_LENGTH_M
                    handle_cap_points = build_frustum_points(
                        handle_cap_end,
                        handle_end,
                        axis_x,
                        axis_y,
                        handle_cap_radius,
                        handle_radius * 1.05,
                        LASER_CYLINDER_SIDES,
                    )

                    emitter_end = center_base + blade_dir * EMITTER_LENGTH_M
                    emitter_points = build_frustum_points(
                        center_base,
                        emitter_end,
                        axis_x,
                        axis_y,
                        emitter_radius * 1.06,
                        emitter_radius * 0.88,
                        LASER_CYLINDER_SIDES,
                    )

                    detail_ring_points = []
                    ring_radius = handle_radius * HANDLE_DETAIL_RING_RADIUS_SCALE
                    ring_half_len = HANDLE_DETAIL_RING_LENGTH_M * 0.5
                    for ring_idx in range(HANDLE_DETAIL_RING_COUNT):
                        ring_ratio = HANDLE_DETAIL_RING_START_RATIO + (ring_idx * HANDLE_DETAIL_RING_STEP_RATIO)
                        if ring_ratio <= 0.0 or ring_ratio >= 1.0:
                            continue

                        ring_center = handle_end + blade_dir * (HANDLE_LENGTH_M * ring_ratio)
                        ring_start = ring_center - blade_dir * ring_half_len
                        ring_end = ring_center + blade_dir * ring_half_len
                        detail_ring_points.append(
                            build_cylinder_points(
                                ring_start,
                                ring_end,
                                axis_x,
                                axis_y,
                                ring_radius,
                                LASER_CYLINDER_SIDES,
                            )
                        )

                    button_points = []
                    button_axis = axis_x.astype(np.float32)
                    button_radius = HALF_SIZE * HANDLE_BUTTON_RADIUS_SCALE
                    for button_ratio in HANDLE_BUTTON_POSITIONS:
                        button_center = handle_end + blade_dir * (HANDLE_LENGTH_M * float(button_ratio))
                        button_start = button_center + button_axis * (handle_radius * 0.92)
                        button_end = button_start + button_axis * HANDLE_BUTTON_HEIGHT_M
                        button_points.append(
                            build_cylinder_points(
                                button_start,
                                button_end,
                                blade_dir,
                                axis_y,
                                button_radius,
                                LASER_CYLINDER_SIDES,
                            )
                        )

                    fin_points = []
                    fin_radius = HALF_SIZE * HANDLE_FIN_RADIUS_SCALE
                    for fin_ratio in HANDLE_FIN_POSITIONS:
                        fin_center = handle_end + blade_dir * (HANDLE_LENGTH_M * float(fin_ratio))
                        for fin_idx in range(HANDLE_FIN_COUNT):
                            angle = (2.0 * np.pi * fin_idx) / HANDLE_FIN_COUNT
                            fin_normal = (axis_x * np.cos(angle)) + (axis_y * np.sin(angle))
                            fin_normal = fin_normal.astype(np.float32)
                            fin_normal_norm = float(np.linalg.norm(fin_normal))
                            if fin_normal_norm <= 1e-6:
                                continue
                            fin_normal /= fin_normal_norm

                            fin_start = fin_center + fin_normal * (handle_radius * 0.95)
                            fin_end = fin_start + fin_normal * HANDLE_FIN_HEIGHT_M
                            fin_tangent = np.cross(blade_dir, fin_normal).astype(np.float32)
                            fin_tangent_norm = float(np.linalg.norm(fin_tangent))
                            if fin_tangent_norm <= 1e-6:
                                fin_tangent = axis_y.astype(np.float32)
                            else:
                                fin_tangent /= fin_tangent_norm

                            fin_points.append(
                                build_cylinder_points(
                                    fin_start,
                                    fin_end,
                                    blade_dir,
                                    fin_tangent,
                                    fin_radius,
                                    LASER_CYLINDER_SIDES,
                                )
                            )

                    pts_hilt_sections = []
                    for hilt_points in hilt_section_points:
                        pts_hilt, _ = cv2.projectPoints(hilt_points, prism_rvec, prism_tvec, camera_matrix, dist_coeffs)
                        pts_hilt_sections.append(pts_hilt)

                    pts_handle_cap, _ = cv2.projectPoints(handle_cap_points, prism_rvec, prism_tvec, camera_matrix, dist_coeffs)
                    pts_emitter, _ = cv2.projectPoints(emitter_points, prism_rvec, prism_tvec, camera_matrix, dist_coeffs)

                    pts_detail_rings = []
                    for ring_points in detail_ring_points:
                        pts_ring, _ = cv2.projectPoints(ring_points, prism_rvec, prism_tvec, camera_matrix, dist_coeffs)
                        pts_detail_rings.append(pts_ring)

                    pts_buttons = []
                    for button_pts in button_points:
                        pts_button, _ = cv2.projectPoints(button_pts, prism_rvec, prism_tvec, camera_matrix, dist_coeffs)
                        pts_buttons.append(pts_button)

                    pts_fins = []
                    for fin_pts in fin_points:
                        pts_fin, _ = cv2.projectPoints(fin_pts, prism_rvec, prism_tvec, camera_matrix, dist_coeffs)
                        pts_fins.append(pts_fin)

                    hilt_hulls = []
                    hilt_safe = True
                    for pts_hilt in pts_hilt_sections:
                        if not projected_points_are_safe(pts_hilt, MAX_PROJECTED_COORD_ABS_PX):
                            hilt_safe = False
                            break
                        hilt_hulls.append(cv2.convexHull(pts_hilt.reshape(-1, 2).astype(np.int32)))

                    cap_safe = projected_points_are_safe(pts_handle_cap, MAX_PROJECTED_COORD_ABS_PX)
                    emitter_safe = projected_points_are_safe(pts_emitter, MAX_PROJECTED_COORD_ABS_PX)

                    ring_hulls = []
                    rings_safe = True
                    for pts_ring in pts_detail_rings:
                        if not projected_points_are_safe(pts_ring, MAX_PROJECTED_COORD_ABS_PX):
                            rings_safe = False
                            break
                        ring_hulls.append(cv2.convexHull(pts_ring.reshape(-1, 2).astype(np.int32)))

                    button_hulls = []
                    buttons_safe = True
                    for pts_button in pts_buttons:
                        if not projected_points_are_safe(pts_button, MAX_PROJECTED_COORD_ABS_PX):
                            buttons_safe = False
                            break
                        button_hulls.append(cv2.convexHull(pts_button.reshape(-1, 2).astype(np.int32)))

                    fin_hulls = []
                    fins_safe = True
                    for pts_fin in pts_fins:
                        if not projected_points_are_safe(pts_fin, MAX_PROJECTED_COORD_ABS_PX):
                            fins_safe = False
                            break
                        fin_hulls.append(cv2.convexHull(pts_fin.reshape(-1, 2).astype(np.int32)))

                    if hilt_safe and cap_safe and emitter_safe and rings_safe and buttons_safe and fins_safe:
                        hull_handle_cap = cv2.convexHull(pts_handle_cap.reshape(-1, 2).astype(np.int32))
                        hull_emitter = cv2.convexHull(pts_emitter.reshape(-1, 2).astype(np.int32))

                        hilt_fill_palette = [
                            (30, 30, 42),
                            (20, 20, 28),
                            (36, 36, 50),
                        ]
                        hilt_edge_palette = saber_color["hilt_section_edges"]
                        for section_idx, section_hull in enumerate(hilt_hulls):
                            fill_color = hilt_fill_palette[section_idx % len(hilt_fill_palette)]
                            edge_color = hilt_edge_palette[section_idx % len(hilt_edge_palette)]
                            cv2.fillConvexPoly(frame, section_hull, fill_color)
                            cv2.polylines(frame, [section_hull], True, edge_color, 2)

                        cv2.fillConvexPoly(frame, hull_handle_cap, (18, 18, 24))
                        cv2.fillConvexPoly(frame, hull_emitter, (62, 62, 78))
                        cv2.polylines(frame, [hull_handle_cap], True, saber_color["hilt_cap_edge"], 2)
                        cv2.polylines(frame, [hull_emitter], True, saber_color["hilt_emitter_edge"], 2)

                        for ring_hull in ring_hulls:
                            cv2.fillConvexPoly(frame, ring_hull, saber_color["ring_fill"])
                            cv2.polylines(frame, [ring_hull], True, saber_color["ring_edge"], 1)

                        button_colors = saber_color["button_colors"]
                        for button_idx, button_hull in enumerate(button_hulls):
                            base_color = button_colors[button_idx % len(button_colors)]
                            cv2.fillConvexPoly(frame, button_hull, base_color)
                            cv2.polylines(frame, [button_hull], True, (235, 240, 245), 1)

                        for fin_hull in fin_hulls:
                            cv2.fillConvexPoly(frame, fin_hull, (16, 16, 22))
                            cv2.polylines(frame, [fin_hull], True, saber_color["fin_edge"], 1)

                    if laser_power > LASER_MIN_VISIBLE_POWER and safe_blade_len > 1e-5:
                        animated_blade_len = min(blade_len * laser_power, safe_blade_len)
                        if animated_blade_len > 1e-5:
                            blade_tip = center_base + (blade_dir * animated_blade_len)

                            layer_outer = build_cylinder_points(
                                center_base,
                                blade_tip,
                                axis_x,
                                axis_y,
                                radius_outer,
                                LASER_CYLINDER_SIDES,
                            )
                            layer_inner = build_cylinder_points(
                                center_base,
                                blade_tip,
                                axis_x,
                                axis_y,
                                radius_inner,
                                LASER_CYLINDER_SIDES,
                            )

                            pts_outer, _ = cv2.projectPoints(layer_outer, prism_rvec, prism_tvec, camera_matrix, dist_coeffs)
                            pts_inner, _ = cv2.projectPoints(layer_inner, prism_rvec, prism_tvec, camera_matrix, dist_coeffs)

                            outer_safe = projected_points_are_safe(pts_outer, MAX_PROJECTED_COORD_ABS_PX)
                            inner_safe = projected_points_are_safe(pts_inner, MAX_PROJECTED_COORD_ABS_PX)
                            if outer_safe and inner_safe:
                                hull_outer = cv2.convexHull(pts_outer.reshape(-1, 2).astype(np.int32))
                                hull_inner = cv2.convexHull(pts_inner.reshape(-1, 2).astype(np.int32))

                                glow_alpha = 0.08 + (0.20 * laser_power)
                                glow_thickness = max(1, int(round(1 + (3 * laser_power))))

                                # Halo de color segun tecla seleccionada (R/G/B).
                                overlay = frame.copy()
                                cv2.fillConvexPoly(overlay, hull_outer, saber_color["outer_fill"])
                                cv2.polylines(overlay, [hull_outer], True, saber_color["outer_edge"], glow_thickness)
                                cv2.addWeighted(overlay, glow_alpha, frame, 1.0 - glow_alpha, 0, frame)

                                # Núcleo claro del láser.
                                cv2.fillConvexPoly(frame, hull_inner, saber_color["inner_fill"])
                                cv2.polylines(frame, [hull_inner], True, (255, 255, 255), 2)

        # --- Preview / HUD ---
        # Aplicamos espejo solo al mostrar (no al tracking) para no afectar la estimación.
        mirrored_frame = cv2.flip(frame, 1) if camera_horizontal_flip else frame.copy()
        h, w = mirrored_frame.shape[:2]

        # Etiquetas de depuración (si SHOW_FACE_INFO está activo).
        if marker_labels:
            max_text_width = 550 if SHOW_FACE_INFO else 160
            for label in marker_labels:
                text_x = max(10, min(label["cx"] + 10, w - max_text_width))
                text_y = max(20, min(label["cy"] - 10, h - 10))
                cv2.putText(mirrored_frame, label["text"], (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Recuperamos el tamaño real de la ventana para ajustar el frame a pantalla completa.
        try:
            _, _, window_w, window_h = cv2.getWindowImageRect(WINDOW_NAME)
        except cv2.error:
            window_w, window_h = mirrored_frame.shape[1], mirrored_frame.shape[0]

        # Ajuste a ventana (manteniendo aspect ratio, con padding si hace falta).
        display_frame = fit_frame_to_window(mirrored_frame, window_w, window_h)
        cv2.imshow(WINDOW_NAME, display_frame)

        # Si el usuario cierra la ventana, salimos limpiamente.
        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print("[App] Ventana cerrada por el usuario.")
                break
        except cv2.error:
            print("[App] Ventana no disponible. Cerrando la app.")
            break

        # --- Teclado ---
        # `waitKey(1)` bombea la cola de eventos y devuelve el último keycode.
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            # Espacio: toggle del sable, con cooldown para evitar rebotes.
            toggle_now = time.perf_counter()
            if (toggle_now - last_toggle_time) > KEY_TOGGLE_COOLDOWN_S:
                laser_target_on = not laser_target_on
                if laser_target_on:
                    stop_all_sounds()
                    loop_playing = False
                    play_saber_on_sound(saber_on_sound_path)
                    loop_start_due_time = toggle_now + max(0.0, loop_start_delay_s)
                else:
                    stop_all_sounds()
                    loop_playing = False
                    play_saber_off_sound(saber_off_sound_path)
                    loop_start_due_time = None
                last_toggle_time = toggle_now
        if key == ord("i"):
            # I: invertir preview horizontal (no toca el tracking interno).
            flip_now = time.perf_counter()
            if (flip_now - last_flip_toggle_time) > KEY_TOGGLE_COOLDOWN_S:
                camera_horizontal_flip = not camera_horizontal_flip
                print(f"[View] Inversion horizontal: {'ON' if camera_horizontal_flip else 'OFF'}")
                last_flip_toggle_time = flip_now
        if key in (ord("r"), ord("g"), ord("b")):
            # R/G/B: cambia preset de color del sable.
            selected_color_key = chr(key)
            if selected_color_key in SABER_COLOR_PRESETS:
                current_saber_color_key = selected_color_key
                print(f"[Saber] Color seleccionado: {SABER_COLOR_PRESETS[selected_color_key]['name']}")
        if key == 27:
            # ESC: salir.
            break

    cap.release()
    if voice_listener is not None:
        voice_listener.stop()
    shutdown_audio_backend()
    cv2.destroyAllWindows()


def main() -> int:
    return run_app()


if __name__ == "__main__":
    raise SystemExit(main())
