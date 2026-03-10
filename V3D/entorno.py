import argparse
from dataclasses import dataclass
import math
import random
import socket

import pygame


@dataclass
class FlyingShape:
    x: float
    y: float
    vx: float
    vy: float
    radius: int
    color: tuple[int, int, int]
    is_bomb: bool = False


@dataclass
class SlashParticle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    color: tuple[int, int, int]


@dataclass
class SplitHalf:
    x: float
    y: float
    vx: float
    vy: float
    radius: int
    color: tuple[int, int, int]
    angle: float
    spin: float
    side: int
    life: float


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def point_to_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq <= 1e-8:
        return math.dist((px, py), (ax, ay))
    t = clamp((apx * abx + apy * aby) / ab_len_sq, 0.0, 1.0)
    cx = ax + t * abx
    cy = ay + t * aby
    return math.dist((px, py), (cx, cy))


def segment_intersects_circle(ax: float, ay: float, bx: float, by: float, cx: float, cy: float, radius: float) -> bool:
    return point_to_segment_distance(cx, cy, ax, ay, bx, by) <= radius


def random_shape(width: int, height: int) -> FlyingShape:
    spawn_x = random.randint(int(width * 0.15), int(width * 0.85))
    spawn_y = height + random.randint(20, 80)

    bomb_spawn_chance = 0.16
    if random.random() < bomb_spawn_chance:
        radius = random.randint(38, 54)
        speed_x = random.uniform(-180.0, 180.0)
        speed_y = random.uniform(-980.0, -760.0)
        return FlyingShape(spawn_x, spawn_y, speed_x, speed_y, radius, (55, 55, 60), True)

    shape_profiles = [
        {
            "weight": 0.45,
            "radius_range": (34, 46),
            "vx_range": (-260.0, 260.0),
            "vy_range": (-1060.0, -860.0),
            "colors": [(255, 110, 110), (255, 185, 90)],
        },
        {
            "weight": 0.35,
            "radius_range": (44, 58),
            "vx_range": (-190.0, 190.0),
            "vy_range": (-920.0, -740.0),
            "colors": [(130, 225, 140), (120, 180, 255)],
        },
        {
            "weight": 0.20,
            "radius_range": (58, 72),
            "vx_range": (-130.0, 130.0),
            "vy_range": (-820.0, -650.0),
            "colors": [(220, 140, 255)],
        },
    ]

    profile = random.choices(shape_profiles, weights=[p["weight"] for p in shape_profiles], k=1)[0]
    radius = random.randint(*profile["radius_range"])
    speed_x = random.uniform(*profile["vx_range"])
    speed_y = random.uniform(*profile["vy_range"])
    color = random.choice(profile["colors"])

    return FlyingShape(spawn_x, spawn_y, speed_x, speed_y, radius, color, False)


def create_split_halves(shape: FlyingShape, slash_dx: float, slash_dy: float) -> list[SplitHalf]:
    length = math.hypot(slash_dx, slash_dy)
    if length < 1e-6:
        slash_dx, slash_dy = 1.0, 0.0
        length = 1.0

    nx = -slash_dy / length
    ny = slash_dx / length
    separation_speed = 220.0

    halves: list[SplitHalf] = []
    for side in (-1, 1):
        halves.append(
            SplitHalf(
                x=shape.x + nx * side * 8.0,
                y=shape.y + ny * side * 8.0,
                vx=shape.vx + nx * side * separation_speed,
                vy=shape.vy + ny * side * separation_speed - 80.0,
                radius=shape.radius,
                color=shape.color,
                angle=math.atan2(slash_dy, slash_dx),
                spin=side * random.uniform(2.5, 4.5),
                side=side,
                life=0.9,
            )
        )
    return halves


def draw_split_half(surface: pygame.Surface, half: SplitHalf) -> None:
    points: list[tuple[float, float]] = []
    steps = 20
    start = half.angle - math.pi / 2
    for i in range(steps + 1):
        t = i / steps
        a = start + t * math.pi
        px = half.radius * math.cos(a)
        py = half.radius * math.sin(a)
        if half.side < 0:
            px = -px
        rx = px * math.cos(half.spin * (1.0 - half.life)) - py * math.sin(half.spin * (1.0 - half.life))
        ry = px * math.sin(half.spin * (1.0 - half.life)) + py * math.cos(half.spin * (1.0 - half.life))
        points.append((half.x + rx, half.y + ry))

    points.append((half.x, half.y))
    pygame.draw.polygon(surface, half.color, points)
    pygame.draw.polygon(surface, (245, 245, 245), points, width=2)


def create_slash_particles(x: float, y: float, color: tuple[int, int, int]) -> list[SlashParticle]:
    particles: list[SlashParticle] = []
    for _ in range(16):
        angle = random.uniform(0.0, math.tau)
        speed = random.uniform(120.0, 420.0)
        particles.append(
            SlashParticle(
                x=x,
                y=y,
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
                life=random.uniform(0.22, 0.45),
                color=color,
            )
        )
    return particles


def parse_tracker_message(message: str) -> tuple[float, float, float, float] | None:
    parts = [p.strip() for p in message.split(",")]
    if len(parts) not in (2, 4):
        return None

    try:
        x = float(parts[0])
        y = float(parts[1])
        if len(parts) == 4:
            source_w = float(parts[2])
            source_h = float(parts[3])
        else:
            source_w = 640.0
            source_h = 480.0
    except ValueError:
        return None

    if source_w <= 0 or source_h <= 0:
        return None
    return x, y, source_w, source_h


def map_tracker_to_screen(x: float, y: float, source_w: float, source_h: float, target_w: int, target_h: int) -> tuple[int, int]:
    normalized_x = clamp(x / source_w, 0.0, 1.0)
    normalized_y = clamp(y / source_h, 0.0, 1.0)
    px = int(normalized_x * target_w)
    py = int(normalized_y * target_h)
    return int(clamp(px, 0, target_w - 1)), int(clamp(py, 0, target_h - 1))


def run_pygame_pointer(width: int, height: int, trail_length: int) -> None:
    pygame.init()
    pygame.display.set_caption("Demo Fruit Ninja - control con raton")
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    width, height = screen.get_size()
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)
    game_over_font = pygame.font.SysFont("consolas", 64, bold=True)
    game_over_sub_font = pygame.font.SysFont("consolas", 24)

    pygame.mouse.set_visible(False)

    tracker_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tracker_sock.bind(("127.0.0.1", 5005))
    tracker_sock.setblocking(False)

    pointer_x, pointer_y = width // 2, height // 2

    gravity = 980.0
    spawn_interval_start = 1.20
    spawn_interval_end = 0.25
    difficulty_ramp_seconds = 75.0

    game_over_duration = 2.0
    game_over = False
    game_over_timer = 0.0

    score = 0
    lives = 3
    shapes: list[FlyingShape] = []
    split_halves: list[SplitHalf] = []
    particles: list[SlashParticle] = []
    trail: list[tuple[float, float]] = []
    spawn_timer = 0.0
    elapsed_time = 0.0

    def reset_game_state() -> None:
        nonlocal score, lives, shapes, split_halves, particles, trail, spawn_timer, elapsed_time
        score = 0
        lives = 3
        shapes = []
        split_halves = []
        particles = []
        trail = []
        spawn_timer = 0.0
        elapsed_time = 0.0

    running = True

    while running:
        dt = clock.tick(144) / 1000.0
        elapsed_time += dt

        difficulty_t = clamp(elapsed_time / difficulty_ramp_seconds, 0.0, 1.0)
        current_spawn_interval = (
            spawn_interval_start + (spawn_interval_end - spawn_interval_start) * difficulty_t
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        while True:
            try:
                packet, _ = tracker_sock.recvfrom(128)
            except BlockingIOError:
                break

            parsed = parse_tracker_message(packet.decode("utf-8", errors="ignore"))
            if parsed is None:
                continue

            tx, ty, source_w, source_h = parsed
            pointer_x, pointer_y = map_tracker_to_screen(tx, ty, source_w, source_h, width, height)

        if not game_over:
            trail.append((float(pointer_x), float(pointer_y)))
            if len(trail) > trail_length:
                trail.pop(0)

            spawn_timer += dt
            while spawn_timer >= current_spawn_interval:
                shapes.append(random_shape(width, height))
                spawn_timer -= current_spawn_interval

            for shape in shapes:
                shape.vy += gravity * dt
                shape.x += shape.vx * dt
                shape.y += shape.vy * dt

            to_remove: set[int] = set()
            if len(trail) >= 2:
                ax, ay = trail[-2]
                bx, by = trail[-1]
                slash_dx = bx - ax
                slash_dy = by - ay
                for idx, shape in enumerate(shapes):
                    if segment_intersects_circle(ax, ay, bx, by, shape.x, shape.y, shape.radius):
                        to_remove.add(idx)
                        if shape.is_bomb:
                            lives -= 1
                            particles.extend(create_slash_particles(shape.x, shape.y, (255, 80, 80)))
                            if lives <= 0:
                                game_over = True
                                game_over_timer = game_over_duration
                                break
                        else:
                            score += 1
                            particles.extend(create_slash_particles(shape.x, shape.y, shape.color))
                            split_halves.extend(create_split_halves(shape, slash_dx, slash_dy))

            for idx, shape in enumerate(shapes):
                if idx in to_remove:
                    continue

                if shape.y - shape.radius > height + 60:
                    to_remove.add(idx)
                    if not shape.is_bomb:
                        lives -= 1
                        if lives <= 0:
                            game_over = True
                            game_over_timer = game_over_duration
                            break
                elif shape.x < -120 or shape.x > width + 120:
                    to_remove.add(idx)

            if to_remove:
                shapes = [shape for i, shape in enumerate(shapes) if i not in to_remove]

            alive_particles: list[SlashParticle] = []
            for p in particles:
                p.life -= dt
                if p.life <= 0:
                    continue
                p.vy += gravity * 0.45 * dt
                p.x += p.vx * dt
                p.y += p.vy * dt
                alive_particles.append(p)
            particles = alive_particles

            alive_halves: list[SplitHalf] = []
            for half in split_halves:
                half.life -= dt
                if half.life <= 0:
                    continue
                half.vy += gravity * 0.9 * dt
                half.x += half.vx * dt
                half.y += half.vy * dt
                if half.y - half.radius > height + 100:
                    continue
                alive_halves.append(half)
            split_halves = alive_halves
        else:
            game_over_timer -= dt
            if game_over_timer <= 0.0:
                reset_game_state()
                game_over = False

        screen.fill((18, 18, 22))

        for shape in shapes:
            if shape.is_bomb:
                cx, cy = int(shape.x), int(shape.y)
                pygame.draw.circle(screen, shape.color, (cx, cy), shape.radius)
                pygame.draw.circle(screen, (220, 220, 220), (cx, cy), shape.radius, width=2)
                x_size = int(shape.radius * 0.52)
                x_color = (255, 90, 90)
                x_width = max(3, int(shape.radius * 0.12))
                pygame.draw.line(screen, x_color, (cx - x_size, cy - x_size), (cx + x_size, cy + x_size), width=x_width)
                pygame.draw.line(screen, x_color, (cx - x_size, cy + x_size), (cx + x_size, cy - x_size), width=x_width)
            else:
                pygame.draw.circle(screen, shape.color, (int(shape.x), int(shape.y)), shape.radius)
                pygame.draw.circle(screen, (250, 250, 250), (int(shape.x), int(shape.y)), shape.radius, width=2)

        for half in split_halves:
            draw_split_half(screen, half)

        for p in particles:
            alpha_life = clamp(p.life / 0.45, 0.15, 1.0)
            radius = int(2 + alpha_life * 4)
            color = (
                int(p.color[0] * alpha_life),
                int(p.color[1] * alpha_life),
                int(p.color[2] * alpha_life),
            )
            pygame.draw.circle(screen, color, (int(p.x), int(p.y)), radius)

        if len(trail) >= 2:
            for i in range(1, len(trail)):
                t = i / len(trail)
                width_line = int(2 + t * 6)
                color = (int(90 + 110 * t), int(200 + 40 * t), 255)
                pygame.draw.line(screen, color, trail[i - 1], trail[i], width=width_line)

        pygame.draw.circle(screen, (80, 170, 255), (pointer_x, pointer_y), 10)
        pygame.draw.circle(screen, (220, 240, 255), (pointer_x, pointer_y), 18, width=2)

        info = [
            "Control: tracker UDP 127.0.0.1:5005",
            "Bomba (X): al cortarla pierdes 1 vida",
            f"Longitud traza: {trail_length}",
            f"Dificultad: {int(difficulty_t * 100)}%",
            f"Score: {score}",
            f"Vidas: {lives}",
            "Salir: ESC o cerrar ventana",
        ]
        for i, text in enumerate(info):
            line = font.render(text, True, (235, 235, 235))
            screen.blit(line, (16, 16 + i * 26))

        if game_over:
            overlay = pygame.Surface((width, height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 130))
            screen.blit(overlay, (0, 0))

            game_over_text = game_over_font.render("GAME OVER", True, (255, 95, 95))
            sub_text = game_over_sub_font.render("Reiniciando...", True, (240, 240, 240))

            game_over_rect = game_over_text.get_rect(center=(width // 2, height // 2 - 20))
            sub_rect = sub_text.get_rect(center=(width // 2, height // 2 + 32))

            screen.blit(game_over_text, game_over_rect)
            screen.blit(sub_text, sub_rect)

        pygame.display.flip()

    tracker_sock.close()
    pygame.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo pygame: corte de figuras con la trayectoria del raton.")
    parser.add_argument("--width", type=int, default=1280, help="Ancho de la ventana.")
    parser.add_argument("--height", type=int, default=720, help="Alto de la ventana.")
    parser.add_argument("--trail-length", type=int, default=36, help="Cantidad de puntos visibles en la traza.")
    args = parser.parse_args()

    if args.width <= 100 or args.height <= 100:
        raise ValueError("--width y --height deben ser mayores a 100")
    if args.trail_length < 6:
        raise ValueError("--trail-length debe ser al menos 6")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    run_pygame_pointer(arguments.width, arguments.height, arguments.trail_length)
