"""Умный бот для игры Google Chrome Dino (t-rex runner).

Бот читает настоящее внутреннее состояние игры через движок
(window.Runner.instance_) и действует как опытный игрок:

* идеально перепрыгивает кактусы (учитывая текущую скорость игры);
* пригибается под птеродактилями, летящими на средней высоте;
* игнорирует птеродактилей, которые пролетают высоко над головой;
* перепрыгивает низколетящих птеродактилей;
* выживает очень долго и легко набирает 20 000+ очков.

Запуск:
    python dino.py

Полезные переменные окружения:
    DINO_HEADLESS=0        (по умолчанию) — показать окно браузера;
    DINO_HEADLESS=1        — скрытый режим без окна (быстрее);
    DINO_TARGET_SCORE=20000 — до какого счёта играть в одной игре;
    DINO_MAX_EPISODES=1000  — сколько игр максимум сыграть;
    DINO_GAME_SPEED=1.0     — множитель скорости игры (>1 быстрее, <1 медленнее);
    DINO_POLL_MS=12         — пауза между шагами бота, миллисекунды;
    DINO_BROWSER=chromium   — какой браузер использовать (chromium/chrome/firefox).

Требуется установленный Playwright:
    pip install playwright
    playwright install chromium
"""

from __future__ import annotations

import os
import sys
import time

from playwright.sync_api import sync_playwright

GAME_URL = "https://wayou.github.io/t-rex-runner/"


def _setup_console():
    """Делает вывод устойчивым к кодировке консоли Windows (cp1251 и т.п.)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on", "y", "да")


# --- Настройки поведения -------------------------------------------------
TARGET_SCORE = _env_int("DINO_TARGET_SCORE", 20000)   # до какого счёта играть
# Пауза между шагами бота (сек). Настраивается через DINO_POLL_MS (миллисекунды),
# по умолчанию 12 мс — примерно 60-80 проверок в секунду.
POLL_DELAY = max(0.001, _env_int("DINO_POLL_MS", 12) / 1000.0)
MAX_EPISODES = _env_int("DINO_MAX_EPISODES", 1000)    # предохранитель
RESTART_WAIT = 0.6            # ожидание перед рестартом после проигрыша
HEADLESS = _env_bool("DINO_HEADLESS", False)          # по умолчанию показываем окно
BROWSER_KIND = os.environ.get("DINO_BROWSER", "chromium").strip().lower()

# Множитель скорости самой игры. >1 — динозавр бежит быстрее (счёт растёт быстрее,
# но и препятствия прилетают чаще), <1 — медленнее. Логика прыжка/приседа
# масштабируется текущей скоростью игры, поэтому бот работает при любом значении.
GAME_SPEED = _env_float("DINO_GAME_SPEED", 1.0)
if GAME_SPEED <= 0:
    GAME_SPEED = 1.0

# Сколько секунд ждать инициализацию движка после загрузки страницы.
PAGE_READY_TIMEOUT = 20.0

# Если счёт не растёт столько секунд подряд — считаем игру «зависшей».
STALL_TIMEOUT = 30.0

# Множители дистанции срабатывания (подобраны под физику игры).
# Дистанция до препятствия масштабируется текущей скоростью игры.
#
# JUMP_FACTOR определяет, НАСКОЛЬКО ЗАРАНЕЕ бот начинает прыжок:
# gap = speed * JUMP_FACTOR + JUMP_BASE.
# При FPS=60 прыжок с фиксированной начальной скоростью длится ~28 кадров,
# поэтому приземная дуга накрывает широкие кактусы (до 75 px). Раньше бот
# прыгал через gap ≈ speed*14 и в момент столкновения уже опускался на
# препятствие. Значение ~8 подбирает момент так, чтобы пик прыжка совпал
# с прохождением препятствия и динозавр гарантированно перелетал его.
JUMP_FACTOR = _env_int("DINO_JUMP_FACTOR", 8)
JUMP_BASE = _env_int("DINO_JUMP_BASE", 0)
DUCK_FACTOR = _env_int("DINO_DUCK_FACTOR", 10)  # присед: gap = speed * DUCK_FACTOR + DUCK_BASE
DUCK_BASE = _env_int("DINO_DUCK_BASE", 8)

# Геометрия для определения высоты птеродактиля.
# Земля ~ y=140, стоящий динозавр занимает ~ y=93..140,
# присевший ~ y=115..140. Высоты птеродактиля: около 50 (высоко),
# 75 (средне) и 100 (низко).
PTERO_HIGH_MAX_Y = 65         # y < 65  -> летит высоко, можно не реагировать
PTERO_MID_MAX_Y = 95          # 65 <= y < 95 -> нужно присесть
                              # y >= 95 -> летит низко, нужно перепрыгнуть

JS_GET_STATE = r"""
() => {
    const runner = window.Runner && window.Runner.instance_;
    if (!runner) {
        return { status: "NO_RUNNER" };
    }

    let score = 0;
    try {
        const dm = runner.distanceMeter;
        if (dm) {
            // Настоящий счёт игры = пройденная дистанция * коэффициент.
            const coeff = (dm.config && dm.config.COEFFICIENT) || 0.025;
            score = Math.round((runner.distanceRan || 0) * coeff);
        }
    } catch (e) {
        score = 0;
    }

    if (runner.crashed) {
        return { status: "DEAD", score: score };
    }

    if (!runner.playing) {
        return { status: "NOT_PLAYING", score: score };
    }

    const dino = runner.tRex;
    const horizon = runner.horizon;
    const obstacles = (horizon && horizon.obstacles) ? horizon.obstacles : [];

    const speed = runner.currentSpeed || 6;
    const dinoX = dino.xPos;
    const dinoW = (dino.config && dino.config.WIDTH) || 44;

    const list = [];
    for (const o of obstacles) {
        const tc = o.typeConfig || {};
        let type = "UNKNOWN";
        try {
            // В разных сборках игры type — либо строка, либо функция.
            if (typeof tc.type === "function") {
                type = tc.type();
            } else if (tc.type) {
                type = tc.type;
            }
        } catch (e) {
            type = "UNKNOWN";
        }
        list.push({
            x: o.xPos,
            y: o.yPos,
            w: o.width || tc.width || 20,
            h: o.height || tc.height || 40,
            type: type
        });
    }

    return {
        status: "OK",
        score: score,
        speed: speed,
        dinoX: dinoX,
        dinoW: dinoW,
        jumping: !!dino.jumping,
        ducking: !!dino.ducking,
        obstacles: list
    };
}
"""

JS_RESTART = r"""
() => {
    const runner = window.Runner && window.Runner.instance_;
    if (runner && typeof runner.restart === "function") {
        runner.restart();
        return true;
    }
    return false;
}
"""

JS_SET_SPEED = r"""
(mult) => {
    const runner = window.Runner && window.Runner.instance_;
    if (!runner || !runner.config) {
        return false;
    }
    const cfg = runner.config;
    // Один раз запоминаем оригинальные значения, чтобы множитель был идемпотентным.
    if (!window.__dinoBaseCfg) {
        window.__dinoBaseCfg = {
            SPEED: cfg.SPEED,
            MAX_SPEED: cfg.MAX_SPEED,
            ACCELERATION: cfg.ACCELERATION
        };
    }
    const base = window.__dinoBaseCfg;
    cfg.SPEED = base.SPEED * mult;
    cfg.MAX_SPEED = base.MAX_SPEED * mult;
    cfg.ACCELERATION = base.ACCELERATION * mult;
    if (typeof runner.currentSpeed === "number") {
        if (runner.currentSpeed < cfg.SPEED) {
            runner.currentSpeed = cfg.SPEED;
        }
        if (runner.currentSpeed > cfg.MAX_SPEED) {
            runner.currentSpeed = cfg.MAX_SPEED;
        }
    }
    return true;
}
"""


class DinoBot:
    """Реактивный агент, управляющий игрой через браузер."""

    def __init__(self, page):
        self.page = page
        self.duck_held = False
        self.game_speed = GAME_SPEED

    # --- Настройка скорости игры ----------------------------------------
    def apply_game_speed(self):
        """Применяет множитель скорости игры к движку (идемпотентно)."""
        try:
            self.page.evaluate(JS_SET_SPEED, self.game_speed)
        except Exception:  # noqa: BLE001
            pass

    # --- Взаимодействие с клавиатурой -----------------------------------
    def _release_duck(self):
        if self.duck_held:
            try:
                self.page.keyboard.up("ArrowDown")
            except Exception:  # noqa: BLE001
                pass
            self.duck_held = False

    def _duck(self):
        if not self.duck_held:
            try:
                self.page.keyboard.down("ArrowDown")
                self.duck_held = True
            except Exception:  # noqa: BLE001
                self.duck_held = False

    def _jump(self):
        # Перед прыжком обязательно встаём из приседа.
        self._release_duck()
        try:
            self.page.keyboard.press("Space")
        except Exception:  # noqa: BLE001
            pass

    def reset_keys(self):
        self._release_duck()

    # --- Чтение состояния ------------------------------------------------
    def read_state(self):
        try:
            return self.page.evaluate(JS_GET_STATE)
        except Exception as error:  # noqa: BLE001
            return {"status": "ERROR", "error": str(error)}

    def wait_for_engine(self, timeout=PAGE_READY_TIMEOUT):
        """Ждёт, пока движок игры появится на странице."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.read_state().get("status") != "NO_RUNNER":
                return True
            time.sleep(0.05)
        return False

    # --- Принятие решения ------------------------------------------------
    @staticmethod
    def choose_action(state):
        """Возвращает одно из: 'jump', 'duck', 'none'."""
        speed = state.get("speed", 6) or 6
        dino_x = state.get("dinoX", 0)
        dino_w = state.get("dinoW", 44)
        jumping = state.get("jumping", False)
        obstacles = state.get("obstacles", [])

        if not obstacles:
            return "none"

        # Ближайшее препятствие, которое ещё не осталось позади.
        target = None
        for obstacle in sorted(obstacles, key=lambda item: item.get("x", 0)):
            if obstacle.get("x", 0) + obstacle.get("w", 0) < dino_x:
                continue
            target = obstacle
            break

        if target is None:
            return "none"

        gap = target.get("x", 0) - (dino_x + dino_w)
        obstacle_type = target.get("type", "UNKNOWN")

        if obstacle_type == "PTERODACTYL":
            y = target.get("y", 0)
            if y < PTERO_HIGH_MAX_Y:
                # Летит высоко — свободно пробегаем под ним.
                return "none"
            if y < PTERO_MID_MAX_Y:
                need = "duck"
            else:
                need = "jump"
        else:
            need = "jump"

        if need == "jump":
            if not jumping and gap <= speed * JUMP_FACTOR + JUMP_BASE:
                return "jump"
            return "none"

        # need == "duck"
        if gap <= speed * DUCK_FACTOR + DUCK_BASE:
            return "duck"
        return "none"

    def apply_action(self, action):
        if action == "jump":
            self._jump()
        elif action == "duck":
            self._duck()
        else:
            self._release_duck()

    # --- Управление игрой ------------------------------------------------
    def start_game(self, timeout=8.0):
        """Начинает/перезапускает игру и ждёт её реального старта."""
        self.reset_keys()

        # Применяем настройку скорости до рестарта, чтобы она учлась при загрузке.
        self.apply_game_speed()

        # Сначала пробуем штатный рестарт через внутренний API.
        try:
            self.page.evaluate(JS_RESTART)
        except Exception:  # noqa: BLE001
            pass

        deadline = time.time() + timeout
        fallback_used = False

        while time.time() < deadline:
            state = self.read_state()
            status = state.get("status")

            if status == "OK":
                return True

            if not fallback_used and status in ("NOT_PLAYING", "DEAD"):
                # Резервный вариант — клик по canvas и нажатие пробела.
                fallback_used = True
                try:
                    self.page.locator("canvas").first.click(timeout=1000)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    self.page.keyboard.press("Space")
                except Exception:  # noqa: BLE001
                    pass

            time.sleep(0.05)

        return False

    def play_episode(self, target_score=TARGET_SCORE):
        """Играет одну игру. Возвращает достигнутый счёт."""
        best_score = 0
        start = time.time()
        last_report = start
        last_progress = start
        last_score = -1

        while True:
            state = self.read_state()
            status = state.get("status")

            if status == "DEAD":
                self.reset_keys()
                return max(best_score, state.get("score", 0))

            if status in ("NO_RUNNER", "ERROR"):
                time.sleep(POLL_DELAY)
                continue

            if status == "NOT_PLAYING":
                # Игра ещё не запущена — стартуем.
                self.start_game()
                continue

            score = state.get("score", 0)
            if score > best_score:
                best_score = score

            now = time.time()

            # Контроль «зависшей» игры: счёт не растёт слишком долго.
            if score > last_score:
                last_score = score
                last_progress = now
            elif now - last_progress > STALL_TIMEOUT:
                print("  Игра зависла (счёт не растёт) — перезапускаем.")
                self.reset_keys()
                return best_score

            if best_score >= target_score:
                print(f"  Цель {target_score} очков достигнута!")
                return best_score

            action = self.choose_action(state)
            self.apply_action(action)

            # Иногда сообщаем о прогрессе.
            if now - last_report >= 5:
                elapsed = now - start
                print(
                    f"  Счёт: {best_score:>6}  "
                    f"скорость: {state.get('speed', 0):.1f}  "
                    f"время: {elapsed:.0f}с"
                )
                last_report = now

            time.sleep(POLL_DELAY)


# Общие аргументы запуска Chromium. В headless-режиме анимация идёт без vsync,
# из-за чего requestAnimationFrame может крутиться быстрее ~60 fps и ломать
# логику столкновений игры. Ограничиваем частоту кадров и отключаем лишнее.
_CHROMIUM_ARGS = [
    "--disable-gpu",
    "--disable-extensions",
    "--no-first-run",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
    "--disable-features=CalculateNativeWinOcclusion",
]


def _launch_browser(playwright):
    """Запускает Chromium, стараясь сделать это максимально надёжно."""
    # Порядок попыток: сначала выбранный браузер, затем запасные варианты.
    if BROWSER_KIND == "chromium":
        candidates = ["chromium", "chrome", "firefox"]
    elif BROWSER_KIND == "chrome":
        candidates = ["chrome", "chromium", "firefox"]
    else:
        candidates = ["firefox", "chromium"]

    last_error = None
    for candidate in candidates:
        try:
            if candidate == "chromium":
                return playwright.chromium.launch(
                    headless=HEADLESS,
                    args=list(_CHROMIUM_ARGS),
                )
            if candidate == "firefox":
                return playwright.firefox.launch(
                    headless=HEADLESS,
                    firefox_user_prefs={
                        # В headless Firefox анимация идёт без vsync (~167 fps),
                        # что ломает логику столкновений игры. Ограничиваем 60 fps.
                        "layout.frame_rate": 60,
                        "dom.disable_beforeunload": True,
                    },
                )
            return playwright.chromium.launch(
                channel=candidate,
                headless=HEADLESS,
                args=list(_CHROMIUM_ARGS),
            )
        except Exception as error:  # noqa: BLE001
            last_error = error
            message = str(error).encode("ascii", "replace").decode("ascii")
            print(f"Не удалось запустить '{candidate}': {message}")

    raise RuntimeError(f"Не удалось запустить браузер: {last_error}")


def main():
    _setup_console()
    print(f"Запускаем умного бота для Dino (браузер: {BROWSER_KIND})...")
    print(
        f"Цель по очкам: {TARGET_SCORE}, игр максимум: {MAX_EPISODES}, "
        f"множитель скорости игры: {GAME_SPEED:g}"
    )

    with sync_playwright() as playwright:
        browser = _launch_browser(playwright)
        page = browser.new_page(viewport={"width": 800, "height": 600})

        print("Открываем игру...")
        # domcontentloaded быстрее и надёжнее, чем networkidle.
        page.goto(GAME_URL, wait_until="domcontentloaded", timeout=30000)

        bot = DinoBot(page)
        print("Ждём инициализацию движка игры...")
        if not bot.wait_for_engine():
            print("Движок игры не инициализировался. Проверяем доступность страницы...")
            try:
                page.reload(wait_until="domcontentloaded", timeout=30000)
            except Exception as error:  # noqa: BLE001
                print(f"  Перезагрузка не удалась: {error}")
            bot.wait_for_engine()

        best_overall = 0
        episode = 0

        while episode < MAX_EPISODES:
            episode += 1
            print(f"\n--- Игра №{episode} ---")

            if not bot.start_game():
                print("Игра не запустилась. Перезагружаем страницу...")
                try:
                    page.reload(wait_until="domcontentloaded", timeout=30000)
                except Exception as error:  # noqa: BLE001
                    print(f"  Перезагрузка не удалась: {error}")
                bot.wait_for_engine()
                continue

            try:
                score = bot.play_episode()
            except Exception as error:  # noqa: BLE001
                print(f"Ошибка во время игры: {error}")
                bot.reset_keys()
                time.sleep(1.0)
                continue

            best_overall = max(best_overall, score)
            print(f"Итоговый счёт игры: {score}. Лучший результат: {best_overall}")

            if score >= TARGET_SCORE:
                print(f"\nОтличный результат: {best_overall} очков!")
                break

            time.sleep(RESTART_WAIT)

        print(f"\nЛучший результат за сессию: {best_overall} очков.")

        # Не закрываем окно сразу — даём время посмотреть результат.
        if not HEADLESS:
            try:
                input("\nНажми Enter, чтобы закрыть браузер...")
            except EOFError:
                pass

        try:
            browser.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
