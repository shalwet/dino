"""Обучение нейросети игре Google Chrome Dino (T-Rex Runner) через NEAT.

Что делает модуль
-----------------
1. ОБУЧЕНИЕ (по умолчанию в скрытых окнах, максимально быстро):
   запускает ``DINO_WORKERS`` (по умолчанию 15) браузеров Chromium
   параллельно через асинхронный Playwright, в каждом живёт свой «мозг»
   (геном NEAT). Мозг читает НАСТОЯЩЕЕ внутреннее состояние игры
   (``window.Runner.instance_``) и решает: прыгнуть, присесть или ничего.
   Когда все особи погибли — считается приспособленность (fitness), NEAT
   строит новое поколение, цикл повторяется.
2. ПОКАЗ (в самом конце, в ВИДИМОМ окне):
   модуль сам открывает окно Chromium и запускает лучшего найденного бота,
   чтобы можно было наблюдать за бегущим динозавриком. Отдельный скрипт
   для просмотра НЕ нужен — всё делает ``python dino.py``.

Сеть
----
Входов — 11 (нормализованные признаки игры), выходов — 2 (jump, duck),
активация — sigmoid, решения по порогу 0.5. Самый важный признак —
«время до столкновения» (gap / speed): именно он позволяет сети
принимать решения заранее и переживать разгон до максимальной скорости,
после чего игра может идти бесконечно (счёт 15000+ достигается примерно
за 13 минут реального времени игры).

Почему так быстро
-----------------
Все браузеры живут внутри одного asyncio-цикла и опрашивают игру
параллельно, поэтому 15 игр идут одновременно на одном ядре координации,
а рендеринг раскидывается по ядрам самим Chromium.

Запуск
------
    python dino.py

Переменные окружения
--------------------
    DINO_WORKERS=15           число параллельных браузеров = pop_size;
    DINO_TRAIN=1              обучаться (1) или только показать сохранённого (0);
    DINO_TRAIN_HEADLESS=1     скрытые окна во время обучения (1 = быстрее);
    DINO_SHOW=1               в конце открыть видимое окно с лучшим ботом;
    DINO_SHOW_EPISODES=0      сколько игр показать (0 = бесконечно);
    DINO_MAX_GENERATIONS=100  сколько поколений учить (0 = бесконечно);
    DINO_TARGET_SCORE=15000   при достижении такого счёта обучение остановится;
    DINO_GAME_SPEED=1.0       множитель скорости игры (>1 быстрее);
    DINO_POLL_MS=10           пауза между шагами бота, миллисекунды;
    DINO_EPISODE_CAP=1500     максимальная длительность одной игры, секунды
                              (предохранитель от бесконечного простоя);
    DINO_SAVE_PREFIX=dino_best  префикс файлов сохранения лучшего генома.

Требования
----------
    pip install playwright neat-python
    playwright install chromium
"""

from __future__ import annotations

import asyncio
import os
import pickle
import sys
import time
from pathlib import Path

import neat
from playwright.async_api import async_playwright

GAME_URL = "https://wayou.github.io/t-rex-runner/"

# Директория проекта (рядом лежат neat_config.txt и файлы сохранения).
_HERE = Path(__file__).resolve().parent
NEAT_CONFIG_PATH = _HERE / "neat_config.txt"


# ---------------------------------------------------------------------------
# Настройки из переменных окружения
# ---------------------------------------------------------------------------
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


# Число параллельных браузеров = размер популяции NEAT.
WORKERS = max(1, _env_int("DINO_WORKERS", 15))
TRAIN_ENABLED = _env_bool("DINO_TRAIN", True)
HEADLESS = _env_bool("DINO_TRAIN_HEADLESS", True)
SHOW_ENABLED = _env_bool("DINO_SHOW", True)
SHOW_EPISODES = _env_int("DINO_SHOW_EPISODES", 0)  # 0 = бесконечно
MAX_GENERATIONS = _env_int("DINO_MAX_GENERATIONS", 100)  # 0 = без ограничения
TARGET_SCORE = _env_int("DINO_TARGET_SCORE", 15000)
GAME_SPEED = _env_float("DINO_GAME_SPEED", 1.0)
if GAME_SPEED <= 0:
    GAME_SPEED = 1.0
POLL_DELAY = max(0.001, _env_int("DINO_POLL_MS", 10) / 1000.0)
EPISODE_CAP = max(30.0, _env_float("DINO_EPISODE_CAP", 1500.0))
SAVE_PREFIX = os.environ.get("DINO_SAVE_PREFIX", "dino_best").strip() or "dino_best"
SHOWCASE_LIMIT = 5  # сколько лучших особей держать для показа


# ---------------------------------------------------------------------------
# Игровая геометрия (из index.js), нужна для нормализации признаков
# ---------------------------------------------------------------------------
CANVAS_WIDTH = 600.0     # Runner.defaultDimensions.WIDTH
GROUND_Y = 140.0         # низ игрового поля = HEIGHT(150) - bottomPad(10)
DINO_Y_TOP = 93.0        # groundYPos = 150 - 47 - 10 (верх динозавра стоя)
DINO_W = 44.0            # Trex.config.WIDTH
DINO_H = 47.0            # Trex.config.HEIGHT
DINO_H_DUCK = 25.0       # Trex.config.HEIGHT_DUCK
SCORE_COEFF = 0.025      # DistanceMeter.config.COEFFICIENT
MAX_SPEED = 13.0         # Runner.config.MAX_SPEED
PTERO_MIN_SPEED = 8.5    # Obstacle.types PTERODACTYL.minSpeed

# Нормализующие масштабы признаков.
GAP_SCALE = 200.0        # «разумная» дистанция до препятствия, px
TTI_MAX = 2.0            # максимум «времени до столкновения», сек
AIR_SCALE = 40.0         # высота динозавра над землёй, px
FLY_SCALE = 50.0         # высота полёта препятствия над землёй, px
DUCK_FLY_MIN = 20.0      # присед спасает, если препятствие выше этой высоты


# ---------------------------------------------------------------------------
# JavaScript, исполняемый в браузере
# ---------------------------------------------------------------------------
# Установка менеджера ввода: перехватывает события клавиатуры и позволяет
# «нажимать/отпускать» клавиши из Python через window.__dino. События
# отправляются как настоящие keydown/keyup, поэтому игра реагирует штатно
# (Space/ArrowUp — прыжок, ArrowDown — присед).
JS_INSTALL = r"""
() => {
    if (window.__dino) { return true; }

    const state = { held: new Set() };

    function dispatch(type, code, key) {
        const evt = new KeyboardEvent(type, {
            key: key,
            code: code,
            keyCode: code === 'Space' ? 32 : (code === 'ArrowDown' ? 40 : 38),
            which: code === 'Space' ? 32 : (code === 'ArrowDown' ? 40 : 38),
            bubbles: true,
            cancelable: true
        });
        document.dispatchEvent(evt);
    }

    window.__dino = {
        press(code, key) {
            if (state.held.has(code)) { return; }
            state.held.add(code);
            dispatch('keydown', code, key);
        },
        release(code, key) {
            if (!state.held.has(code)) { return; }
            state.held.delete(code);
            dispatch('keyup', code, key);
        },
        releaseAll() {
            for (const code of Array.from(state.held)) {
                const key = code === 'Space' ? ' ' : (code === 'ArrowDown' ? 'ArrowDown' : 'ArrowUp');
                this.release(code, key);
            }
        },
        _state: state
    };
    return true;
}
"""

# Возвращает сырое состояние игры. Координаты — в пикселях; GROUND = 140.
JS_READ = r"""
() => {
    const runner = window.Runner && window.Runner.instance_;
    if (!runner) { return { status: 'NO_RUNNER' }; }

    let score = 0;
    try {
        const dm = runner.distanceMeter;
        const coeff = (dm && dm.config && dm.config.COEFFICIENT) || 0.025;
        score = Math.round((runner.distanceRan || 0) * coeff);
    } catch (e) { score = 0; }

    if (runner.crashed) { return { status: 'DEAD', score: score }; }
    if (!runner.playing) { return { status: 'NOT_PLAYING', score: score }; }

    const GROUND = 140;
    const dino = runner.tRex;
    const horizon = runner.horizon;
    const obstacles = (horizon && horizon.obstacles) ? horizon.obstacles : [];
    const speed = runner.currentSpeed || 6;
    const dinoX = dino.xPos;
    const dinoW = (dino.config && dino.config.WIDTH) || 44;
    const dinoH = dino.ducking ? 25 : 47;
    const dinoAir = Math.max(0, GROUND - (dino.yPos + dinoH));

    // Ближайшее препятствие впереди (наименьший xPos, ещё не позади динозавра).
    let first = null;
    for (const o of obstacles) {
        const w = o.width || 20;
        if (o.xPos + w < dinoX) { continue; }
        let type = 'UNKNOWN';
        try {
            const tc = o.typeConfig || {};
            type = (typeof tc.type === 'function') ? tc.type() : (tc.type || 'UNKNOWN');
        } catch (e) { type = 'UNKNOWN'; }
        const h = (o.typeConfig && o.typeConfig.height) || 40;
        const bottom = (o.yPos || GROUND) + h;
        const fly = Math.max(0, GROUND - bottom);
        if (!first || o.xPos < first.x) {
            first = { x: o.xPos, y: o.yPos, w: w, h: h, type: type, fly: fly };
        }
    }

    let gap = null, obsW = 0, obsH = 0, obsFly = 0, isPtero = 0;
    if (first) {
        gap = first.x - (dinoX + dinoW);
        obsW = first.w;
        obsH = first.h;
        obsFly = first.fly;
        if (first.type === 'PTERODACTYL') { isPtero = 1; }
        else if (first.type === 'UNKNOWN' && speed >= 8.5 && first.h <= 40) { isPtero = 1; }
    }

    return {
        status: 'OK',
        score: score,
        speed: speed,
        gap: gap,
        obsW: obsW,
        obsH: obsH,
        obsFly: obsFly,
        isPtero: isPtero,
        dinoAir: dinoAir,
        jumping: !!dino.jumping,
        ducking: !!dino.ducking
    };
}
"""


# ---------------------------------------------------------------------------
# Нормализация признаков
# ---------------------------------------------------------------------------
def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def build_inputs(state: dict) -> list[float]:
    """Преобразует состояние игры в 11 нормализованных входов сети в [0, 1].

    Порядок признаков совпадает с описанием в ``neat_config.txt``.
    """
    speed = float(state.get("speed", 6) or 6)
    if speed < 1.0:
        speed = 1.0

    gap_raw = state.get("gap")
    if gap_raw is None:
        gap = GAP_SCALE
    else:
        gap = max(0.0, float(gap_raw))

    obs_w = float(state.get("obsW", 0) or 0)
    obs_h = float(state.get("obsH", 0) or 0)
    obs_fly = float(state.get("obsFly", 0) or 0)
    is_ptero = 1.0 if state.get("isPtero") else 0.0

    # Главный признак: сколько времени осталось до столкновения.
    tti = gap / speed
    # Подсказка: спасает ли присед от этого препятствия.
    duck_needed = 1.0 if (is_ptero and obs_fly >= DUCK_FLY_MIN) else 0.0

    return [
        _clamp(speed / MAX_SPEED),               # 1. скорость игры
        _clamp(tti / TTI_MAX),                    # 2. время до столкновения
        _clamp(gap / GAP_SCALE),                  # 3. дистанция до препятствия
        _clamp(obs_w / FLY_SCALE),                # 4. ширина препятствия
        _clamp(obs_h / FLY_SCALE),                # 5. высота препятствия
        _clamp(obs_fly / FLY_SCALE),              # 6. высота полёта над землёй
        is_ptero,                                 # 7. птеродактиль?
        _clamp(float(state.get("dinoAir", 0) or 0) / AIR_SCALE),  # 8. высота динозавра
        1.0 if state.get("jumping") else 0.0,     # 9. в прыжке?
        1.0 if state.get("ducking") else 0.0,     # 10. приседает?
        duck_needed,                              # 11. присед поможет?
    ]


def decide_action(outputs: list[float], state: dict) -> str:
    """По выходам сети выбирает действие: 'jump', 'duck' или 'none'."""
    jump = outputs[0] > 0.5
    duck = outputs[1] > 0.5
    if jump and duck:
        # Оба выхода активны: от птеродактиля обычно спасает присед,
        # от наземных кактусов — прыжок.
        return "duck" if state.get("isPtero") else "jump"
    if jump:
        return "jump"
    if duck:
        return "duck"
    return "none"


# ---------------------------------------------------------------------------
# Асинхронный игрок
# ---------------------------------------------------------------------------
class AsyncDinoPlayer:
    """Управляет одной игрой в одном браузере и играет одним геномом NEAT."""

    def __init__(self, net, genome, page, worker_id: int):
        self.net = net
        self.genome = genome
        self.page = page
        self.worker_id = worker_id
        self.best_score = 0
        self.duck_held = False

    async def _apply_game_speed(self) -> None:
        if GAME_SPEED == 1.0:
            return
        try:
            await self.page.evaluate(
                """(mult) => {
                    const r = window.Runner && window.Runner.instance_;
                    if (!r || !r.config) { return; }
                    if (!window.__dinoBaseCfg) {
                        window.__dinoBaseCfg = {
                            SPEED: r.config.SPEED,
                            MAX_SPEED: r.config.MAX_SPEED,
                            ACCELERATION: r.config.ACCELERATION
                        };
                    }
                    const b = window.__dinoBaseCfg;
                    r.config.SPEED = b.SPEED * mult;
                    r.config.MAX_SPEED = b.MAX_SPEED * mult;
                    r.config.ACCELERATION = b.ACCELERATION * mult;
                    if (typeof r.currentSpeed === 'number') {
                        if (r.currentSpeed < r.config.SPEED) { r.currentSpeed = r.config.SPEED; }
                        if (r.currentSpeed > r.config.MAX_SPEED) { r.currentSpeed = r.config.MAX_SPEED; }
                    }
                }""",
                GAME_SPEED,
            )
        except Exception:  # noqa: BLE001
            pass

    async def _release_duck(self) -> None:
        if self.duck_held:
            try:
                await self.page.evaluate("() => window.__dino.release('ArrowDown', 'ArrowDown')")
            except Exception:  # noqa: BLE001
                pass
            self.duck_held = False

    async def apply_action(self, action: str) -> None:
        try:
            if action == "jump":
                await self._release_duck()
                await self.page.evaluate("() => window.__dino.press('Space', ' ')")
                await self.page.evaluate("() => window.__dino.release('Space', ' ')")
            elif action == "duck":
                if not self.duck_held:
                    await self.page.evaluate(
                        "() => window.__dino.press('ArrowDown', 'ArrowDown')"
                    )
                    self.duck_held = True
            else:
                await self._release_duck()
        except Exception:  # noqa: BLE001
            pass

    async def _read_state(self) -> dict:
        try:
            return await self.page.evaluate(JS_READ)
        except Exception as error:  # noqa: BLE001
            return {"status": "ERROR", "error": str(error)}

    async def _wait_engine(self, timeout: float = 20.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = await self._read_state()
            if state.get("status") not in ("NO_RUNNER", "ERROR"):
                return True
            await asyncio.sleep(0.05)
        return False

    async def _start_game(self) -> bool:
        """Перезапускает игру и ждёт её реального старта."""
        await self._release_duck()
        try:
            await self.page.evaluate(
                "() => { const r = window.Runner && window.Runner.instance_;"
                " if (r && typeof r.restart === 'function') { r.restart(); } }"
            )
        except Exception:  # noqa: BLE001
            pass
        await self._apply_game_speed()

        deadline = time.monotonic() + 6.0
        fallback = False
        while time.monotonic() < deadline:
            state = await self._read_state()
            status = state.get("status")
            if status == "OK":
                return True
            if not fallback and status in ("NOT_PLAYING", "DEAD"):
                fallback = True
                try:
                    await self.page.evaluate("() => window.__dino.press('Space', ' ')")
                    await self.page.evaluate("() => window.__dino.release('Space', ' ')")
                except Exception:  # noqa: BLE001
                    pass
            await asyncio.sleep(0.03)
        return False

    async def play(self, y: dict, report_every: float = 0.0) -> float:
        """Играет одну игру. Возвращает счёт (очки).

        ``y`` — небольшой словарь состояния воркера (в него пишется лучший
        счёт). Если ``report_every > 0`` — периодически печатает прогресс
        (используется в режиме показа в видимом окне).
        """
        y["best"] = 0
        started = time.monotonic()
        last_report = started

        while True:
            state = await self._read_state()
            status = state.get("status")

            if status == "DEAD":
                await self._release_duck()
                self.best_score = max(self.best_score, int(state.get("score", 0) or 0))
                y["best"] = self.best_score
                return self.best_score

            if status in ("NO_RUNNER", "ERROR"):
                await asyncio.sleep(POLL_DELAY)
                continue

            if status == "NOT_PLAYING":
                await self._start_game()
                continue

            score = int(state.get("score", 0) or 0)
            if score > self.best_score:
                self.best_score = score
            y["best"] = self.best_score

            outputs = self.net.activate(build_inputs(state))
            await self.apply_action(decide_action(outputs, state))

            now = time.monotonic()
            if report_every > 0 and now - last_report >= report_every:
                print(
                    f"    счёт {self.best_score:>6}  "
                    f"скорость {float(state.get('speed', 0) or 0):.1f}  "
                    f"время {now - started:.0f}с",
                    flush=True,
                )
                last_report = now

            # Предохранитель от бесконечной «идеальной» игры.
            if now - started > EPISODE_CAP:
                await self._release_duck()
                return self.best_score

            await asyncio.sleep(POLL_DELAY)


# ---------------------------------------------------------------------------
# Пул браузеров
# ---------------------------------------------------------------------------
# Аргументы Chromium, оптимизированные под запуск МНОЖЕСТВА инстансов:
# отключаем GPU/лишние процессы, фоновое троттлинг-замедление вкладок,
# сжатие и восстановление вкладок, чтобы 15 окон работали стабильно.
_CHROMIUM_ARGS = [
    "--disable-gpu",
    "--disable-extensions",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-dev-shm-usage",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
    "--disable-features=CalculateNativeWinOcclusion,TranslateUI",
    "--mute-audio",
    "--hide-scrollbars",
]


class BrowserPool:
    """Держит один Chromium и создаёт изолированные контексты (вкладки)."""

    def __init__(self, playwright):
        self._playwright = playwright
        self.browser = None

    async def start(self) -> None:
        self.browser = await self._playwright.chromium.launch(
            headless=HEADLESS,
            args=list(_CHROMIUM_ARGS),
        )

    async def new_player_page(self):
        """Новый изолированный контекст (свои cookies/storage) + загрузка игры."""
        context = await self.browser.new_context(
            viewport={"width": 800, "height": 600},
            device_scale_factor=1,
        )
        page = await context.new_page()
        await page.goto(GAME_URL, wait_until="domcontentloaded", timeout=30000)
        await page.evaluate(JS_INSTALL)
        return context, page

    async def close(self) -> None:
        if self.browser is not None:
            try:
                await self.browser.close()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Общие утилиты NEAT
# ---------------------------------------------------------------------------
def _load_config() -> neat.Config:
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(NEAT_CONFIG_PATH),
    )
    # Размер популяции NEAT должен совпадать с числом браузеров.
    config.pop_size = WORKERS
    return config


def _genome_path() -> Path:
    return _HERE / f"{SAVE_PREFIX}.pickle"


def _save_best(genome, config) -> None:
    """Сохраняет лучший геном в pickle (можно загрузить и прогнать отдельно)."""
    try:
        with open(_genome_path(), "wb") as handle:
            pickle.dump(genome, handle)
        with open(_HERE / f"{SAVE_PREFIX}_config.pickle", "wb") as handle:
            pickle.dump(config.genome_config, handle)
    except Exception as error:  # noqa: BLE001
        print(f"  Не удалось сохранить геном: {error}")


def _load_saved_genome():
    path = _genome_path()
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    except Exception as error:  # noqa: BLE001
        print(f"  Не удалось загрузить геном {path}: {error}")
        return None


def _noop_eval(genomes, config) -> None:  # pragma: no cover
    """Оцениватель-заглушка: геномы уже оценены вручную, тут ничего не делаем."""
    return


class _Showcase:
    """Хранит лучшие найденные геномы для финального показа."""

    def __init__(self, limit: int = SHOWCASE_LIMIT):
        self._limit = limit
        self._items: list[tuple[int, object]] = []

    def remember(self, genome, score: int) -> None:
        self._items.append((int(score), genome))
        self._items.sort(key=lambda item: item[0], reverse=True)
        del self._items[self._limit:]

    def genomes(self) -> list:
        return [genome for _, genome in self._items]

    @property
    def best_score(self) -> int:
        return self._items[0][0] if self._items else 0


# ---------------------------------------------------------------------------
# Основной цикл обучения (asyncio)
# ---------------------------------------------------------------------------
async def train() -> tuple[list, neat.Config]:
    """Обучает популяцию и возвращает (список лучших геномов, config)."""
    print("=" * 70)
    print("  ЭТАП 1/2 — ОБУЧЕНИЕ Dino-бота через NEAT")
    print("=" * 70)
    print(f"  Браузеров (особей) : {WORKERS}")
    print(f"  Скрытые окна       : {HEADLESS}")
    print(f"  Поколений (макс)   : {'∞' if MAX_GENERATIONS == 0 else MAX_GENERATIONS}")
    print(f"  Целевой счёт       : {TARGET_SCORE}")
    print(f"  Множитель скорости : {GAME_SPEED:g}")
    print(f"  Пауза между шагами : {POLL_DELAY * 1000:.0f} мс")
    print(f"  Лимит на игру      : {EPISODE_CAP:.0f} с")
    print("=" * 70)

    config = _load_config()
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(neat.Checkpointer(10, filename_prefix="neat-checkpoint-"))

    showcase = _Showcase()
    best_overall = {"genome": None, "score": 0}

    async with async_playwright() as playwright:
        pool = BrowserPool(playwright)
        await pool.start()
        print("Chromium запущен. Начинаем эволюцию...\n")

        try:
            generation = 0
            while MAX_GENERATIONS == 0 or generation < MAX_GENERATIONS:
                generation += 1
                # population.population — словарь {genome_id: genome}.
                genomes = [genome for _, genome in population.population.items()]
                if len(genomes) < WORKERS:
                    genomes += [genomes[-1]] * (WORKERS - len(genomes))
                genomes = genomes[:WORKERS]

                nets, y_states = [], []
                for _ in genomes:
                    nets.append(None)
                    y_states.append({"best": 0, "tick": generation})

                # Строим сети и параллельно готовим вкладки.
                for i, genome in enumerate(genomes):
                    nets[i] = neat.nn.FeedForwardNetwork.create(genome, config)
                pages = await asyncio.gather(
                    *[pool.new_player_page() for _ in genomes]
                )

                print(f"\n--- Поколение {generation} ---")
                t0 = time.monotonic()
                results = await asyncio.gather(
                    *[
                        AsyncDinoPlayer(nets[i], genomes[i], pages[i][1], i).play(y_states[i])
                        for i in range(len(genomes))
                    ]
                )
                gen_time = time.monotonic() - t0

                # Приспособленность = счёт (очки). Небольшой бонус за время
                # помогает различать особей с нулевым счётом на старте.
                for i, genome in enumerate(genomes):
                    genome.fitness = float(results[i]) + 0.001 * gen_time

                for context, _ in pages:
                    try:
                        await context.close()
                    except Exception:  # noqa: BLE001
                        pass

                best_idx = max(range(len(results)), key=lambda i: results[i])
                gen_best_score = results[best_idx]
                avg_score = sum(results) / len(results) if results else 0.0

                print(
                    f"  Лучший счёт: {gen_best_score:>6}  |  "
                    f"средний счёт: {avg_score:>8.1f}  |  "
                    f"время: {gen_time:.1f} c"
                )

                # Запоминаем лучших особей для финального показа.
                for i, genome in enumerate(genomes):
                    if results[i] > 0:
                        showcase.remember(genome, results[i])

                # Сохраняем геном, если он побил абсолютный рекорд за всё время.
                if gen_best_score > best_overall["score"]:
                    best_overall["score"] = gen_best_score
                    best_overall["genome"] = genomes[best_idx]
                    _save_best(genomes[best_idx], config)

                if gen_best_score >= TARGET_SCORE:
                    print(f"\nЦель {TARGET_SCORE} очков достигнута (счёт {gen_best_score}).")
                    break

                # Эволюционный шаг NEAT.
                population.run(_noop_eval, 1)

        finally:
            await pool.close()

    print("\n" + "=" * 70)
    print(f"  Обучение завершено. Лучший счёт: {showcase.best_score}")
    print(f"  Лучший геном сохранён в: {_genome_path().name}")
    print("=" * 70)
    return showcase.genomes(), config


# ---------------------------------------------------------------------------
# Показ лучшего бота в ВИДИМОМ окне
# ---------------------------------------------------------------------------
async def showcase_best(candidates: list, config: neat.Config) -> None:
    """Открывает видимое окно Chromium и играет лучшими геномами по кругу."""
    global HEADLESS

    if not candidates:
        print(
            "\nНет обученного генома для показа.\n"
            "Сначала обучите бота:  python dino.py"
        )
        return

    HEADLESS = False

    print("=" * 70)
    print("  ЭТАП 2/2 — ПОКАЗ ЛУЧШЕГО БОТА (видимое окно Chromium)")
    print("=" * 70)
    print(f"  Особей к показу    : {len(candidates)}")
    print(f"  Игр к показу       : {'∞' if SHOW_EPISODES == 0 else SHOW_EPISODES}")
    print(f"  Множитель скорости : {GAME_SPEED:g}")
    print("  Закройте окно или нажмите Ctrl+C, чтобы остановить.")
    print("=" * 70)

    async with async_playwright() as playwright:
        pool = BrowserPool(playwright)
        await pool.start()
        context, page = await pool.new_player_page()

        session_best = 0
        episode = 0
        idx = 0
        try:
            while SHOW_EPISODES == 0 or episode < SHOW_EPISODES:
                genome = candidates[idx % len(candidates)]
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                player = AsyncDinoPlayer(net, genome, page, 0)
                player.best_score = 0
                episode += 1
                print(
                    f"\n--- Игра №{episode} (особь #{idx % len(candidates) + 1}) ---",
                    flush=True,
                )
                try:
                    score = await player.play({}, report_every=3.0)
                except Exception as error:  # noqa: BLE001
                    print(f"  Ошибка во время игры: {error}")
                    score = 0
                session_best = max(session_best, score)
                print(
                    f"  Итог игры: {score}  |  лучший за сессию: {session_best}",
                    flush=True,
                )
                idx += 1
                await asyncio.sleep(0.8)  # пауза перед новым рестартом
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\nОстановлено.")

        print(f"\nЛучший счёт за показ: {session_best}")
        try:
            await context.close()
        except Exception:  # noqa: BLE001
            pass
        await pool.close()


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------
async def _run() -> None:
    candidates: list = []
    config = _load_config()

    if TRAIN_ENABLED:
        candidates, config = await train()
    else:
        genome = _load_saved_genome()
        if genome is not None:
            candidates = [genome]
            print(f"Загружен сохранённый геном: {_genome_path().name}")

    if SHOW_ENABLED:
        await showcase_best(candidates, config)
    else:
        print("\nПоказ отключён (DINO_SHOW=0).")


def main() -> None:
    _setup_console()
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.")


def _setup_console() -> None:
    """Делает вывод устойчивым к кодировке консоли Windows (cp1251 и т.п.)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
