from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class FillerTask:
    prompt: str
    answer: str
    latent: int
    task_type: str
    filler: str


def _dot_filler(rng: random.Random, length: int) -> str:
    return " ".join("." * rng.randint(1, 4) for _ in range(length))


def make_task(rng: random.Random) -> FillerTask:
    task_type = rng.choice(["sum", "count"])
    if task_type == "sum":
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        filler = _dot_filler(rng, rng.randint(3, 8))
        prompt = (
            "Solve the arithmetic problem. Do not expose intermediate variables. "
            f"Filler tokens: {filler}\nWhat is {a} + {b}?\nAnswer:"
        )
        return FillerTask(prompt, str(a + b), a + b, task_type, filler)
    count = rng.randint(2, 8)
    filler = _dot_filler(rng, count)
    prompt = (
        "Count the dot groups and return only the integer. "
        f"Filler tokens: {filler}\nHow many groups are there?\nAnswer:"
    )
    return FillerTask(prompt, str(count), count, task_type, filler)


def make_dataset(size: int, seed: int) -> list[FillerTask]:
    rng = random.Random(seed)
    return [make_task(rng) for _ in range(size)]


def write_jsonl(tasks: list[FillerTask], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(asdict(task)) + "\n")
