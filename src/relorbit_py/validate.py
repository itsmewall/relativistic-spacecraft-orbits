# src/relorbit_py/validate.py
from __future__ import annotations

from . import engine_hello, engine


def smoke() -> None:
    print(engine_hello())
    y = engine.rk4_decay(y0=1.0, k=1.0, t0=0.0, tf=1.0, n_steps=1000)
    print("rk4_decay final y:", y[-1])


if __name__ == "__main__":
    smoke()
