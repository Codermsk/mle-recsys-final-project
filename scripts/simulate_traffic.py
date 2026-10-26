"""Нагрузка на сервис для проверки мониторинга.

Шлёт одиночные и пакетные запросы рекомендаций. Часть идентификаторов - клиенты из
хранилища признаков, часть - несуществующие (новые клиенты, ответ из fallback).

Использование: python scripts/simulate_traffic.py --duration 300 --rps 5
"""
import argparse
import random
import time
from pathlib import Path

import pandas as pd
import requests

FEATURES_PATH = Path(__file__).resolve().parents[1] / "data" / "features_latest.parquet"


def load_client_ids(n: int = 50_000) -> list[int]:
    if not FEATURES_PATH.exists():
        return list(range(15_889, 15_889 + n))
    ids = pd.read_parquet(FEATURES_PATH, columns=[]).index
    return ids.to_series().sample(min(n, len(ids)), random_state=0).tolist()


def random_client(known: list[int], new_share: float) -> int:
    if random.random() < new_share:
        return random.randint(2_000_000, 3_000_000)
    return random.choice(known)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--duration", type=int, default=300, help="секунд")
    parser.add_argument("--rps", type=float, default=5)
    parser.add_argument("--new-share", type=float, default=0.1, help="доля новых клиентов")
    parser.add_argument("--batch-share", type=float, default=0.2, help="доля пакетных запросов")
    args = parser.parse_args()

    known = load_client_ids()
    stats = {"ok": 0, "error": 0}
    finish = time.time() + args.duration
    while time.time() < finish:
        try:
            if random.random() < args.batch_share:
                ids = [random_client(known, args.new_share) for _ in range(random.randint(2, 100))]
                resp = requests.post(f"{args.url}/recommendations", json={"client_ids": ids}, timeout=10)
            else:
                client = random_client(known, args.new_share)
                resp = requests.get(f"{args.url}/recommendations/{client}", timeout=10)
            stats["ok" if resp.ok else "error"] += 1
        except requests.RequestException:
            stats["error"] += 1
        time.sleep(random.expovariate(args.rps))
    print(stats)


if __name__ == "__main__":
    main()
