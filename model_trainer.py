import random
from collections import Counter
from typing import Dict, List, Tuple, Optional

import numpy as np

from utils import multiset_overlap, greedy_multiset_mapping, to_digits

class LotteryModelTrainer:
    def __init__(self, n_digits: int = 4):
        self.n_digits = n_digits
        self.digit_freq_ = np.ones(10)  # Laplace smoothing
        self.trans_counts_ = Counter()
        self.alpha_ = 0.5  # smoothing

    def fit(self, history_numbers: List[Tuple[int, ...]]):
        # Frequency
        freq = Counter()
        for t in history_numbers:
            for d in t:
                freq[d] += 1
        self.digit_freq_ = np.array([freq.get(i, 0) + 1.0 for i in range(10)], dtype=float)
        self.digit_freq_ /= self.digit_freq_.sum()

        # Positionless transitions lag-1
        self.trans_counts_ = Counter()
        for i in range(len(history_numbers) - 1):
            a, b = history_numbers[i], history_numbers[i + 1]
            for x, y in greedy_multiset_mapping(a, b):
                self.trans_counts_[(x, y)] += 1

    def transition_probs(self, boosts: Optional[Dict[str, float]] = None, boost_weight: float = 1.0) -> Dict[Tuple[int, int], float]:
        cnt = self.trans_counts_.copy()
        totals = Counter()
        for (x, y), c in cnt.items():
            totals[x] += c
        if boosts:
            for k, v in boosts.items():
                try:
                    xs, ys = k.split("->")
                    x, y = int(xs), int(ys)
                    w = float(v) * boost_weight
                    cnt[(x, y)] += w
                    totals[x] += w
                except Exception:
                    continue
        probs: Dict[Tuple[int, int], float] = {}
        alpha = self.alpha_
        for x in range(10):
            denom = totals[x] + alpha * 10
            for y in range(10):
                c = cnt.get((x, y), 0.0)
                probs[(x, y)] = (c + alpha) / max(denom, 1e-9)
        return probs

    def base_predict_distribution(self, last_draw: Tuple[int, ...], boosts: Optional[Dict[str, float]] = None, boost_weight: float = 1.0):
        probs = self.transition_probs(boosts=boosts, boost_weight=boost_weight)
        per_digit = []
        for d in last_draw:
            row = [(y, probs.get((d, y), 1e-6)) for y in range(10)]
            row.sort(key=lambda t: t[1], reverse=True)
            per_digit.append(row[:4])
        cands: Dict[Tuple[int, ...], float] = {}
        def dfs(i, cur, p):
            if i == len(per_digit):
                t = tuple(cur)
                cands[t] = cands.get(t, 0.0) + p
                return
            for y, py in per_digit[i]:
                cur.append(y)
                dfs(i + 1, cur, p * py)
                cur.pop()
        dfs(0, [], 1.0)
        total = sum(cands.values()) or 1.0
        ranked = sorted(((t, v / total) for t, v in cands.items()), key=lambda kv: kv[1], reverse=True)
        return ranked

    def mutate(self, t: Tuple[int, ...], strength: float) -> Tuple[int, ...]:
        max_step = max(1, int(5 * strength))
        return tuple((d + random.randint(-max_step, max_step)) % 10 for d in t)

    def generate_predictions(self, last_draw: Tuple[int, ...], n: int, mutation_strength: float,
                             boosts: Optional[Dict[str, float]] = None, boost_weight: float = 1.0) -> List[Tuple[int, ...]]:
        ranked = self.base_predict_distribution(last_draw, boosts=boosts, boost_weight=boost_weight)
        seeds = [t for t, _ in ranked[:max(5, n//2)]]
        preds = set(seeds)
        while len(preds) < n:
            base = random.choice(seeds)
            preds.add(self.mutate(base, mutation_strength))
        return list(preds)[:n]

    # Optional model interfaces for RF/NN compatibility with your original scaffold.
    # They’re placeholders here so the dashboard runs end-to-end.
    def train_models(self, X_train, y_train=None):
        return

    def get_predictions(self, scaled_input, mutation_strength: float = 0.4):
        # Produce two “families” by sampling twice with different seeds
        last_draw = (0, 0, 0, 0)  # placeholder; not used in this path
        rf_mut = []
        nn_mut = []
        return rf_mut, nn_mut

def best_overlap(preds: List[Tuple[int, ...]], actual: Tuple[int, ...]) -> int:
    if not preds:
        return 0
    return max(multiset_overlap(p, actual) for p in preds)
