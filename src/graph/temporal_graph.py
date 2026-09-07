from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalGraph:
    edge_index: np.ndarray
    node_count: int

    @property
    def edge_count(self) -> int:
        return int(self.edge_index.shape[1])


def build_temporal_entity_graph(
    frame: pd.DataFrame,
    time_column: str,
    entity_groups: list[list[str]],
    window_seconds: int,
    max_previous_neighbors: int = 3,
) -> TemporalGraph:
    """Build past-to-current edges only, preventing future-to-past leakage."""
    if max_previous_neighbors < 1:
        raise ValueError("max_previous_neighbors must be positive")
    ordered = frame.reset_index(drop=True)
    times = ordered[time_column].to_numpy(dtype=np.int64)
    if len(times) > 1 and np.any(times[1:] < times[:-1]):
        raise ValueError("Frame must be ordered by time before graph construction")
    edges: set[tuple[int, int]] = set()
    for columns in entity_groups:
        usable = [c for c in columns if c in ordered.columns]
        if not usable:
            continue
        history: dict[str, deque[tuple[int, int]]] = defaultdict(deque)
        pending = []
        last_time = None
        for current in range(len(ordered)):
            current_time = int(times[current])
            if current_time != last_time:
                for entity, node, timestamp in pending:
                    history[entity].append((node, timestamp))
                    while len(history[entity]) > max_previous_neighbors:
                        history[entity].popleft()
                pending = []
                last_time = current_time
            entities = {str(ordered.at[current, c]) for c in usable if pd.notna(ordered.at[current, c])}
            for entity in entities:
                queue = history[entity]
                while queue and current_time - queue[0][1] > window_seconds:
                    queue.popleft()
                for previous, _ in queue:
                    edges.add((previous, current))
                pending.append((entity, current, current_time))
    if edges:
        edge_index = np.asarray(sorted(edges), dtype=np.int64).T
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
    return TemporalGraph(edge_index=edge_index, node_count=len(ordered))
