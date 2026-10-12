"""Exact incoming neighborhoods for target batches in a causal graph."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TargetSubgraph:
    nodes: np.ndarray
    edge_index: np.ndarray
    target_positions: np.ndarray


class IncomingNeighborhoods:
    """CSR index: a batch retains all ancestors needed by the GNN layers.

    No labels, random neighbor sampling or future edges are introduced. In
    evaluation mode, target logits agree with full-graph inference (up to
    floating-point reduction order). Outer nodes provide input context only.
    """

    def __init__(self, edge_index: np.ndarray, node_count: int):
        edges = np.asarray(edge_index, dtype=np.int64)
        if edges.ndim != 2 or edges.shape[0] != 2 or node_count < 0:
            raise ValueError("Expected shape (2, E) and a nonnegative node count")
        if edges.size and (edges.min() < 0 or edges.max() >= node_count):
            raise ValueError("Edge endpoints are outside the graph")
        order = np.argsort(edges[1], kind="stable")
        self.sources = edges[0, order]
        counts = np.bincount(edges[1], minlength=node_count)
        self.pointer = np.concatenate(([0], np.cumsum(counts)))
        self.node_count = node_count

    def mean_features(self, features: np.ndarray) -> np.ndarray:
        """Cache the immutable first-layer neighborhood means, without labels."""
        from scipy.sparse import csr_matrix

        if len(features) != self.node_count:
            raise ValueError("Feature rows must match the graph")
        counts = np.diff(self.pointer)
        weights = np.repeat(1.0 / np.maximum(counts, 1), counts).astype(np.float32)
        matrix = csr_matrix((weights, self.sources, self.pointer), shape=(self.node_count, self.node_count))
        return matrix @ features

    def _incoming(self, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        counts = self.pointer[nodes + 1] - self.pointer[nodes]
        total = int(counts.sum())
        if total == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        starts = np.repeat(self.pointer[nodes], counts)
        offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        return self.sources[starts + offsets], np.repeat(nodes, counts)

    def extract(self, targets: np.ndarray, depth: int) -> TargetSubgraph:
        targets = np.asarray(targets, dtype=np.int64)
        if targets.ndim != 1 or not len(targets) or depth < 0:
            raise ValueError("A nonempty target vector and nonnegative depth are required")
        if targets.min() < 0 or targets.max() >= self.node_count:
            raise ValueError("Targets are outside the graph")
        nodes = np.unique(targets)
        frontier = nodes
        for _ in range(depth):
            sources, _ = self._incoming(frontier)
            frontier = np.setdiff1d(np.unique(sources), nodes, assume_unique=True)
            nodes = np.union1d(nodes, frontier)
            if not len(frontier):
                break
        sources, destinations = self._incoming(nodes)
        local_source = np.searchsorted(nodes, sources)
        valid = local_source < len(nodes)
        valid[valid] &= nodes[local_source[valid]] == sources[valid]
        edges = np.stack((local_source[valid], np.searchsorted(nodes, destinations[valid])))
        return TargetSubgraph(nodes, edges, np.searchsorted(nodes, targets))
