"""Group matched records into clusters (one cluster = one book) with union-find.

Cannot-link constraint: records flagged `exclusive` are known to be distinct books (Goodreads
records, de-duplicated by bookId). A union that would put two of them in one cluster is
refused. This stops chains like
    Goodreads "Huntress" -isbn- BX "Night World" -title- BX "Night World" -isbn- Goodreads "Black Dawn"
where a source stored a series name as the title of every volume.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class UnionFind:
    """Disjoint sets over 0..n-1 with path halving, union by size and a cannot-link flag."""

    def __init__(self, n: int, exclusive: np.ndarray | None = None) -> None:
        self.parent = np.arange(n)
        self.size = np.ones(n, dtype=np.int64)
        self.exclusive = np.zeros(n, dtype=bool) if exclusive is None else np.asarray(exclusive, dtype=bool).copy()

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    def union(self, a: int, b: int) -> bool:
        """Merge the sets of a and b. Returns False if refused by the cannot-link constraint."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if self.exclusive[ra] and self.exclusive[rb]:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.exclusive[ra] |= self.exclusive[rb]
        return True


def assign_clusters(n_records: int, edges: pd.DataFrame, exclusive: np.ndarray | None = None) -> np.ndarray:
    """Cluster label per record: records connected by accepted edges share a label.

    Edges are applied in the given order, so put the most trustworthy method first.
    Labels are the smallest rid in the cluster, so they don't depend on union order.
    """
    uf = UnionFind(n_records, exclusive)
    refused = sum(not uf.union(int(a), int(b)) for a, b in zip(edges["rid_a"].to_numpy(), edges["rid_b"].to_numpy()))
    if refused:
        logger.info("cluster: refused %d edges that would merge two exclusive records", refused)
    roots = np.fromiter((uf.find(i) for i in range(n_records)), dtype=np.int64, count=n_records)
    return pd.Series(np.arange(n_records)).groupby(roots).transform("min").to_numpy()
