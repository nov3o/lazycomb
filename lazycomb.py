"""Lazy combinatorial sequences: itertools' product, permutations and combinations as sequences you can index.

    >>> p = Product(range(10), repeat=100)
    >>> big = Product(p, repeat=100)          # nothing is expanded
    >>> big.size == 10 ** 10000
    True
    >>> x = big[123456789]                   # unrank: number -> element
    >>> big.index(x)                          # rank: element -> number
    123456789

Every sequence has `.size`, `[i]` (also negative and slices), `.index(x)`, `in` and iteration, in exactly the order of
the matching itertools function. A sequence only reads its pools through their size and `[i]`, so pools can be other
lazy sequences of any size: unranking costs O(r) to O(r^2) pool lookups, never O(size). Python's `len()` is limited to
2**63 - 1, so use `.size` for big sequences.
"""

from math import comb, perm


def _size(pool) -> int:
    if isinstance(pool, Lazy):
        return pool.size
    if isinstance(pool, range):  # len(range(10**100)) overflows
        return max(0, (pool.stop - pool.start + pool.step - (1 if pool.step > 0 else -1)) // pool.step)
    return len(pool)


def _pool(pool):
    """Keep sequences as they are (they are indexable); materialize plain iterators, like itertools does."""
    return pool if isinstance(pool, (Lazy, range, list, tuple, str, bytes)) else tuple(pool)


def _index(pool, x) -> int:
    if isinstance(pool, range):
        if x not in pool:
            raise ValueError(f"{x!r} is not in the pool")
        return (x - pool.start) // pool.step
    return pool.index(x)


class Lazy:
    """Base: subclasses define `size`, `_get(i)` for 0 <= i < size and `index(x)`."""

    size: int

    def __len__(self):
        return self.size  # raises OverflowError past 2**63 - 1: use .size

    def __getitem__(self, i):
        if isinstance(i, slice):
            return Slice(self, i)
        if i < 0:
            i += self.size
        if not 0 <= i < self.size:
            raise IndexError(f"index out of range for a sequence of size {self.size}")
        return self._get(i)

    def __iter__(self):
        for i in range(self.size):
            yield self._get(i)

    def __contains__(self, x):
        try:
            self.index(x)
            return True
        except (ValueError, TypeError):
            return False

    def __bool__(self):
        return self.size > 0


class Seq(Lazy):
    """A plain sequence (list, tuple, range, ...) seen as a lazy one."""

    def __init__(self, pool):
        self.pool = _pool(pool)
        self.size = _size(self.pool)

    def _get(self, i):
        return self.pool[i]

    def index(self, x):
        return _index(self.pool, x)

    def __repr__(self):
        return f"Seq({self.pool!r})"


class Product(Lazy):
    """itertools.product: mixed radix, the last pool changes fastest. Iteration is an odometer."""

    def __init__(self, *pools, repeat: int = 1):
        self.pools = [_pool(p) for p in pools] * repeat
        self.sizes = [_size(p) for p in self.pools]
        self.size = 1
        for s in self.sizes:
            self.size *= s

    def _get(self, i):
        out = []
        for pool, s in zip(reversed(self.pools), reversed(self.sizes)):
            i, d = divmod(i, s)
            out.append(pool[d])
        return tuple(reversed(out))

    def index(self, x):
        if len(x) != len(self.pools):
            raise ValueError("wrong length")
        i = 0
        for pool, s, v in zip(self.pools, self.sizes, x):
            i = i * s + _index(pool, v)
        return i

    def __iter__(self):
        if self.size == 0:
            return
        digits = [0] * len(self.pools)
        current = [pool[0] for pool in self.pools]
        while True:
            yield tuple(current)
            k = len(digits) - 1
            while k >= 0 and digits[k] + 1 == self.sizes[k]:
                digits[k] = 0
                current[k] = self.pools[k][0]
                k -= 1
            if k < 0:
                return
            digits[k] += 1
            current[k] = self.pools[k][digits[k]]

    def __repr__(self):
        return f"Product({', '.join(map(repr, self.pools))})"


def _nth_unused(j: int, used_sorted: list[int]) -> int:
    """The j-th (0-based) index not in `used_sorted`, in O(len(used))."""
    for u in used_sorted:
        if u <= j:
            j += 1
        else:
            break
    return j


class Permutations(Lazy):
    """itertools.permutations: the factorial number system (Lehmer code). Works for huge pools: picking the j-th
    unused element only looks at the r - 1 elements already picked."""

    def __init__(self, pool, r: int | None = None):
        self.pool = _pool(pool)
        self.n = _size(self.pool)
        self.r = self.n if r is None else r
        self.size = perm(self.n, self.r) if 0 <= self.r <= self.n else 0

    def _indices(self, i):
        used, out = [], []
        for k in range(self.r):
            block = perm(self.n - k - 1, self.r - k - 1)
            j, i = divmod(i, block)
            v = _nth_unused(j, used)
            out.append(v)
            used.insert(_bisect(used, v), v)
        return out

    def _get(self, i):
        return tuple(self.pool[v] for v in self._indices(i))

    def index(self, x):
        if len(x) != self.r:
            raise ValueError("wrong length")
        idx = [_index(self.pool, v) for v in x]
        if len(set(idx)) != self.r:
            raise ValueError("repeated element")
        i, used = 0, []
        for k, v in enumerate(idx):
            j = v - _bisect(used, v)  # rank of v among the unused
            i += j * perm(self.n - k - 1, self.r - k - 1)
            used.insert(_bisect(used, v), v)
        return i

    def __repr__(self):
        return f"Permutations({self.pool!r}, {self.r})"


def _bisect(a: list[int], x: int) -> int:
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


class Combinations(Lazy):
    """itertools.combinations: the combinatorial number system, with a binary search per position, so huge pools
    cost O(r log n) binomials."""

    def __init__(self, pool, r: int):
        self.pool = _pool(pool)
        self.n = _size(self.pool)
        self.r = r
        self.size = comb(self.n, r) if 0 <= r <= self.n else 0

    def _indices(self, i):
        out, start = [], 0
        for k in range(self.r):
            m = self.r - k
            total = comb(self.n - start, m)
            # combinations with first element < v: total - C(n - v, m); find the largest v with that <= i
            lo, hi = start, self.n - m
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if total - comb(self.n - mid, m) <= i:
                    lo = mid
                else:
                    hi = mid - 1
            i -= total - comb(self.n - lo, m)
            out.append(lo)
            start = lo + 1
        return out

    def _get(self, i):
        return tuple(self.pool[v] for v in self._indices(i))

    def _rank(self, idx):
        i, start = 0, 0
        for k, v in enumerate(idx):
            m = self.r - k
            i += comb(self.n - start, m) - comb(self.n - v, m)
            start = v + 1
        return i

    def index(self, x):
        if len(x) != self.r:
            raise ValueError("wrong length")
        idx = [_index(self.pool, v) for v in x]
        if any(a >= b for a, b in zip(idx, idx[1:])):
            raise ValueError("not a combination in pool order")
        return self._rank(idx)

    def __repr__(self):
        return f"Combinations({self.pool!r}, {self.r})"


class CombinationsWithReplacement(Lazy):
    """itertools.combinations_with_replacement, through the bijection c_k -> c_k + k onto plain combinations of
    n + r - 1 (it keeps the lexicographic order)."""

    def __init__(self, pool, r: int):
        self.pool = _pool(pool)
        self.n = _size(self.pool)
        self.r = r
        self._plain = Combinations(range(self.n + r - 1), r) if self.n else None
        self.size = self._plain.size if self.n else (1 if r == 0 else 0)

    def _get(self, i):
        if self._plain is None:  # empty pool, r = 0
            return ()
        return tuple(self.pool[v - k] for k, v in enumerate(self._plain._indices(i)))

    def index(self, x):
        if len(x) != self.r:
            raise ValueError("wrong length")
        idx = [_index(self.pool, v) for v in x]
        if any(a > b for a, b in zip(idx, idx[1:])):
            raise ValueError("not in pool order")
        if self._plain is None:
            return 0
        return self._plain._rank([v + k for k, v in enumerate(idx)])

    def __repr__(self):
        return f"CombinationsWithReplacement({self.pool!r}, {self.r})"


class Map(Lazy):
    """f applied to each element; `inverse` (optional) makes `.index` work."""

    def __init__(self, f, pool, inverse=None):
        self.f, self.pool, self.inverse = f, _pool(pool), inverse
        self.size = _size(self.pool)

    def _get(self, i):
        return self.f(self.pool[i])

    def index(self, x):
        if self.inverse is None:
            raise ValueError("Map without an inverse can't be ranked")
        return _index(self.pool, self.inverse(x))


class Chain(Lazy):
    """itertools.chain over sequences."""

    def __init__(self, *pools):
        self.pools = [_pool(p) for p in pools]
        self.sizes = [_size(p) for p in self.pools]
        self.size = sum(self.sizes)

    def _get(self, i):
        for pool, s in zip(self.pools, self.sizes):
            if i < s:
                return pool[i]
            i -= s

    def index(self, x):
        offset = 0
        for pool, s in zip(self.pools, self.sizes):
            try:
                return offset + _index(pool, x)
            except (ValueError, TypeError):
                offset += s
        raise ValueError(f"{x!r} is not in the chain")


class Slice(Lazy):
    """A lazy view of seq[start:stop:step]."""

    def __init__(self, seq: Lazy, s: slice):
        self.seq = seq
        self.range = range(seq.size)[s]
        self.size = _size(self.range)

    def _get(self, i):
        return self.seq._get(self.range[i])

    def index(self, x):
        return _index(self.range, self.seq.index(x))
