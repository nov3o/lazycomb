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


def _one_shot(pool) -> bool:
    try:
        return iter(pool) is pool
    except TypeError:
        raise TypeError(f"{type(pool).__name__!r} object is not iterable") from None


def _pool(pool, once: bool = False):
    """Pools are never copied. A one-shot iterator (a generator, `iter(...)`) can only be read once, so it is only
    accepted where it is (the first pool of a Product); anywhere else, pass something re-iterable: a range, list,
    lazy sequence, or any object whose __iter__ returns a fresh iterator."""
    if not once and _one_shot(pool):
        raise TypeError("this pool is a one-shot iterator and would have to be read more than once; "
                        "pass a re-iterable (range, list, lazy sequence, ...) instead")
    return pool


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
        pools = list(pools) * repeat
        if repeat > 1 and pools and _one_shot(pools[0]):
            raise TypeError("repeat > 1 reads the first pool more than once: pass a re-iterable")
        self.pools = [_pool(p, once=(i == 0)) for i, p in enumerate(pools)]

    @property
    def sizes(self):
        return [_size(p) for p in self.pools]

    @property
    def size(self):
        n = 1
        for s in self.sizes:
            n *= s
        return n

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
        """An odometer of live iterators, one per pool: when the last one runs out it is restarted and the one
        before it advances. Nothing is indexed, sized or copied, so pools can be any re-iterables, of any size."""
        n = len(self.pools)
        if n == 0:
            yield ()
            return
        its, current = [], []
        for pool in self.pools:
            it = iter(pool)
            try:
                current.append(next(it))
            except StopIteration:
                return
            its.append(it)
        while True:
            yield tuple(current)
            k = n - 1
            while True:
                try:
                    current[k] = next(its[k])
                    break
                except StopIteration:
                    if k == 0:
                        return
                    its[k] = iter(self.pools[k])  # restart this wheel ...
                    current[k] = next(its[k])
                    k -= 1  # ... and advance the previous one

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

    def __iter__(self):
        """Next r-permutation of indices in lexicographic order: O(r) per step, only r indices held."""
        n, r = self.n, self.r
        if not 0 <= r <= n:
            return
        idx = list(range(r))
        while True:
            yield tuple(self.pool[i] for i in idx)
            used = set(idx)
            for k in range(r - 1, -1, -1):
                used.discard(idx[k])
                nxt = idx[k] + 1
                while nxt in used:
                    nxt += 1
                if nxt < n:
                    idx[k] = nxt
                    used.add(nxt)
                    fill, v = k + 1, 0
                    while fill < r:  # smallest unused indices, in order
                        while v in used:
                            v += 1
                        idx[fill] = v
                        used.add(v)
                        fill += 1
                    break
            else:
                return

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

    def __iter__(self):
        """itertools' index algorithm: O(1) amortized per step, only r indices held."""
        n, r = self.n, self.r
        if not 0 <= r <= n:
            return
        idx = list(range(r))
        yield tuple(self.pool[i] for i in idx)
        while True:
            for k in range(r - 1, -1, -1):
                if idx[k] != k + n - r:
                    break
            else:
                return
            idx[k] += 1
            for j in range(k + 1, r):
                idx[j] = idx[j - 1] + 1
            yield tuple(self.pool[i] for i in idx)

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

    def __iter__(self):
        n, r = self.n, self.r
        if n == 0:
            if r == 0:
                yield ()
            return
        idx = [0] * r
        yield tuple(self.pool[i] for i in idx)
        while True:
            for k in range(r - 1, -1, -1):
                if idx[k] != n - 1:
                    break
            else:
                return
            idx[k:] = [idx[k] + 1] * (r - k)
            yield tuple(self.pool[i] for i in idx)

    def __repr__(self):
        return f"CombinationsWithReplacement({self.pool!r}, {self.r})"


class Map(Lazy):
    """f applied to each element; `inverse` (optional) makes `.index` work."""

    def __init__(self, f, pool, inverse=None):
        self.f, self.pool, self.inverse = f, _pool(pool, once=True), inverse

    @property
    def size(self):
        return _size(self.pool)

    def _get(self, i):
        return self.f(self.pool[i])

    def __iter__(self):
        for v in self.pool:
            yield self.f(v)

    def index(self, x):
        if self.inverse is None:
            raise ValueError("Map without an inverse can't be ranked")
        return _index(self.pool, self.inverse(x))


class Chain(Lazy):
    """itertools.chain over sequences (iterating also works over one-shot iterators, each read once)."""

    def __init__(self, *pools):
        self.pools = [_pool(p, once=True) for p in pools]

    @property
    def sizes(self):
        return [_size(p) for p in self.pools]

    @property
    def size(self):
        return sum(self.sizes)

    def _get(self, i):
        for pool, s in zip(self.pools, self.sizes):
            if i < s:
                return pool[i]
            i -= s

    def __iter__(self):
        for pool in self.pools:
            yield from pool

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


# ---------------------------------------------------------------------------------------------------------------------
# Search: a pattern (one entry per position) selects the matching elements as another lazy sequence, in the parent's
# order, without scanning. `ANY` matches anything; a value matches itself; for Product, a set or list matches any of
# its values. Matches has `.size` (how many), `[i]`, `.index(x)`, iteration, and `.position(i)`: where the i-th match
# sits in the parent sequence.
# ---------------------------------------------------------------------------------------------------------------------

class _Any:
    def __repr__(self):
        return "ANY"


ANY = _Any()


class Without(Lazy):
    """range(n) without the sorted indices `skip`: the free indices left once some are fixed."""

    def __init__(self, n: int, skip):
        self.n, self.skip = n, sorted(skip)
        self.size = n - len(self.skip)

    def _get(self, i):
        return _nth_unused(i, self.skip)

    def index(self, v):
        if not 0 <= v < self.n or (self.skip and self.skip[min(_bisect(self.skip, v), len(self.skip) - 1)] == v):
            raise ValueError(f"{v!r} is not free")
        return v - _bisect(self.skip, v)


class Matches(Lazy):
    """The elements of `parent` matching a pattern. `inner` is a lazy sequence of index tuples (pool indices, one per
    position) in the parent's order; `assemble` turns one into an element, `split` an element back into one."""

    def __init__(self, parent, inner, assemble, split):
        self.parent, self.inner, self.assemble, self.split = parent, inner, assemble, split
        self.size = inner.size

    def _get(self, i):
        return self.assemble(self.inner._get(i))

    def __iter__(self):
        for t in self.inner:
            yield self.assemble(t)

    def index(self, x):
        return self.inner.index(self.split(x))

    def position(self, i):
        """Index in the parent of the i-th match."""
        return self.parent.index(self[i])

    def __repr__(self):
        return f"Matches({self.size} in {self.parent!r})"


def _check_len(pattern, r):
    if len(pattern) != r:
        raise ValueError(f"pattern has {len(pattern)} positions, expected {r}")


def _product_search(self, *pattern):
    _check_len(pattern, len(self.pools))
    choices = []
    for pool, p in zip(self.pools, pattern):
        if p is ANY:
            choices.append(range(_size(pool)))
        elif isinstance(p, (set, frozenset, list)):
            choices.append(sorted({_index(pool, v) for v in p if v in pool}))
        else:
            choices.append([_index(pool, p)] if p in pool else [])
    inner = Product(*choices)
    pools = self.pools
    return Matches(self, inner, lambda t: tuple(pool[j] for pool, j in zip(pools, t)),
                   lambda x: tuple(_index(pool, v) for pool, v in zip(pools, x)))


def _fixed(self, pattern):
    """Pool indices of the fixed positions, or None if a fixed value isn't in the pool."""
    fixed = {}
    for k, p in enumerate(pattern):
        if p is ANY:
            continue
        if isinstance(p, (set, frozenset, list)):
            raise TypeError("sets of values are only supported for Product; search each value separately")
        if p not in self.pool:
            return None
        fixed[k] = _index(self.pool, p)
    return fixed


def _empty_matches(self):
    return Matches(self, Product(range(0)), lambda t: t, lambda x: x)


def _permutations_search(self, *pattern):
    """Fixed positions take their values; the free positions are a permutation of the unused indices, and the
    parent's lexicographic order restricted to the matches is exactly that permutation order."""
    _check_len(pattern, self.r)
    fixed = _fixed(self, pattern)
    if fixed is None or len(set(fixed.values())) != len(fixed):
        return _empty_matches(self)
    free = [k for k in range(self.r) if k not in fixed]
    rest = Without(self.n, fixed.values())
    inner = Permutations(rest, len(free))
    pool, r = self.pool, self.r

    def assemble(t):
        idx = [0] * r
        for k, v in fixed.items():
            idx[k] = v
        for k, v in zip(free, t):
            idx[k] = v
        return tuple(pool[v] for v in idx)

    return Matches(self, inner, assemble, lambda x: tuple(_index(pool, x[k]) for k in free))


def _combinations_search(self, *pattern, replacement=False):
    """Fixed values split the free positions into runs; a run between fixed indices a < b is a combination of the
    indices strictly between them (with replacement: between them inclusive), and the runs combine as a product in
    the parent's order."""
    _check_len(pattern, self.r)
    fixed = _fixed(self, pattern)
    if fixed is None:
        return _empty_matches(self)
    ks = sorted(fixed)
    vals = [fixed[k] for k in ks]
    if any((a > b) if replacement else (a >= b) for a, b in zip(vals, vals[1:])):
        return _empty_matches(self)
    bounds = [(-1, -1)] + [(k, v) for k, v in zip(ks, vals)] + [(self.r, self.n)]
    runs, segs = [], []
    for (k0, v0), (k1, v1) in zip(bounds, bounds[1:]):
        m = k1 - k0 - 1
        if m == 0:
            continue
        lo = (v0 if replacement and k0 >= 0 else v0 + 1)
        hi = (v1 + 1 if replacement and k1 < self.r else v1)
        seg = range(max(lo, 0), hi)
        segs.append(list(range(k0 + 1, k1)))
        runs.append(CombinationsWithReplacement(seg, m) if replacement else Combinations(seg, m))
    inner = Product(*runs)
    pool, r = self.pool, self.r

    def assemble(t):
        idx = [0] * r
        for k, v in fixed.items():
            idx[k] = v
        for positions, run in zip(segs, t):
            for k, v in zip(positions, run):
                idx[k] = v
        return tuple(pool[v] for v in idx)

    def split(x):
        return tuple(tuple(_index(pool, x[k]) for k in positions) for positions in segs)

    return Matches(self, inner, assemble, split)


Product.search = _product_search
Permutations.search = _permutations_search
Combinations.search = _combinations_search
CombinationsWithReplacement.search = lambda self, *p: _combinations_search(self, *p, replacement=True)


def find(seq, pred, start: int = 0):
    """Index of the first element at or after `start` with pred(element) true, or -1. This one does scan: use
    `.search` when the condition is a pattern."""
    for i in range(start, seq.size):
        if pred(seq._get(i)):
            return i
    return -1
