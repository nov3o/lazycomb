"""Python vs Rust lazycomb vs itertools: same results, then speed of iteration, indexing and ranking."""
import itertools, random, time
import lazycomb as py
import lazycomb_rs as rs

random.seed(0)
KINDS = ["product", "permutations", "combinations", "combinations_with_replacement"]


def make(mod, kind, pool, r):
    f = getattr(mod, kind)
    return f(pool, repeat=r) if kind == "product" else f(pool, r)


# 1. identical to itertools, element by element, plus [i] and index round trips
n_cases = 0
for kind in KINDS:
    for pool in ["", "a", "abc", list(range(5)), range(3, 15, 3)]:
        for r in range(0, 5):
            want = list(make(itertools, kind, pool, r))
            for mod in (py, rs):
                s = make(mod, kind, pool, r)
                assert s.size == len(want), (mod.__name__, kind, pool, r)
                assert list(s) == want, (mod.__name__, kind, pool, r)
                for i, x in enumerate(want):
                    assert s[i] == x and s.index(x) == i and x in s
                if want:
                    assert s[-1] == want[-1]
            n_cases += 1
print(f"1. Python and Rust identical to itertools: {n_cases} cases (order, [i], [-1], index, in)")

# 2. huge sequences agree between Python and Rust
for kind, pool, r in [("product", range(10), 100), ("permutations", range(10**6), 30),
                      ("combinations", range(10**9), 20), ("combinations_with_replacement", range(10**4), 50)]:
    a, b = make(py, kind, pool, r), make(rs, kind, pool, r)
    assert a.size == b.size
    for _ in range(200):
        i = random.randrange(a.size)
        x = a[i]
        assert b[i] == x and b.index(x) == i == a.index(x)
print("2. huge sequences: Python and Rust give the same elements and ranks (200 random indices each)")


def timeit(f, reps=1):
    t = time.perf_counter()
    for _ in range(reps): f()
    return (time.perf_counter() - t) / reps


# 3. iteration speed: 10^6 elements
print("\n3. iterate 10^6 elements (s)             itertools   Python    Rust   Rust vs Python  Rust vs itertools")
for kind, pool, r in [("product", range(10), 6), ("permutations", range(12), 6), ("combinations", range(40), 6),
                      ("combinations_with_replacement", range(20), 6), ("product", "abcdefghij", 6)]:
    N = 10**6
    t = {name: timeit(lambda: sum(1 for _ in itertools.islice(make(m, kind, pool, r), N)))
         for name, m in [("it", itertools), ("py", py), ("rs", rs)]}
    print(f"   {kind:<30} {str(pool)[:12]:<12} {t['it']:8.3f} {t['py']:8.3f} {t['rs']:7.3f} {t['py'] / t['rs']:10.1f}x {t['rs'] / t['it']:12.1f}x slower")

# 4. indexing / ranking speed
print("\n4. random access (us per call)                     Python [i]  Rust [i]   Python index  Rust index")
for kind, pool, r in [("product", range(10), 100), ("permutations", range(10**6), 30),
                      ("combinations", range(10**9), 20), ("combinations_with_replacement", range(10**4), 50)]:
    a, b = make(py, kind, pool, r), make(rs, kind, pool, r)
    idx = [random.randrange(a.size) for _ in range(2000)]
    xs = [a[i] for i in idx]
    ta = timeit(lambda: [a[i] for i in idx]) / len(idx) * 1e6
    tb = timeit(lambda: [b[i] for i in idx]) / len(idx) * 1e6
    ra = timeit(lambda: [a.index(x) for x in xs]) / len(xs) * 1e6
    rb = timeit(lambda: [b.index(x) for x in xs]) / len(xs) * 1e6
    print(f"   {kind:<30} r={r:<4}            {ta:9.1f} {tb:9.1f} {ra:13.1f} {rb:11.1f}")
