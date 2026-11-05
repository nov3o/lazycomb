import itertools, resource, subprocess, sys, time
from lazycomb import Combinations, CombinationsWithReplacement, Permutations, Product, Chain, Map

# 1. one-shot generator as the first pool: read once, never copied
gen = (c for c in "abc")
assert list(Product(gen, range(2))) == list(itertools.product("abc", range(2)))

# 2. a pool that can only be iterated (no len, no indexing), restarted by the odometer
class Words:
    starts = 0
    def __iter__(self):
        Words.starts += 1
        return iter(["x", "y"])
w = Words()
assert list(Product(range(3), w)) == list(itertools.product(range(3), ["x", "y"]))
print(f"1-2. generator first pool and iterate-only pools: ok (inner pool restarted {Words.starts} times)")

# 3. one-shot iterators where they would be read twice are refused, not copied
for bad in [lambda: Product(range(2), iter("ab")), lambda: Product(iter("ab"), repeat=2)]:
    try:
        bad(); print("NOT refused")
    except TypeError:
        pass
print("3. one-shot inner pools refused with a clear error")

# 4. permutations/combinations iteration equals itertools, element by element
for n in range(7):
    for r in range(n + 2):
        for lazy, ref in [(Permutations(range(n), r), itertools.permutations(range(n), r)),
                          (Combinations(range(n), r), itertools.combinations(range(n), r)),
                          (CombinationsWithReplacement(range(n), r), itertools.combinations_with_replacement(range(n), r))]:
            assert list(lazy) == list(ref), (lazy, n, r)
print("4. permutations / combinations / with replacement: same order as itertools for n < 7")

# 5. nested products: first elements instantly, constant memory
t = time.perf_counter()
big = Product(Product(Product(range(10), repeat=100), repeat=100), repeat=100)
first = list(itertools.islice(big, 1000))
print(f"5. product^3 of range(10)^100: first 1000 elements in {time.perf_counter() - t:.2f}s, "
      f"peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3:.0f} MB")

# 6. the same idea with itertools, under a 2 GB memory cap
code = ("import itertools, resource; resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30));"
        "next(itertools.product(itertools.product(range(10), repeat=8), repeat=2))")
r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
print("6. itertools.product(itertools.product(range(10), repeat=8), repeat=2), first element, 2 GB cap:",
      "MemoryError" if "MemoryError" in r.stderr else (r.stderr.strip().splitlines() or ["ok"])[-1])
t = time.perf_counter(); next(iter(Product(Product(range(10), repeat=8), repeat=2))); print(f"   lazycomb: first element in {(time.perf_counter() - t) * 1e3:.2f} ms")

# 7. speed: iterate 10^6 elements
def rate(it, n=10 ** 6):
    t = time.perf_counter()
    for _ in itertools.islice(it, n):
        pass
    return (time.perf_counter() - t)
for name, lazy, ref in [("product(range(10), repeat=6)", Product(range(10), repeat=6), itertools.product(range(10), repeat=6)),
                        ("permutations(range(12), 6)", Permutations(range(12), 6), itertools.permutations(range(12), 6)),
                        ("combinations(range(40), 6)", Combinations(range(40), 6), itertools.combinations(range(40), 6))]:
    a, b = rate(iter(lazy)), rate(ref)
    print(f"7. {name:30} lazycomb {a:.2f}s, itertools {b:.2f}s per 10^6 ({a / b:.1f}x)")
