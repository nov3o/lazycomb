import itertools, random, time
from lazycomb import Chain, Combinations, CombinationsWithReplacement, Map, Permutations, Product, Seq

random.seed(0)
# 1. identical to itertools, element by element, on every small case
checked = 0
for n in range(0, 7):
    pool = list("abcdefg"[:n])
    for r in range(0, n + 2):
        for lazy, ref in [(Permutations(pool, r), itertools.permutations(pool, r)),
                          (Combinations(pool, r), itertools.combinations(pool, r)),
                          (CombinationsWithReplacement(pool, r), itertools.combinations_with_replacement(pool, r))]:
            ref = list(ref)
            assert lazy.size == len(ref), (lazy, lazy.size, len(ref))
            assert [lazy[i] for i in range(lazy.size)] == ref == list(lazy), lazy
            assert all(lazy.index(t) == i for i, t in enumerate(ref)), lazy
            checked += len(ref)
    for rep in range(0, 4):
        ref = list(itertools.product(pool, range(3), repeat=rep))
        lazy = Product(pool, range(3), repeat=rep)
        assert list(lazy) == ref == [lazy[i] for i in range(lazy.size)] and all(lazy.index(t) == i for i, t in enumerate(ref))
        checked += len(ref)
print(f"1. same as itertools: {checked} elements, every index and every rank")

# 2. nested: lazy pools inside lazy sequences, against itertools on the expanded pools
inner = Permutations(range(4), 2)
lazy = Combinations(inner, 3)
ref = list(itertools.combinations(list(itertools.permutations(range(4), 2)), 3))
assert list(lazy) == ref and all(lazy.index(t) == i for i, t in enumerate(ref))
lazy = Product(Combinations(range(5), 2), CombinationsWithReplacement("xy", 2))
ref = list(itertools.product(itertools.combinations(range(5), 2), itertools.combinations_with_replacement("xy", 2)))
assert list(lazy) == ref
print("2. nested sequences match itertools")

# 3. the 'crash' case: product(product(product(range(10), repeat=100), repeat=100), repeat=100)
t = time.perf_counter()
big = Product(Product(Product(range(10), repeat=100), repeat=100), repeat=100)
assert big.size == 10 ** 1000000
for i in [0, 1, 123, 6969, 10 ** 999999 + 7, big.size - 1]:
    assert big.index(big[i]) == i
print(f"3. product^3 of range(10)^100: size 10^1000000, 6 round-trips in {time.perf_counter() - t:.2f}s")

# 4. Ilia's example: Z_2^4, Z_3^3, Z_4^2 -> permutations of each set -> choose 3 -> cartesian product
t = time.perf_counter()
parts = [Combinations(Permutations(Product(range(k), repeat=e)), 3) for k, e in [(2, 4), (3, 3), (4, 2)]]
space = Product(*parts)
print(f"4. size = {space.size:.3e} (~10^{len(str(space.size)) - 1})")
for i in [123, 6969, space.size // 3]:
    x = space[i]
    assert space.index(x) == i
    print(f"   {i if i < 10**6 else '%.3e' % i} -> first permutation of first pick: {x[0][0][0][:4]}...  round-trip ok")
for _ in range(200):
    i = random.randrange(space.size)
    assert space.index(space[i]) == i
print(f"   200 random round-trips ok, {time.perf_counter() - t:.2f}s")

# 5. huge pools, small r
p = Permutations(range(10 ** 30), 5)
c = Combinations(range(10 ** 30), 5)
for s in (p, c, CombinationsWithReplacement(range(10 ** 30), 5)):
    for _ in range(100):
        i = random.randrange(s.size)
        assert s.index(s[i]) == i
print("5. pools of 10^30 with r=5: 300 random round-trips ok")

# 6. slices, negative indices, membership, Map, Chain
s = Product(range(10), repeat=3)
assert s[-1] == (9, 9, 9) and list(s[10:20:3]) == list(itertools.product(range(10), repeat=3))[10:20:3]
assert (1, 2, 3) in s and (1, 2, 10) not in s
m = Map(lambda t: "".join(map(str, t)), s, inverse=lambda w: tuple(map(int, w)))
assert m[123] == "123" and m.index("987") == 987
ch = Chain(range(3), "ab", Combinations(range(3), 2))
assert list(ch) == [0, 1, 2, "a", "b", (0, 1), (0, 2), (1, 2)] and ch.index((0, 2)) == 6
print("6. slices, negatives, in, Map, Chain ok")

# 7. speed
s = Product(range(10), repeat=12)
t = time.perf_counter(); n = sum(1 for _ in zip(range(10 ** 6), s)); it = time.perf_counter() - t
t = time.perf_counter(); n = sum(1 for _ in zip(range(10 ** 6), itertools.product(range(10), repeat=12))); ref = time.perf_counter() - t
pc = Combinations(range(10 ** 6), 4)
t = time.perf_counter(); [pc[random.randrange(pc.size)] for _ in range(10 ** 4)]; un = (time.perf_counter() - t) / 1e4
print(f"7. iterate 1M of product: {it:.2f}s (itertools {ref:.2f}s); unrank a 4-combination of 10^6: {un * 1e6:.0f} us")
