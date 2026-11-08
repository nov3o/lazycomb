import itertools, random, time
from lazycomb import ANY, Product, Permutations, Combinations, CombinationsWithReplacement, find

random.seed(0)


def brute(it, pattern):
    def ok(x):
        return all(p is ANY or (v in p if isinstance(p, (set, list)) else v == p) for v, p in zip(x, pattern))
    return [(i, x) for i, x in enumerate(it) if ok(x)]


def check(seq, it, pattern):
    m = seq.search(*pattern)
    want = brute(it, pattern)
    assert m.size == len(want), (seq, pattern, m.size, len(want))
    assert list(m) == [x for _, x in want], (seq, pattern)
    for j, (i, x) in enumerate(want):
        assert m[j] == x and m.index(x) == j and m.position(j) == i
    return len(want)


n_checks = 0
pool = "abcde"
for r in range(0, 5):
    for trial in range(40):
        pat = [random.choice([ANY, ANY, *pool]) for _ in range(r)]
        n_checks += 1
        check(Permutations(pool, r), itertools.permutations(pool, r), pat)
        check(Combinations(pool, r), itertools.combinations(pool, r), pat)
        check(CombinationsWithReplacement(pool, r), itertools.combinations_with_replacement(pool, r), pat)
        ppat = [random.choice([ANY, random.choice(pool), set(random.sample(pool, 2)), "z"]) for _ in range(r)]
        check(Product(pool, repeat=r), itertools.product(pool, repeat=r), ppat)
print(f"search matches brute force: {n_checks} patterns x 4 kinds, sizes up to 5^4")

# big: counting and indexing matches without scanning
t = time.perf_counter()
big = Product(range(10), repeat=1000)
m = big.search(*([7] + [ANY] * 998 + [{1, 2, 3}]))
assert m.size == 3 * 10 ** 998
x = m[10 ** 500]
assert x[0] == 7 and x[-1] in (1, 2, 3) and big[m.position(10 ** 500)] == x
p = Permutations(range(10 ** 6), 50)
pm = p.search(*([ANY] * 10 + [123456] + [ANY] * 38 + [42]))
y = pm[pm.size // 3]
assert y[10] == 123456 and y[49] == 42 and len(set(y)) == 50 and p[pm.position(pm.size // 3)] == y
c = Combinations(range(10 ** 9), 20)
cm = c.search(*([ANY] * 5 + [500_000_000] + [ANY] * 14))
z = cm[cm.size - 1]
assert z[5] == 500_000_000 and list(z) == sorted(z) and c[cm.position(cm.size - 1)] == z
print(f"big patterns (10^1000 products, 10^6-pool permutations, 10^9-pool combinations): {time.perf_counter() - t:.2f} s")

assert find(Permutations(range(8)), lambda x: x[:3] == (7, 6, 5)) == Permutations(range(8)).index((7, 6, 5, 0, 1, 2, 3, 4))
print("find ok")
print("ALL OK")
