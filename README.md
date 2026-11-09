# lazycomb

itertools' `product`, `permutations`, `combinations` and `combinations_with_replacement`, without evaluating their inputs.

itertools copies every input into a tuple before producing the first element, so nested or large inputs blow up memory: `itertools.product(itertools.product(range(10), repeat=8), repeat=2)` builds all 10^8 inner tuples first. lazycomb keeps references to its inputs and walks them through indices, so nothing is evaluated except the elements you take, at any size and any depth of nesting. Same names, arguments and order as itertools.

Because every object knows its inputs' sizes, you also get, as a by-product, `.size`, element number i (`seq[i]`), the position of an element (`seq.index(x)`), slicing and pattern search, all without generating anything. The cost is speed: iterating is slower than itertools, which is C (see below).

```python
from lazycomb import product, permutations, combinations, ANY

big = product(range(10), repeat=100)     # 10^100 elements, nothing generated
big.size == 10**100                      # True (len() stops at 2^63: use .size)
x = big[123456789]                       # element number 123456789
big.index(x)                             # 123456789
x in big                                 # True

c = combinations(range(10**9), 5)        # a billion items, 5 at a time
c[c.size // 2]                           # the middle combination, instantly

pp = permutations(big, 3)                # sequences nest, at any size
pp[pp.size // 2]

words = product("abcde", repeat=4)
m = words.search(ANY, "c", ANY, {"a", "e"})   # pattern: any, 'c', any, 'a' or 'e'
m.size, m[7], m.position(7)              # 50 matches; the 8th; where it is in `words`

for t in combinations("abcd", 2):        # iterates like itertools
    print(t)
```

## Install

```bash
pip install git+https://github.com/nov3o/lazycomb
```

Optional Rust implementation (needs a Rust toolchain), 3-6x faster for permutations and combinations:

```bash
pip install "git+https://github.com/nov3o/lazycomb#subdirectory=rs"
```

```python
import lazycomb_rs
lazycomb_rs.permutations(range(12), 6)[1000]
```

## What it does

Every sequence has `.size`, `[i]` (negative indices and slices too), `.index(x)`, `in`, and iteration in exactly the order of the matching itertools function.

| | how `[i]` works | cost |
|---|---|---|
| `product` | mixed radix | O(r) |
| `permutations` | factorial number system (Lehmer code) | O(r²) |
| `combinations` | combinatorial number system, binary search per position | O(r log n) binomials |
| `combinations_with_replacement` | maps to combinations of n + r - 1 | same |

Also `Seq`, `Map(f, seq, inverse)`, `Chain`, slices, `.search(pattern)` (matches as another lazy sequence, without scanning) and `find(seq, pred)` (scans).

A sequence only reads its pools through their size and `[i]`, so a pool can be another lazy sequence of any size. itertools copies its inputs into tuples first, so nesting `itertools.product` over large inputs runs out of memory before the first element; lazycomb returns it in microseconds. As itertools does, a one-shot iterator (a generator) is copied into a tuple; ranges, lists and lazy sequences are never copied.

## Speed

Iterating is slower than itertools, which is C. 10^6 elements on one core:

| | itertools | lazycomb (Python) | lazycomb_rs (Rust) |
|---|---|---|---|
| product | 0.063 s | 0.166 s | 0.220 s |
| permutations | 0.045 s | 1.006 s | 0.157 s |
| combinations | 0.065 s | 1.225 s | 0.221 s |
| combinations_with_replacement | 0.012 s | 0.236 s | 0.039 s |

Indexing (`[i]`, µs per call): product of 10^100 elements: Python 89, Rust 7; permutations of 10^6 taken 30 at a time: Python 57, Rust 13.

`python3 bench_backends.py` reruns the comparison and checks both implementations against itertools.

## Tests

```bash
python3 test_lazycomb.py && python3 test_iter.py && python3 test_search.py
```

## License

MIT, see [LICENSE](LICENSE).
