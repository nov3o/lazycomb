//! Rust core of lazycomb: product, permutations, combinations and combinations_with_replacement as indexable
//! sequences in itertools' order. Element counts are big integers; each pool must have at most 2^64 - 1 elements
//! (the Python implementation has no such limit). Pools are read through indexing, never copied, except lists,
//! tuples and strings, whose element references are cached for speed.

use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::{One, ToPrimitive, Zero};
use pyo3::exceptions::{PyIndexError, PyOverflowError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString, PyTuple};

enum PoolKind {
    Range { start: i64, step: i64 },
    Items(Vec<PyObject>),
    Obj,
}

struct Pool {
    kind: PoolKind,
    obj: PyObject,
    n: u64,
}

impl Pool {
    fn new(ob: &Bound<'_, PyAny>) -> PyResult<Pool> {
        let py = ob.py();
        if ob.get_type().name()?.to_cow()? == "range" {
            let (start, step): (i64, i64) = match (ob.getattr("start")?.extract(), ob.getattr("step")?.extract()) {
                (Ok(a), Ok(b)) => (a, b),
                _ => return Err(PyOverflowError::new_err("range bounds too large for the Rust backend")),
            };
            let n: u64 = ob.call_method0("__len__").and_then(|l| l.extract()).map_err(|_| {
                PyOverflowError::new_err("pool too large for the Rust backend")
            })?;
            if n <= (1 << 16) {
                // small ranges: create the ints once, like a tuple
                let items: Vec<PyObject> = ob.try_iter()?.map(|x| x.map(|x| x.unbind())).collect::<PyResult<_>>()?;
                return Ok(Pool { kind: PoolKind::Items(items), obj: ob.clone().unbind(), n });
            }
            return Ok(Pool { kind: PoolKind::Range { start, step }, obj: ob.clone().unbind(), n });
        }
        if ob.is_instance_of::<PyList>() || ob.is_instance_of::<PyTuple>() || ob.is_instance_of::<PyString>() {
            let items: Vec<PyObject> = ob.try_iter()?.map(|x| x.map(|x| x.unbind())).collect::<PyResult<_>>()?;
            let n = items.len() as u64;
            return Ok(Pool { kind: PoolKind::Items(items), obj: ob.clone().unbind(), n });
        }
        let size = match ob.getattr("size") {
            Ok(s) => s,
            Err(_) => ob.call_method0("__len__")?,
        };
        let n: u64 = size.extract().map_err(|_| PyOverflowError::new_err("pool too large for the Rust backend"))?;
        let _ = py;
        Ok(Pool { kind: PoolKind::Obj, obj: ob.clone().unbind(), n })
    }

    #[inline]
    fn get(&self, py: Python<'_>, i: u64) -> PyResult<PyObject> {
        match &self.kind {
            PoolKind::Items(v) => Ok(v[i as usize].clone_ref(py)),
            PoolKind::Range { start, step } => Ok((*start as i128 + *step as i128 * i as i128).into_pyobject(py)?.into_any().unbind()),
            PoolKind::Obj => Ok(self.obj.bind(py).get_item(i)?.unbind()),
        }
    }

    fn index(&self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<u64> {
        if let PoolKind::Range { start, step } = &self.kind {
            let v: i128 = x.extract().map_err(|_| PyValueError::new_err("not in the pool"))?;
            let d = v - *start as i128;
            if d % *step as i128 != 0 || d / (*step as i128) < 0 || (d / *step as i128) as u128 >= self.n as u128 {
                return Err(PyValueError::new_err("not in the pool"));
            }
            return Ok((d / *step as i128) as u64);
        }
        self.obj.bind(py).call_method1("index", (x,))?.extract()
    }
}

fn pools_from(items: &[Bound<'_, PyAny>]) -> PyResult<Vec<Pool>> {
    items.iter().map(Pool::new).collect()
}

fn perm(n: u64, r: u64) -> BigUint {
    if r > n {
        return BigUint::zero();
    }
    let mut p = BigUint::one();
    for k in (n - r + 1)..=n {
        p *= k;
    }
    p
}

fn comb(n: u64, k: u64) -> BigUint {
    if k > n {
        return BigUint::zero();
    }
    let k = k.min(n - k);
    let mut c = BigUint::one();
    for i in 0..k {
        c = c * (n - i) / (i + 1);
    }
    c
}

/// Normalize a Python index (may be negative) against `size`.
fn norm_index(i: BigInt, size: &BigUint) -> PyResult<BigUint> {
    let i = if i.sign() == Sign::Minus { i + BigInt::from(size.clone()) } else { i };
    match i.to_biguint() {
        Some(u) if &u < size => Ok(u),
        _ => Err(PyIndexError::new_err(format!("index out of range for a sequence of size {}", size))),
    }
}

fn to_u64(b: &BigUint) -> u64 {
    b.to_u64().unwrap()
}

fn tuple_of(py: Python<'_>, pools: &[&Pool], idx: &[u64]) -> PyResult<PyObject> {
    let items: Vec<PyObject> = pools.iter().zip(idx).map(|(p, &i)| p.get(py, i)).collect::<PyResult<_>>()?;
    Ok(PyTuple::new(py, items)?.into_any().unbind())
}

fn one_pool_tuple(py: Python<'_>, pool: &Pool, idx: &[u64]) -> PyResult<PyObject> {
    let items: Vec<PyObject> = idx.iter().map(|&i| pool.get(py, i)).collect::<PyResult<_>>()?;
    Ok(PyTuple::new(py, items)?.into_any().unbind())
}

/// Return the next result tuple. Like itertools: when the caller no longer holds the previous tuple (we hold the
/// only reference), it is updated in place from position `from` on instead of allocating a new one.
fn emit(py: Python<'_>, result: &mut Option<PyObject>, r: usize, from: usize,
        get: impl Fn(usize) -> PyResult<PyObject>) -> PyResult<PyObject> {
    if let Some(t) = result.as_ref() {
        if unsafe { pyo3::ffi::Py_REFCNT(t.as_ptr()) } == 1 {
            for k in from..r {
                let v = get(k)?;
                unsafe {
                    let old = pyo3::ffi::PyTuple_GET_ITEM(t.as_ptr(), k as isize);
                    pyo3::ffi::PyTuple_SET_ITEM(t.as_ptr(), k as isize, v.into_ptr());
                    pyo3::ffi::Py_DECREF(old);
                }
            }
            return Ok(t.clone_ref(py));
        }
    }
    let items: Vec<PyObject> = (0..r).map(get).collect::<PyResult<_>>()?;
    let t = PyTuple::new(py, items)?.into_any().unbind();
    *result = Some(t.clone_ref(py));
    Ok(t)
}

fn len_of(size: &BigUint) -> PyResult<usize> {
    size.to_usize().filter(|&n| n <= isize::MAX as usize).ok_or_else(|| PyOverflowError::new_err("size too large for len(): use .size"))
}

// ------------------------------------------------------------------------------------------------- product

#[pyclass(module = "lazycomb_rs")]
struct Product {
    pools: Vec<Pool>,
    size: BigUint,
}

#[pymethods]
impl Product {
    #[new]
    #[pyo3(signature = (*iterables, repeat = 1))]
    fn new(iterables: Vec<Bound<'_, PyAny>>, repeat: usize) -> PyResult<Self> {
        let mut all = Vec::new();
        for _ in 0..repeat {
            all.extend(iterables.iter().cloned());
        }
        let pools = pools_from(&all)?;
        let size = pools.iter().fold(BigUint::one(), |a, p| a * p.n);
        Ok(Product { pools, size })
    }
    #[getter]
    fn size(&self) -> BigUint {
        self.size.clone()
    }
    fn __len__(&self) -> PyResult<usize> {
        len_of(&self.size)
    }
    fn __getitem__(&self, py: Python<'_>, i: BigInt) -> PyResult<PyObject> {
        let mut i = norm_index(i, &self.size)?;
        let mut idx = vec![0u64; self.pools.len()];
        for (k, p) in self.pools.iter().enumerate().rev() {
            let (q, d) = i.div_rem(&BigUint::from(p.n));
            idx[k] = to_u64(&d);
            i = q;
        }
        tuple_of(py, &self.pools.iter().collect::<Vec<_>>(), &idx)
    }
    fn index(&self, py: Python<'_>, x: Vec<Bound<'_, PyAny>>) -> PyResult<BigUint> {
        if x.len() != self.pools.len() {
            return Err(PyValueError::new_err("wrong length"));
        }
        let mut i = BigUint::zero();
        for (p, v) in self.pools.iter().zip(&x) {
            i = i * p.n + p.index(py, v)?;
        }
        Ok(i)
    }
    fn __contains__(&self, py: Python<'_>, x: Vec<Bound<'_, PyAny>>) -> bool {
        self.index(py, x).is_ok()
    }
    fn __iter__(slf: PyRef<'_, Self>) -> ProductIter {
        let empty = slf.pools.iter().any(|p| p.n == 0);
        ProductIter { owner: slf.into(), idx: None, done: empty, result: None }
    }
}

#[pyclass(module = "lazycomb_rs")]
struct ProductIter {
    owner: Py<Product>,
    idx: Option<Vec<u64>>,
    done: bool,
    result: Option<PyObject>,
}

#[pymethods]
impl ProductIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        if self.done {
            return Ok(None);
        }
        let owner = self.owner.borrow(py);
        let pools = &owner.pools;
        let mut from = 0;
        match &mut self.idx {
            None => self.idx = Some(vec![0; pools.len()]),
            Some(idx) => {
                let mut k = pools.len();
                loop {
                    if k == 0 {
                        self.done = true;
                        return Ok(None);
                    }
                    k -= 1;
                    idx[k] += 1;
                    if idx[k] < pools[k].n {
                        break;
                    }
                    idx[k] = 0;
                }
                from = k;
            }
        }
        let idx = self.idx.as_ref().unwrap();
        Ok(Some(emit(py, &mut self.result, pools.len(), from, |k| pools[k].get(py, idx[k]))?))
    }
}

// --------------------------------------------------------------------------------------------- permutations

#[pyclass(module = "lazycomb_rs")]
struct Permutations {
    pool: Pool,
    r: u64,
    size: BigUint,
}

fn nth_unused(mut j: u64, used_sorted: &[u64]) -> u64 {
    for &u in used_sorted {
        if u <= j {
            j += 1;
        } else {
            break;
        }
    }
    j
}

#[pymethods]
impl Permutations {
    #[new]
    #[pyo3(signature = (iterable, r = None))]
    fn new(iterable: Bound<'_, PyAny>, r: Option<u64>) -> PyResult<Self> {
        let pool = Pool::new(&iterable)?;
        let r = r.unwrap_or(pool.n);
        let size = perm(pool.n, r);
        Ok(Permutations { pool, r, size })
    }
    #[getter]
    fn size(&self) -> BigUint {
        self.size.clone()
    }
    fn __len__(&self) -> PyResult<usize> {
        len_of(&self.size)
    }
    fn __getitem__(&self, py: Python<'_>, i: BigInt) -> PyResult<PyObject> {
        let mut i = norm_index(i, &self.size)?;
        let (n, r) = (self.pool.n, self.r);
        let mut used: Vec<u64> = Vec::with_capacity(r as usize);
        let mut out = Vec::with_capacity(r as usize);
        for k in 0..r {
            let block = perm(n - k - 1, r - k - 1);
            let (j, rest) = i.div_rem(&block);
            i = rest;
            let v = nth_unused(to_u64(&j), &used);
            out.push(v);
            let pos = used.partition_point(|&u| u < v);
            used.insert(pos, v);
        }
        one_pool_tuple(py, &self.pool, &out)
    }
    fn index(&self, py: Python<'_>, x: Vec<Bound<'_, PyAny>>) -> PyResult<BigUint> {
        if x.len() as u64 != self.r {
            return Err(PyValueError::new_err("wrong length"));
        }
        let (n, r) = (self.pool.n, self.r);
        let mut i = BigUint::zero();
        let mut used: Vec<u64> = Vec::new();
        for (k, v) in x.iter().enumerate() {
            let v = self.pool.index(py, v)?;
            let pos = used.partition_point(|&u| u < v);
            if pos < used.len() && used[pos] == v {
                return Err(PyValueError::new_err("repeated element"));
            }
            i += perm(n - k as u64 - 1, r - k as u64 - 1) * (v - pos as u64);
            used.insert(pos, v);
        }
        Ok(i)
    }
    fn __contains__(&self, py: Python<'_>, x: Vec<Bound<'_, PyAny>>) -> bool {
        self.index(py, x).is_ok()
    }
    fn __iter__(slf: PyRef<'_, Self>) -> PermIter {
        let (n, r) = (slf.pool.n, slf.r);
        let done = r > n;
        let small = n <= (1 << 22);
        PermIter { owner: slf.into(), idx: None, used: if small { vec![false; n as usize] } else { vec![] }, done, result: None }
    }
}

#[pyclass(module = "lazycomb_rs")]
struct PermIter {
    owner: Py<Permutations>,
    idx: Option<Vec<u64>>,
    used: Vec<bool>, // n <= 4M: a flag per pool index; otherwise empty and idx is searched
    done: bool,
    result: Option<PyObject>,
}

impl PermIter {
    #[inline]
    fn is_used(&self, idx: &[u64], upto: usize, v: u64) -> bool {
        if !self.used.is_empty() {
            self.used[v as usize]
        } else {
            idx[..upto].contains(&v)
        }
    }
}

#[pymethods]
impl PermIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        if self.done {
            return Ok(None);
        }
        let owner = self.owner.borrow(py);
        let (n, r) = (owner.pool.n, owner.r as usize);
        let mut from = 0;
        match self.idx.take() {
            None => {
                let idx: Vec<u64> = (0..r as u64).collect();
                if !self.used.is_empty() {
                    for &v in &idx {
                        self.used[v as usize] = true;
                    }
                }
                self.idx = Some(idx);
            }
            Some(mut idx) => {
                // next r-permutation in lexicographic order
                let mut k = r;
                let mut advanced = false;
                while k > 0 {
                    k -= 1;
                    let old = idx[k];
                    if !self.used.is_empty() {
                        self.used[old as usize] = false;
                    }
                    let mut nxt = old + 1;
                    while nxt < n && self.is_used(&idx, k, nxt) {
                        nxt += 1;
                    }
                    if nxt < n {
                        idx[k] = nxt;
                        if !self.used.is_empty() {
                            self.used[nxt as usize] = true;
                        }
                        let mut v = 0u64;
                        for fill in k + 1..r {
                            while self.is_used(&idx, fill, v) {
                                v += 1;
                            }
                            idx[fill] = v;
                            if !self.used.is_empty() {
                                self.used[v as usize] = true;
                            }
                            v += 1;
                        }
                        advanced = true;
                        from = k;
                        break;
                    }
                }
                if !advanced {
                    self.done = true;
                    return Ok(None);
                }
                self.idx = Some(idx);
            }
        }
        let idx = self.idx.as_ref().unwrap();
        Ok(Some(emit(py, &mut self.result, r, from, |k| owner.pool.get(py, idx[k]))?))
    }
}

// --------------------------------------------------------------------------------------------- combinations

fn comb_unrank(n: u64, r: u64, mut i: BigUint) -> Vec<u64> {
    let mut out = Vec::with_capacity(r as usize);
    let mut start = 0u64;
    for k in 0..r {
        let m = r - k;
        let total = comb(n - start, m);
        let (mut lo, mut hi) = (start, n - m);
        while lo < hi {
            let mid = (lo + hi + 1) / 2;
            if &total - comb(n - mid, m) <= i {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        i -= &total - comb(n - lo, m);
        out.push(lo);
        start = lo + 1;
    }
    out
}

fn comb_rank(n: u64, r: u64, idx: &[u64]) -> BigUint {
    let mut i = BigUint::zero();
    let mut start = 0u64;
    for (k, &v) in idx.iter().enumerate() {
        let m = r - k as u64;
        i += comb(n - start, m) - comb(n - v, m);
        start = v + 1;
    }
    i
}

#[pyclass(module = "lazycomb_rs")]
struct Combinations {
    pool: Pool,
    r: u64,
    replacement: bool,
    size: BigUint,
}

impl Combinations {
    /// combinations with replacement map to plain combinations of n + r - 1 by c_k -> c_k + k
    fn plain_n(&self) -> u64 {
        if self.replacement {
            self.pool.n + self.r - 1
        } else {
            self.pool.n
        }
    }
}

#[pymethods]
impl Combinations {
    #[new]
    #[pyo3(signature = (iterable, r, replacement = false))]
    fn new(iterable: Bound<'_, PyAny>, r: u64, replacement: bool) -> PyResult<Self> {
        let pool = Pool::new(&iterable)?;
        let size = if replacement {
            if pool.n == 0 {
                if r == 0 { BigUint::one() } else { BigUint::zero() }
            } else {
                comb(pool.n + r - 1, r)
            }
        } else {
            comb(pool.n, r)
        };
        Ok(Combinations { pool, r, replacement, size })
    }
    #[getter]
    fn size(&self) -> BigUint {
        self.size.clone()
    }
    fn __len__(&self) -> PyResult<usize> {
        len_of(&self.size)
    }
    fn __getitem__(&self, py: Python<'_>, i: BigInt) -> PyResult<PyObject> {
        let i = norm_index(i, &self.size)?;
        if self.pool.n == 0 {
            return Ok(PyTuple::empty(py).into_any().unbind());
        }
        let mut idx = comb_unrank(self.plain_n(), self.r, i);
        if self.replacement {
            for (k, v) in idx.iter_mut().enumerate() {
                *v -= k as u64;
            }
        }
        one_pool_tuple(py, &self.pool, &idx)
    }
    fn index(&self, py: Python<'_>, x: Vec<Bound<'_, PyAny>>) -> PyResult<BigUint> {
        if x.len() as u64 != self.r {
            return Err(PyValueError::new_err("wrong length"));
        }
        let mut idx: Vec<u64> = x.iter().map(|v| self.pool.index(py, v)).collect::<PyResult<_>>()?;
        let bad = idx.windows(2).any(|w| if self.replacement { w[0] > w[1] } else { w[0] >= w[1] });
        if bad {
            return Err(PyValueError::new_err("not a combination in pool order"));
        }
        if self.pool.n == 0 {
            return Ok(BigUint::zero());
        }
        if self.replacement {
            for (k, v) in idx.iter_mut().enumerate() {
                *v += k as u64;
            }
        }
        Ok(comb_rank(self.plain_n(), self.r, &idx))
    }
    fn __contains__(&self, py: Python<'_>, x: Vec<Bound<'_, PyAny>>) -> bool {
        self.index(py, x).is_ok()
    }
    fn __iter__(slf: PyRef<'_, Self>) -> CombIter {
        let done = slf.size.is_zero();
        CombIter { owner: slf.into(), idx: None, done, result: None }
    }
}

#[pyclass(module = "lazycomb_rs")]
struct CombIter {
    owner: Py<Combinations>,
    idx: Option<Vec<u64>>,
    done: bool,
    result: Option<PyObject>,
}

#[pymethods]
impl CombIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        if self.done {
            return Ok(None);
        }
        let owner = self.owner.borrow(py);
        let (n, r, rep) = (owner.pool.n, owner.r as usize, owner.replacement);
        let mut from = 0;
        match &mut self.idx {
            None => self.idx = Some(if rep { vec![0; r] } else { (0..r as u64).collect() }),
            Some(idx) => {
                let mut k = r;
                let mut found = false;
                while k > 0 {
                    k -= 1;
                    let top = if rep { n - 1 } else { k as u64 + n - r as u64 };
                    if idx[k] != top {
                        found = true;
                        break;
                    }
                }
                if !found {
                    self.done = true;
                    return Ok(None);
                }
                idx[k] += 1;
                for j in k + 1..r {
                    idx[j] = if rep { idx[k] } else { idx[j - 1] + 1 };
                }
                from = k;
            }
        }
        let idx = self.idx.as_ref().unwrap();
        Ok(Some(emit(py, &mut self.result, r, from, |k| owner.pool.get(py, idx[k]))?))
    }
}

#[pyfunction]
fn combinations(iterable: Bound<'_, PyAny>, r: u64) -> PyResult<Combinations> {
    Combinations::new(iterable, r, false)
}

#[pyfunction]
fn combinations_with_replacement(iterable: Bound<'_, PyAny>, r: u64) -> PyResult<Combinations> {
    Combinations::new(iterable, r, true)
}

#[pymodule]
fn lazycomb_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Product>()?;
    m.add_class::<Permutations>()?;
    m.add_class::<Combinations>()?;
    m.add_function(wrap_pyfunction!(combinations, m)?)?;
    m.add_function(wrap_pyfunction!(combinations_with_replacement, m)?)?;
    m.add("product", m.getattr("Product")?)?;
    m.add("permutations", m.getattr("Permutations")?)?;
    Ok(())
}
