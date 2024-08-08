![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/wetzlingerm/0a3fab03f3da8db62e046b3f913af3fa/raw/coverage.json)


The Python library [continuoussets](https://github.com/wetzlingerm/continuoussets) provides classes for the representation of continuous sets and the evaluation of set operations on them.

- [Installation](https://github.com/wetzlingerm/continuoussets/#installation)
- [Set Representations](https://github.com/wetzlingerm/continuoussets/#set-representations)
  - [Intervals](https://github.com/wetzlingerm/continuoussets/#intervals)
  - [Zonotopes](https://github.com/wetzlingerm/continuoussets/#zonotopes)
  - [V-Polytopes](https://github.com/wetzlingerm/continuoussets/#v-polytopes)
  - [H-Polyhedra](https://github.com/wetzlingerm/continuoussets/#h-polyhedra)
- [Set Operations](https://github.com/wetzlingerm/continuoussets/#set-operations)
- [References](https://github.com/wetzlingerm/continuoussets/#references)

## Installation

The library `continuoussets` requires Python 3.9 or higher.
The only direct dependencies are [numpy](https://numpy.org/), [scipy](https://scipy.org/), and [matplotlib](https://matplotlib.org/).
Installation via PyPI is recommended:
```python
pip install continuoussets
```
Proficient users may choose their own installation path.

## Set Representations

The implemented classes inherit from the interface `IConvexSet`.
They represent continuous sets of n-dimensional vectors.

> [!TIP]
> Many operations, notably including the constructors, also support scalar types, such as `int` and `float`, as well as vectors defined using the type `list`. However, these are internally converted to numpy arrays.

> [!IMPORTANT]
> The usage of keyword arguments for constructors is **mandatory**.

### Intervals

An interval `I` is defined using a lower bound `lb` and an upper bound `ub`:

> I = { x | lb <= x <= ub }

where `lb` and `ub` are vectors of equal length, and the inequality holds elementwise.

The `Interval` class allows to instantiate such objects:
```python
I = Interval(lb = numpy.array([-2., 0.]), ub = numpy.array([2., 1.]))
```


### Zonotopes

A zonotope `Z` is defined using a center `c` and a generator matrix `G`:

> Z = { c + sum_i G_i a_i | -1 <= a_i <= 1 }

where `G_i` are the columns of `G`.
In contrast to intervals, zonotopes can represent dependencies between different dimensions.

The `Zonotope` class allows to instantiate such objects:
```python
Z = Zonotope(c = numpy.array([1., 0.]), G = numpy.array([[1. -1.], [0., 1.], [2., 1.]]))
```
Note that the ith generator is stored as `G[i]` in the matrix `G`.


### V-Polytopes

A polytope in vertex representation `VP` is defined via the convex hull of a set of points `v`:

> VP = { sum_i v_i beta_i | sum_i beta_i = 1, beta_i >= 0 }

The `VPolytope` class allows to instantiate such objects:
```python
VP = VPolytope(V = numpy.array([[1., 0.], [-1., 1.], [-2., -1.]]))
```
Note that the ith point is stored as `V[i]` in the matrix `V`.


### H-Polyhedra

A polyhedron in vertex representation `HP` is defined via the constraint matrix `A` and the constraint offset `b`:

> HP = { x | A x <= b }

The `HPolyhedron` class allows to instantiate such objects:
```python
HP = HPolyhedron(A = numpy.array([[1., 0.], [-1., 1.], [-2., -1.]]), b = numpy.array([1., 2., 1.]))
```
Note that the ith constraint vector is stored as `A[i]` in the matrix `A`.
Furthermore, h-polyhedra may be unbounded or empty.


## Set Operations
  
Many standard set operations are implemented:

- `boundary_point`: Computation of boundary points
- `cartesian_product`: Cartesian product
- `compact`: Reduction to the minimal set representation size without changing the set
- `convex_hull`: Convex hull
- `matmul`: Linear map
- `minkowski_sum`: Minkowski sum
- `minkowski_difference`: Minkowski/Pontryagin difference
- `project`: Projection of the set onto an axis-aligned subspace
- `reduce`: Reduction of the set representation size, possibly incurring enlargement
- `support_function`: Support function evaluation
- `vertices`: Vertex enumeration
- `volume`: Volume computation

> [!IMPORTANT]
> All operations return **new class instances**.

> [!NOTE]
> Operands for binary operations can also be vectors, represented by 1D numpy arrays.

> [!TIP]
> Many operations support **various evaluation modes** via the keyword argument `mode`. These detail whether an exact solution (`mode = 'exact'`, by default), an outer approximation (`mode = 'outer'`) or an inner approximation (`mode = 'inner'`) should be computed. Not all modes are supported for each operation, some operations cannot be evaluated exactly, and runtime may differ strongly between modes.

Furthermore, the following checks are supported:

- `bounded`: Boundedness of a set
- `contains`: Containment of one set or vector in another set
- `degenerate`: Degeneracy of a set
- `empty`: Emptiness of a set
- `intersects`: Intersection between a set and another set or vector
- `__eq__`: Equality of a set and another set or vector
- `represents`: Equivalent representation of a set by another set representation

The `Interval` class additionally supports **interval arithmetic** for range bounding purposes. This includes

- basic operations: addition (`__add__`), subtraction (`__sub__`), multiplication (`__mul__`), division (`__truediv__`), exponentiation (`__pow__`)
- trigonometric functions: `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`
- other standard functions: `sqrt`, `log`, `log10`


## References

This library is heavily inspired by the MATLAB toolbox [CORA](https://cora.in.tum.de).
However, this implementation is based on original sources, e.g.,

- G. Alefeld and G. Mayer. “Interval analysis: Theory and applications”.
  In: Computational and Applied Mathematics 121.1-2 (2000), pp. 421–464. doi: 10.1016/S0377-0427(00)00342-3
- M. Althoff. “Reachability analysis and its application to the safety assessment of autonomous cars”.
  Dissertation. Technische Universität München, 2010.
- M. Wetzlinger, V. Kotsev, A. Kulmburg, and M. Althoff. "Implementation of polyhedral operations in CORA 2024".
  In: Proceedings of the 11th Workshop on Applied Verification of Continuous and Hybrid Systems.