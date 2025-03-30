

## Covariance Matrix is Positive-Semi-Definite
(https://statproofbook.github.io/P/covmat-psd.html)

One property of the covariance matrix is that it's PSD. Intuitively, this seems given since variances in nivariate scenarios already have to be nonnegative, and since covariance is a direct multivatriate, symmetric
 extension of it. Let us more properly prove this:

The covariance matrix of a multivariate random variable X is defined as

$\Sigma = E[(X-E[X])(X-E[X])]$

$\Sigma_{ij} = E[(X_{i}-E[X_{i}])(X_{j}-E[X_{j}])]$

The definition of PSD is

$x^TMx \geq 0 \text{ for all } x \in \R^{n}$

Consider $M=\Sigma$ such that

$ x^T\Sigma x = x^T E[(X_{i}-E[X_{i}])(X_{j}-E[X_{j}])] x $

Because x is a scalar (vector) and not a random variable we may insert it inside the expection:

$ = E[x^T (X_{i}-E[X_{i}])(X_{j}-E[X_{j}]) x ] $

Technically above considers $ X_{i}, X_{j} $ as separate variables and $ \Sigma $ is defined like a cross-covariance matrix. When considering multivariate random variable such as a Gaussian with covariance matrix $ \Sigma $ we are interested in the following setup:

$ = E[x^T (X-E[X])(X-E[X]) x ] $

Where $ X $ is a random vector variable. This is only possible because we are viewing this as a *direct extension* of the 1D setup. Then

$ = E[(x^T (X-E[X]))^{2}] $

$ \geq 0 $

Without too much loss of generality we observe that we may rearrange it as a square of the vector product, which is guarenteed to be positive and therefore PSD.

## Einsum

Recently learned about einsum, a flexible operator that can represent a lot of matrix operations. Essentially, einsum conceptually works by writing input and output in terms of axes, which implicitly shows how the output should be calculated.

```
# setup example matrices
a = np.random()

# permute axis
```

## MACs and FLOPs

* MAC: Multiply Accumulate (more commonly used for matrix operations)
* FLOPs: Floating point operation (includes any common but elementary calculations e.g. +,-,*,/ )

## << operator in YAML

The & operator assigns reference key to a mapping, while * calls it, and << does a merge of mappings accordingly. Hence for
```
foo:
  a: b
  <<:
    c: d
  e: f
```

The output comes out as
```
foo:
  a: b
  c: d
  e: f
```

While if we are merging them via references, the merging mapping has less priority such that

```
foo: &foo
  a: b
  c: d
  e: f

fee: &fee
  a: q
  <<: *foo

far: &far
  z: x
  <<: *foo

```

Produces
```
foo:
  a: b
  c: d
  e: f

fee:
  a: q
  c: d
  e: f

far:
  a: b
  c: d
  e: f
  z: x


```

[source](https://stackoverflow.com/questions/41063361/what-is-the-double-left-arrow-syntax-in-yaml-called-and-wheres-it-specifi)