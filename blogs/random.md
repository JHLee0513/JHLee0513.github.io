
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