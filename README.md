# NBLAST for large datasets
Experimental version of [navis](https://github.com/navis-org/navis)'
NBLAST containing a few tweaks to enable blasting of very large (upwards 100k)
sets of neurons.

> :exclamation: **If you need to NBLAST only a few thousand neurons, you are probably better off with the implementation in navis**!

At this point we offer two variants:

## 1. NBLAST to disk
This implementation immediately writes scores to disk. Notably, we:

1. Use [Zarr](https://zarr.readthedocs.io) to create an on-disk array which is then populated in parallel by the spawned NBLAST processes.
2. Use [Dask](https://docs.dask.org) to wrap the `Zarr` array as on-disk DataFrame.
3. Some (minimal) effort to balance the load for each process.

```python
>>> from nblast import nblast_disk
>>> # `queries`/`targets` are navis.Dotprops
>>> scores = .nblast_disk(queries, targets,
...                       out='./nblast_results'  # where to store Zarr array with results
...                       return_frame=True)      # default is `return_frame=False`
>>> # Convert Dask to in-memory pandas DataFrame
>>> scores = scores.compute()
```

The scores are stored as Zarr array in `./nblast_results/scores`. To re-open:

```python
>>> import zarr
>>> # Open the array
>>> z = zarr.open_array("./nblast_results/scores", mode="r")
>>> # IDs of queries/targets are stored as attributes
>>> z.attrs['queries']
[5812980561, 5813056261, 1846312261, 2373845661, ...]
```

### Why Zarr for the on-disk array?
Because Zarr has a very easy to use interface for coordinating parallel writes.

### Is this slower than the original navis implementation?
Not as far as I can tell: the overhead from using Zarr/Dask appears surprisingly
low. It does, however, introduce new dependencies and the returned Dask
DataFrame is more difficult to handle than run-of-the-mill pandas DataFrames.


## 2. NBLAST to sparse matrix
This implementation drops scores below a given threshold and stores results in
a sparse matrix. This is useful if you don't care about low (say less than 0)
scores.

```python
>>> from nblast import nblast_sparse
>>> # `queries`/`targets` are navis.Dotprops
>>> scores = nblast_sparse(queries, targets, threshold=0)
>>> scores.dtypes
603270961     Sparse[float32, nan]
2312333561    Sparse[float32, nan]
1934410061    Sparse[float32, nan]
...
>>> scores.sparse.density
0.05122
```

`scores` is a pandas DataFrame where each column is a sparse array. It will,
in many ways, handle like a normal DataFrame. See the
[docs](https://pandas.pydata.org/pandas-docs/stable/user_guide/sparse.html) for
details.
