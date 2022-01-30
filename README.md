# NBLAST for large datasets
This is an experimental version of [navis](https://github.com/navis-org/navis)'
NBLAST containing a few tweaks to enable blasting of very large (upwards 100k)
sets of neurons. Notably, we:

1. Use [Zarr](https://zarr.readthedocs.io) to create an on-disk array which is then populated in parallel by the spawned NBLAST processes.
2. Use [Dask](https://docs.dask.org) to wrap the `Zarr` array as on-disk DataFrame.
3. Some (minimal) effort to balance the load for each process.

> :exclamation: **If you need to NBLAST only a few thousand neurons, you are probably better off with the implementation in navis**!

## Usage

```python
>>> import nblast
>>> # `queries`/`targets` are navis.Dotprops
>>> scores = nblast.nblast(queries, targets,
...                        out='./nblast_results'  # where to store Zarr array with results
...                        return_frame=True)  # default is `return_frame=False`
>>> # Convert Dask to in-memory pandas DataFrame
>>> scores = scores.compute()
```

The scores (+ meta data) are stored in  `./nblast_results/`:

```python
>>> import zarr
>>> # Open the array
>>> z = zarr.open_array("./nblast_results/", mode="r")
>>> # IDs of queries/targets are stored as attributes
>>> z.attrs['queries']
[5812980561, 5813056261, 1846312261, 2373845661, ...]
```

## Notes

### Why Zarr for the on-disk array?
Because Zarr has a very easy to use interface for coordinating parallel writes.

### Is this slower than the original navis implementation?
Not as far as I can tell: the overhead from using Zarr/Dask is surprisingly low
as best as I can tell. It does, however, introduce new dependencies and the
returned Dask DataFrame is more difficult to handle than a normal
pandas DataFrame.
