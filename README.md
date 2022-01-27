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
>>> scores = nblast.nblast(queries, targets, out='./nblast_results')
>>> # Write partitions to individual parquet files
>>> scores.to_parquet('./nblast_parquet')
>>> # Convert to in-memory pandas DataFrame
>>> scores = scores.compute()
```

Note that the Zarr data (scores + meta data) `./nblast_results/` is not
cleaned up automatically. 

## Notes

### Why Zarr for the on-disk array?
Because Zarr has a very easy to use interface for coordinating parallel writes.

### Is this slower than the original navis implementation?
Not as far as I can tell: the overhead from using Zarr/Dask is surprisingly low
as best as I can tell. It does, however, introduce new dependencies and the
returned Dask DataFrame is more difficult to handle than a normal
pandas DataFrame.
