# NBLAST for large datasets
This is an experimental version of [navis](https://github.com/navis-org/navis)'
NBLAST containing a few tweaks to enable blasting of very large (upwards 100k)
sets of neurons:

1. Use [Dask](https://docs.dask.org) to avoid running out of memory
2. Some (minimal) effort to balance the load for each process
3. Avoid fancy indexing/writing of arrays

> :exclamation: **If you need to NBLAST only a few thousand neurons, you are probably better off with the implementation in navis**!

## Usage

```python
>>> import nblast
>>> # `queries`/`targets` are navis.Dotprops
>>> scores = nblast.nblast(queries, targets)
>>> # Write partitions to individual parquet files
>>> scores.to_parquet('')
>>> # Convert to in-memory pandas DataFrame
>>> scores = scores.compute()
```

## Notes

The overhead from using Dask is surprisingly low as best as I can tell but it
does introduce a new dependency and the returned Dask DataFrame is more
difficult to handle than a normal pandas DataFrame. 
