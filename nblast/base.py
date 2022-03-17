import navis
import numbers
import os
import random
import uuid
import tempfile
import zarr
import logging

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import scipy.sparse as sp

from concurrent.futures import ProcessPoolExecutor
from typing import Union, Optional, List
from typing_extensions import Literal
from pathlib import Path

from time import time
from pykdtree.kdtree import KDTree

utils = navis.utils
config = navis.config
smat_path = navis.nbl.nblast_funcs.smat_path
Dotprops = navis.Dotprops
NeuronList = navis.NeuronList


class LargeBlaster(navis.nbl.nblast_funcs.NBlaster):
    """Base class for large NBLASTs."""

    def __init__(self, offset=(0, 0), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neurons_ds = []
        self.self_hits_ds = []
        self.offset = offset

    def append(self, dotprops, downsampled=None):
        """Append dotprops.

        Parameters
        ----------
        dotprops :      list | NeuronList
                        Dotprops to add. Must be an iterable.
        downsampled :   list | NeuronList, optional
                        Downsampled versions of `dotprops`.

        Returns
        -------
        None

        """
        if downsampled:
            if len(downsampled) != len(dotprops):
                raise ValueError(f'Got {len(downsampled)} downsampled for '
                                 f'{len(dotprops)} full dotprops.')

            for dp in downsampled:
                self.neurons_ds.append(dp)
                self.self_hits_ds.append(self.calc_self_hit(dp))

        for dp in dotprops:
            self.neurons.append(dp)
            # Calculate score for self hit
            self.self_hits.append(self.calc_self_hit(dp))

    def single_query_target_ds(self, q_idx, t_idx, scores='forward'):
        """Query single downsampled target against single downsampled target."""
        # Take a short-cut if this is a self-self comparison
        if q_idx == t_idx:
            if self.normalized:
                return 1
            return self.self_hits_ds[q_idx]

        # Run nearest-neighbor search for query against target
        data = self.neurons_ds[q_idx].dist_dots(self.neurons_ds[t_idx],
                                                alpha=self.use_alpha,
                                                distance_upper_bound=self.distance_upper_bound)
        if self.use_alpha:
            dists, dots, alpha = data
            dots *= np.sqrt(alpha)
        else:
            dists, dots = data

        scr = self.score_fn(dists, dots).sum()

        # Normalize against best hit
        if self.normalized:
            scr /= self.self_hits_ds[q_idx]

        # For the mean score we also have to produce the reverse score
        if scores in ('mean', 'min', 'max'):
            reverse = self.single_query_target_ds(t_idx, q_idx, scores='forward')
            if scores == 'mean':
                scr = (scr + reverse) / 2
            elif scores == 'min':
                scr = min(scr, reverse)
            elif scores == 'max':
                scr = max(scr, reverse)

        return scr


class DiskBlaster(LargeBlaster):
    """NBLAST to disk."""
    def __init__(self, out, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out = out

    def multi_query_target(self, q_idx, t_idx,
                          chunksize=10,
                          scores='forward',
                          smart=False):
        """BLAST multiple queries against multiple targets.

        Parameters
        ----------
        q_idx,t_idx :       iterable
                            Iterable of query/target neuron indices to BLAST.
        chunksize :         int
                            Data will be written to the array in chunks of this
                            size.
        scores :            "forward" | "mean" | "min" | "max"
                            Which scores to return.
        smart :             bool
                            If True will run a pre-NBLAST on the downsampled
                            dotprops and then only full NBLAST the top 10% for
                            each query.

        Returns
        -------
        None

        """
        # Top 10% of neurons to take to full NBLAST
        top_N = len(t_idx) // 10

        if utils.is_jupyter() and config.tqdm == config.tqdm_notebook:
            # Jupyter does not like the progress bar position for some reason
            position = None

            # For some reason we have to do this if we are in a Jupyter environment
            # and are using multi-processing because otherwise the progress bars
            # won't show. See this issue:
            # https://github.com/tqdm/tqdm/issues/485#issuecomment-473338308
            print(' ', end='', flush=True)
        else:
            position = getattr(self, 'pbar_position', 0)

        # Access array
        lockfile = Path(f'{self.out}.sync')
        synchronizer = zarr.ProcessSynchronizer(lockfile)
        z = zarr.open_array(self.out, mode='r+', synchronizer=synchronizer)

        # Create empty chunk
        chunk = np.zeros((chunksize, len(t_idx)))
        row_offset, col_offset = self.offset  # shorthands
        for i, q in enumerate(config.tqdm(q_idx,
                                          desc=self.desc,
                                          leave=False,
                                          position=position,
                                          disable=not self.progress)):
            for k, t in enumerate(t_idx):
                if not smart:
                    chunk[i % chunksize, k] = self.single_query_target(q, t, scores=scores)
                else:
                    chunk[i % chunksize, k] = self.single_query_target_ds(q, t, scores=scores)

            if smart:
                # Run full NBLAST for the top 10%
                for k in np.argsort(chunk[i % chunksize])[-top_N:]:
                    chunk[i % chunksize, k] = self.single_query_target(q, t_idx[k], scores=scores)

            # Whenever a chunk is full, we need to write to the Zarr array
            if i > 0 and not (i + 1) % chunksize:
                z[row_offset: row_offset + chunksize,
                  col_offset: col_offset + chunk.shape[1]] = chunk

                row_offset += chunksize

        # If the last chunk has not been filled, we need to write the partial
        # chunk to disk
        if (i + 1) % chunksize:
            z[row_offset: row_offset + (i + 1) % chunksize,
              col_offset: col_offset + chunk.shape[1]] = chunk[:(i + 1) % chunksize]

        return


class SparseBlaster(LargeBlaster):
    """NBLAST to sparse matrix (coo/csc format)."""
    def __init__(self, threshold, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    def multi_query_target(self, q_idx, t_idx,
                          chunksize=10,
                          scores='forward',
                          smart=False):
        """BLAST multiple queries against multiple targets.

        Parameters
        ----------
        q_idx,t_idx :       iterable
                            Iterable of query/target neuron indices to BLAST.
        chunksize :         int
                            Data will be written to the array in chunks of this
                            size.
        scores :            "forward" | "mean" | "min" | "max"
                            Which scores to return.
        smart :             bool
                            If True will run a pre-NBLAST on the downsampled
                            dotprops and then only full NBLAST the top 10% for
                            each query.

        Returns
        -------
        None

        """
        # Top 10% of neurons to take to full NBLAST
        top_N = len(t_idx) // 10

        if utils.is_jupyter() and config.tqdm == config.tqdm_notebook:
            # Jupyter does not like the progress bar position for some reason
            position = None

            # For some reason we have to do this if we are in a Jupyter environment
            # and are using multi-processing because otherwise the progress bars
            # won't show. See this issue:
            # https://github.com/tqdm/tqdm/issues/485#issuecomment-473338308
            print(' ', end='', flush=True)
        else:
            position = getattr(self, 'pbar_position', 0)

        # Create empty chunk
        all_scores = []
        all_coo = []
        row_offset, col_offset = self.offset  # shorthands
        for i, q in enumerate(config.tqdm(q_idx,
                                          desc=self.desc,
                                          leave=False,
                                          position=position,
                                          disable=not self.progress)):
            this_scores = []
            this_coo = []
            this_t = []
            for k, t in enumerate(t_idx):
                if not smart:
                    s = self.single_query_target(q, t, scores=scores)
                else:
                    s = self.single_query_target_ds(q, t, scores=scores)

                if s >= self.threshold:
                    this_scores.append(s)
                    this_coo.append([i, k])
                    this_t.append(t)

            if smart and len(this_scores):
                # Run full NBLAST for the top 10%
                for k in np.argsort(this_scores)[-top_N:]:
                    this_scores[k] = self.single_query_target(q, this_t[k], scores=scores)
            all_scores += this_scores
            all_coo += this_coo

        all_scores = np.array(all_scores, dtype=self.dtype)
        all_coo = np.array(all_coo, dtype=np.int32)

        if len(all_coo):
            all_coo += self.offset
        else:
            all_coo = np.zeros((0, 2), dtype=np.int32)

        return all_scores, all_coo


def nblast_disk(query: Union[Dotprops, NeuronList],
                target: Optional[str] = None,
                out: str = None,
                scores: Union[Literal['forward'],
                              Literal['mean'],
                              Literal['min'],
                              Literal['max']] = 'forward',
                normalized: bool = True,
                smart: bool = False,
                use_alpha: bool = False,
                smat: Optional[Union[str, pd.DataFrame]] = 'auto',
                limit_dist: Optional[Union[Literal['auto'], int, float]] = None,
                n_cores: int = os.cpu_count() // 2,
                dtype: Union[str, np.dtype] = 'float32',
                return_frame: bool = False,
                progress: bool = True) -> pd.DataFrame:
    """NBLAST query against target neurons and write results to disk.

    Parameters
    ----------
    query :         Dotprops | NeuronList
                    Query neuron(s) to NBLAST against the targets. Neurons
                    should be in microns as NBLAST is optimized for that and
                    have similar sampling resolutions.
    target :        Dotprops | NeuronList, optional
                    Target neuron(s) to NBLAST against. Neurons should be in
                    microns as NBLAST is optimized for that and have
                    similar sampling resolutions. If not provided, will NBLAST
                    queries against themselves.
    out :           Path-like
                    A folder where to store the Zarr array (and some supporting
                    data):
                      - `{out}/scores` contains the Zarr array (index and columns
                        are stored as `.attrs['queries']` and `.attrs['targets']`,
                        respectively)
                      - `{out}/scores.lock` contains lockfiles
                      - `{out}/scores.sync` contains
                    Importantly, these are not cleaned up automatically -
                    you have to remove the files manually when you're done with
                    the results! Also note that any existing Zarr array in
                    `{out}/scores` are overwritten.
    scores :        'forward' | 'mean' | 'min' | 'max'
                    Determines the final scores:
                      - 'forward' (default) returns query->target scores
                      - 'mean' returns the mean of query->target and
                        target->query scores
                      - 'min' returns the minium between query->target and
                        target->query scores
                      - 'max' returns the maximum between query->target and
                        target->query scores
    normalized :    bool, optional
                    Whether to return normalized NBLAST scores.
    smart :         bool
                    If True, will NBLAST downsampled dotprops first and, for
                    each query, take only the top 10% forward for a full NBLAST.
    use_alpha :     bool, optional
                    Emphasizes neurons' straight parts (backbone) over parts
                    that have lots of branches.
    smat :          str | pd.DataFrame
                    Score matrix. If 'auto' (default), will use scoring matrices
                    from FCWB. Same behaviour as in R's nat.nblast
                    implementation. If ``smat=None`` the scores will be
                    generated as the product of the distances and the dotproduct
                    of the vectors of nearest-neighbor pairs.
    limit_dist :    float | "auto" | None
                    Sets the max distance for the nearest neighbor search
                    (`distance_upper_bound`). Typically this should be the
                    highest distance considered by the scoring function. If
                    "auto", will extract that value from the scoring matrix.
    n_cores :       int, optional
                    Max number of cores to use for nblasting. Default is
                    ``os.cpu_count() // 2``. This should ideally be an even
                    number as that allows optimally splitting queries onto
                    individual processes.
    dtype :         str [e.g. "float32"] | np.dtype
                    Precision for scores. Defaults to 32 bit (single) floats.
                    This is useful to reduce the memory footprint for very large
                    matrices. In real-world scenarios 32 bit (single)- and
                    depending on the purpose even 16 bit (half) - are typically
                    sufficient.
    return_frame :  bool
                    Whether to return a Dask dataframe.
    progress :      bool
                    Whether to show progress bars.

    References
    ----------
    Costa M, Manton JD, Ostrovsky AD, Prohaska S, Jefferis GS. NBLAST: Rapid,
    Sensitive Comparison of Neuronal Structure and Construction of Neuron
    Family Databases. Neuron. 2016 Jul 20;91(2):293-311.
    doi: 10.1016/j.neuron.2016.06.012.

    Returns
    -------
    Dask DataFrame
                    Only if ``return_frame=True``. Note that the Dask DataFrame
                    will be alphabetically ordered along both rows and columns.
                    This will likely be different from the order in the
                    query/target dotprops. The stored Zarr array will have the
                    correct order though.

    """
    if isinstance(out, type(None)):
        raise ValueError('Must provide a output folder as `out`')

    # Make sure we're working on NeuronLists
    query_dps = NeuronList(query)
    # We will shuffle to avoid having imbalanced loading of the NBLASTERs
    random.shuffle(query_dps.neurons)

    if smart:
        query_ds = navis.downsample_neuron(query_dps, 10,
                                           parallel=n_cores > 1,
                                           n_cores=n_cores)
    else:
        query_ds = None

    aba = False
    if isinstance(target, type(None)):
        aba = True
        target, target_dps, target_ds = query, query_dps, query_ds
    else:
        target_dps = NeuronList(target)
        random.shuffle(target_dps.neurons)

        if smart:
            target_ds = navis.downsample_neuron(target_dps, 10,
                                                parallel=n_cores > 1,
                                                n_cores=n_cores)

    # Run NBLAST preflight checks
    navis.nbl.nblast_funcs.nblast_preflight(query_dps, target_dps,
                                            n_cores,
                                            req_unique_ids=True)

    # Find an optimal partition that minimizes the number of neurons
    # we have to send to each process
    n_rows, n_cols = navis.nbl.nblast_funcs.find_optimal_partition(n_cores,
                                                                   query_dps,
                                                                   target_dps)

    # Prepare output directory
    out = Path(out).expanduser().absolute() / 'scores'
    lockfile = Path(f'{out}.sync')
    synchronizer = zarr.ProcessSynchronizer(lockfile)
    z = zarr.open_array(out, mode='w', shape=(len(query_dps), len(target_dps)),
                        chunks=True,  # (1000, 1000),
                        synchronizer=synchronizer,
                        dtype=dtype)

    try:
        z.attrs['queries'] = query_dps.id.tolist()
    except BaseException:
        logging.warning('Failed to write query IDs to Zarr array as attribute')

    try:
        z.attrs['targets'] = target_dps.id.tolist()
    except BaseException:
        logging.warning('Failed to write target IDs to Zarr array as attribute')

    nblasters = []
    with config.tqdm(desc='Preparing',
                     total=n_rows * n_cols,
                     leave=False,
                     disable=not progress) as pbar:
        query_idx = np.arange(len(query_dps))
        target_idx = np.arange(len(target_dps))
        offset = [0, 0]
        for q in np.array_split(query_idx, n_rows):
            offset[1] = 0
            for t in np.array_split(target_idx, n_cols):
                # Initialize NBlaster
                this = DiskBlaster(use_alpha=use_alpha,
                                   normalized=normalized,
                                   smat=smat,
                                   limit_dist=limit_dist,
                                   dtype=dtype,
                                   offset=tuple(offset),  # need to make copy here
                                   out=out,
                                   progress=progress)

                # Add queries and target
                this.append(query_dps[q],
                            downsampled=None if not smart else query_ds[q])
                this.append(target_dps[t],
                            downsampled=None if not smart else target_dps[t])

                # Keep track of indices of queries and targets
                this.queries = np.arange(len(q))
                this.targets = np.arange(len(t)) + len(q)
                this.pbar_position = len(nblasters) % n_cores

                offset[1] += len(t)

                nblasters.append(this)
                pbar.update(1)

            offset[0] += len(q)

    # If only one core, we don't need to break out the multiprocessing
    if n_cores == 1:
        _ = this.multi_query_target(this.queries,
                                    this.targets,
                                    smart=smart,
                                    scores=scores)
    else:
        with ProcessPoolExecutor(max_workers=n_cores) as pool:
            # Each nblaster is passed to its own process
            futures = [pool.submit(this.multi_query_target,
                                   q_idx=this.queries,
                                   t_idx=this.targets,
                                   smart=smart,
                                   scores=scores) for this in nblasters]

            results = [f.result() for f in futures]

    if return_frame:
        scores = dd.from_array(z, columns=target_dps.id)
        scores.columns.name = 'target'

        scores['query'] = dd.from_array(query_dps.id)
        scores = scores.set_index('query')
        # This aligns rows with columns after re-ordering
        scores = scores[sorted(scores.columns)]

        return scores


def nblast_sparse(query: Union[Dotprops, NeuronList],
                  target: Optional[str] = None,
                  threshold: float = 0,
                  scores: Union[Literal['forward'],
                                Literal['mean'],
                                Literal['min'],
                                Literal['max']] = 'forward',
                  normalized: bool = True,
                  smart: bool = False,
                  use_alpha: bool = False,
                  smat: Optional[Union[str, pd.DataFrame]] = 'auto',
                  limit_dist: Optional[Union[Literal['auto'], int, float]] = None,
                  n_cores: int = os.cpu_count() // 2,
                  dtype: Union[str, np.dtype] = 'float32',
                  progress: bool = True) -> pd.DataFrame:
    """NBLAST query against target neurons and store results as sparse matrix.

    Parameters
    ----------
    query :         Dotprops | NeuronList
                    Query neuron(s) to NBLAST against the targets. Neurons
                    should be in microns as NBLAST is optimized for that and
                    have similar sampling resolutions.
    target :        Dotprops | NeuronList, optional
                    Target neuron(s) to NBLAST against. Neurons should be in
                    microns as NBLAST is optimized for that and have
                    similar sampling resolutions. If not provided, will NBLAST
                    queries against themselves.
    threshold :     float
                    NBLAST scores below this will be ignored and not stored.
                    If this value is too low (i.e. to few values get dropped)
                    the resulting sparse matrix might turn out using more memory
                    than a corresponding dense matrix.
    scores :        'forward' | 'mean' | 'min' | 'max'
                    Determines the final scores:
                      - 'forward' (default) returns query->target scores
                      - 'mean' returns the mean of query->target and
                        target->query scores
                      - 'min' returns the minium between query->target and
                        target->query scores
                      - 'max' returns the maximum between query->target and
                        target->query scores
    normalized :    bool, optional
                    Whether to return normalized NBLAST scores.
    smart :         bool
                    If True, will NBLAST downsampled dotprops first and, for
                    each query, take only the top 10% forward for a full NBLAST.
    use_alpha :     bool, optional
                    Emphasizes neurons' straight parts (backbone) over parts
                    that have lots of branches.
    smat :          str | pd.DataFrame
                    Score matrix. If 'auto' (default), will use scoring matrices
                    from FCWB. Same behaviour as in R's nat.nblast
                    implementation. If ``smat=None`` the scores will be
                    generated as the product of the distances and the dotproduct
                    of the vectors of nearest-neighbor pairs.
    limit_dist :    float | "auto" | None
                    Sets the max distance for the nearest neighbor search
                    (`distance_upper_bound`). Typically this should be the
                    highest distance considered by the scoring function. If
                    "auto", will extract that value from the scoring matrix.
    n_cores :       int, optional
                    Max number of cores to use for nblasting. Default is
                    ``os.cpu_count() // 2``. This should ideally be an even
                    number as that allows optimally splitting queries onto
                    individual processes.
    dtype :         str [e.g. "float32"] | np.dtype
                    Precision for scores. Defaults to 32 bit (single) floats.
                    This is useful to reduce the memory footprint for very large
                    matrices. In real-world scenarios 32 bit (single)- and
                    depending on the purpose even 16 bit (half) - are typically
                    sufficient.
    progress :      bool
                    Whether to show progress bars.

    References
    ----------
    Costa M, Manton JD, Ostrovsky AD, Prohaska S, Jefferis GS. NBLAST: Rapid,
    Sensitive Comparison of Neuronal Structure and Construction of Neuron
    Family Databases. Neuron. 2016 Jul 20;91(2):293-311.
    doi: 10.1016/j.neuron.2016.06.012.

    Returns
    -------
    pandas.DataFrame
                    Where each column is a sparse COO-array.

    """
    # Make sure we're working on NeuronLists
    query_dps = NeuronList(query)
    # We will shuffle to avoid having imbalanced loading of the NBLASTERs
    random.shuffle(query_dps.neurons)

    if smart:
        query_ds = navis.downsample_neuron(query_dps, 10,
                                           parallel=n_cores > 1,
                                           n_cores=n_cores)
    else:
        query_ds = None

    aba = False
    if isinstance(target, type(None)):
        aba = True
        target, target_dps, target_ds = query, query_dps, query_ds
    else:
        target_dps = NeuronList(target)
        random.shuffle(target_dps.neurons)

        if smart:
            target_ds = navis.downsample_neuron(target_dps, 10,
                                                parallel=n_cores > 1,
                                                n_cores=n_cores)

    # Run NBLAST preflight checks
    navis.nbl.nblast_funcs.nblast_preflight(query_dps, target_dps,
                                            n_cores,
                                            req_unique_ids=True)

    # Find an optimal partition that minimizes the number of neurons
    # we have to send to each process
    n_rows, n_cols = navis.nbl.nblast_funcs.find_optimal_partition(n_cores,
                                                                   query_dps,
                                                                   target_dps)

    nblasters = []
    with config.tqdm(desc='Preparing',
                     total=n_rows * n_cols,
                     leave=False,
                     disable=not progress) as pbar:
        query_idx = np.arange(len(query_dps))
        target_idx = np.arange(len(target_dps))
        offset = [0, 0]
        for q in np.array_split(query_idx, n_rows):
            offset[1] = 0
            for t in np.array_split(target_idx, n_cols):
                # Initialize NBlaster
                this = SparseBlaster(use_alpha=use_alpha,
                                     normalized=normalized,
                                     smat=smat,
                                     limit_dist=limit_dist,
                                     dtype=dtype,
                                     offset=tuple(offset),  # need to make copy here
                                     threshold=threshold,
                                     progress=progress)

                # Add queries and target
                this.append(query_dps[q],
                            downsampled=None if not smart else query_ds[q])
                this.append(target_dps[t],
                            downsampled=None if not smart else target_dps[t])

                # Keep track of indices of queries and targets
                this.queries = np.arange(len(q))
                this.targets = np.arange(len(t)) + len(q)
                this.pbar_position = len(nblasters) % n_cores

                offset[1] += len(t)

                nblasters.append(this)
                pbar.update(1)

            offset[0] += len(q)

    # If only one core, we don't need to break out the multiprocessing
    if n_cores == 1:
        data, coo = this.multi_query_target(this.queries,
                                              this.targets,
                                              smart=smart,
                                              scores=scores)
    else:
        with ProcessPoolExecutor(max_workers=n_cores) as pool:
            # Each nblaster is passed to its own process
            futures = [pool.submit(this.multi_query_target,
                                   q_idx=this.queries,
                                   t_idx=this.targets,
                                   smart=smart,
                                   scores=scores) for this in nblasters]

            results = [f.result() for f in futures]

            data = np.concatenate([r[0] for r in results], dtype=dtype)
            coo = np.vstack([r[1] for r in results])

    scores = sp.csc_array((data, (coo[:, 0], coo[:, 1])))

    scores = pd.DataFrame.sparse.from_spmatrix(scores,
                                               index=query_dps.id,
                                               columns=target_dps.id)
    scores = scores.astype(pd.SparseDtype(dtype, np.nan))

    return scores


def matches(query: Union[Dotprops, NeuronList],
            target: Optional[str] = None,
            N: int = 10,
            scores: Union[Literal['forward'],
                          Literal['mean'],
                          Literal['min'],
                          Literal['max']] = 'forward',
            normalized: bool = True,
            smart: bool = False,
            use_alpha: bool = False,
            smat: Optional[Union[str, pd.DataFrame]] = 'auto',
            limit_dist: Optional[Union[Literal['auto'], int, float]] = None,
            n_cores: int = os.cpu_count() // 2,
            precision: Union[int, str, np.dtype] = 32,
            batch_size: Optional[int] = None,
            progress: bool = True) -> pd.DataFrame:
    """Find matches for query neurons amongst pool of target.

    Parameters
    ----------
    query :         Dotprops | NeuronList
                    Query neuron(s) to NBLAST against the targets. Neurons
                    should be in microns as NBLAST is optimized for that and
                    have similar sampling resolutions.
    target :        Dotprops | NeuronList, optional
                    Target neuron(s) to NBLAST against. Neurons should be in
                    microns as NBLAST is optimized for that and have
                    similar sampling resolutions. If not provided, will NBLAST
                    queries against themselves.
    N :             int [1-len(targets)]
                    Number of matches to calculate.
    scores :        'forward' | 'mean' | 'min' | 'max'
                    Determines the final scores:
                      - 'forward' (default) returns query->target scores
                      - 'mean' returns the mean of query->target and
                        target->query scores
                      - 'min' returns the minium between query->target and
                        target->query scores
                      - 'max' returns the maximum between query->target and
                        target->query scores
    normalized :    bool, optional
                    Whether to return normalized NBLAST scores.
    smart :         bool
                    If True, will NBLAST downsampled dotprops first and, for
                    each query, take only the top 10% forward for a full NBLAST.
    use_alpha :     bool, optional
                    Emphasizes neurons' straight parts (backbone) over parts
                    that have lots of branches.
    smat :          str | pd.DataFrame
                    Score matrix. If 'auto' (default), will use scoring matrices
                    from FCWB. Same behaviour as in R's nat.nblast
                    implementation. If ``smat=None`` the scores will be
                    generated as the product of the distances and the dotproduct
                    of the vectors of nearest-neighbor pairs.
    limit_dist :    float | "auto" | None
                    Sets the max distance for the nearest neighbor search
                    (`distance_upper_bound`). Typically this should be the
                    highest distance considered by the scoring function. If
                    "auto", will extract that value from the scoring matrix.
    n_cores :       int, optional
                    Max number of cores to use for nblasting. Default is
                    ``os.cpu_count() // 2``. This should ideally be an even
                    number as that allows optimally splitting queries onto
                    individual processes.
    precision :     int [16, 32, 64] | str [e.g. "float64"] | np.dtype
                    Precision for scores. Defaults to 64 bit (double) floats.
                    This is useful to reduce the memory footprint for very large
                    matrices. In real-world scenarios 32 bit (single)- and
                    depending on the purpose even 16 bit (half) - are typically
                    sufficient.
    batch_size :    int, optional
                    Set the number of Dotprops per process. This can be useful
                    to reduce the memory footprint at the cost of longer run
                    times. By default (``None``), queries and targets will be
                    evenly distributed across n_cores. Ignored if ``n_cores=1``.
    progress :      bool
                    Whether to show progress bars.

    References
    ----------
    Costa M, Manton JD, Ostrovsky AD, Prohaska S, Jefferis GS. NBLAST: Rapid,
    Sensitive Comparison of Neuronal Structure and Construction of Neuron
    Family Databases. Neuron. 2016 Jul 20;91(2):293-311.
    doi: 10.1016/j.neuron.2016.06.012.

    Returns
    -------
    dask.DataFrame
                    Notable rows/columns are sorted and not in the original
                    order!

    """
    pass
