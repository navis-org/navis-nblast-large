import navis
import numbers
import os
import random
import uuid

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor
from typing import Union, Optional, List
from typing_extensions import Literal

from time import time
from pykdtree.kdtree import KDTree

utils = navis.utils
config = navis.config
smat_path = navis.nbl.nblast_funcs.smat_path
Dotprops = navis.Dotprops
NeuronList = navis.NeuronList


class NBlaster(navis.nbl.nblast_funcs.NBlaster):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neurons_ds = []
        self.self_hits_ds = []

    def append(self, dotprops, downsampled=None):
        """Append dotprops."""
        if downsampled :
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
        data = self.neurons_ds[q_idx].dist_dots(self.neurons[t_idx],
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

    def multi_query_target(self, q_idx, t_idx, scores='forward', smart=False):
        """BLAST multiple queries against multiple targets.

        Parameters
        ----------
        q_idx,t_idx :       iterable
                            Iterable of query/target neuron indices to BLAST.
        scores :            "forward" | "mean" | "min" | "max"
                            Which scores to return.

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

        res = da.zeros((len(q_idx), len(t_idx)),
                       chunks=(2500, len(t_idx)),  # only chunk along first axis
                       dtype=self.dtype)
        for i, q in enumerate(config.tqdm(q_idx,
                                          desc=self.desc,
                                          leave=False,
                                          position=position,
                                          disable=not self.progress)):
            row = np.zeros(len(t_idx))
            for k, t in enumerate(t_idx):
                if not smart:
                    row[k] = self.single_query_target(q, t, scores=scores)
                else:
                    row[k] = self.single_query_target_ds(q, t, scores=scores)

            if smart:
                # Run full NBLAST for the top 10%
                for k in np.argsort(row)[-top_N:]:
                    row[k] = self.single_query_target(q, t_idx[k], scores=scores)

            # Writing a row at a time is way more efficient than writing single
            # values at a time
            res[i, :] = row

        return res


def nblast(query: Union[Dotprops, NeuronList],
           target: Optional[str] = None,
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
    """NBLAST query against target neurons.

    This implements the NBLAST algorithm from Costa et al. (2016) (see
    references) and mirror the implementation in R's ``nat.nblast``
    (https://github.com/natverse/nat.nblast).

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
                    ``os.cpu_count() - 2``. This should ideally be an even
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
                                            batch_size=batch_size,
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
        for q in np.array_split(query_idx, n_rows):
            for t in np.array_split(target_idx, n_cols):
                # Initialize NBlaster
                this = NBlaster(use_alpha=use_alpha,
                                normalized=normalized,
                                smat=smat,
                                limit_dist=limit_dist,
                                dtype=precision,
                                progress=progress)

                # Add queries and target
                this.append(query_dps[q],
                            downsampled=None if not smart else query_ds[q])
                this.append(target_dps[q],
                            downsampled=None if not smart else target_dps[q])

                # Keep track of indices of queries and targets
                this.queries = np.arange(len(q))
                this.targets = np.arange(len(t)) + len(q)
                this.pbar_position = len(nblasters) % n_cores

                nblasters.append(this)
                pbar.update(1)

    # If only one core, we don't need to break out the multiprocessing
    if n_cores == 1:
        return this.multi_query_target(this.queries,
                                       this.targets,
                                       smart=smart,
                                       scores=scores)

    with ProcessPoolExecutor(max_workers=n_cores) as pool:
        # Each nblaster is passed to its own process
        futures = [pool.submit(this.multi_query_target,
                               q_idx=this.queries,
                               t_idx=this.targets,
                               smart=smart,
                               scores=scores) for this in nblasters]

        results = [f.result() for f in futures]

    # Combine individual tiles into a single Dask DataFrame
    rows = []
    for i in range(n_rows):
        rows.append(da.hstack(results[i * n_cols: (i+1) * n_cols]))
    scores = da.vstack(rows)
    scores = dd.from_dask_array(scores, columns=target_dps.id)
    scores.columns.name = 'target'

    # Set the index
    # Note that this will re-order rows
    scores['query'] = dd.from_array(query_dps.id)
    scores = scores.set_index('query')
    # This aligns rows with columns after re-ordering
    scores = scores[sorted(scores.columns)]

    return scores
