from sotodlib import core
from sotodlib.site_pipeline.utils.obsdb import get_obslist
from sotodlib.utils.procs_pool import get_exec_env
from sotodlib.site_pipeline.utils.pipeline import main_launcher
import argparse

def compute_samp_counts(context, obs):
    ctx = core.Context(context)
    try:
        meta = ctx.get_meta(obs)
    except Exception as e:
        print(e)
        return 0
    dethrs = meta.dets.count*meta.samps.count/200/3600
    print(obs, dethrs)
    return dethrs

def _main(executor, as_completed_callable, context, obslist):
    ctx = core.Context(context)
    obslist = get_obslist(ctx, query=obslist)
    futures = []
    detsum = 0
    for obs in obslist:
        future = executor.submit(compute_samp_counts, context, obs['obs_id'])
        futures.append(future)
    for future in as_completed_callable(futures):
        detsum += future.result()
        futures.remove(future)
    print(detsum, 'complete')

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('context', help="Preprocessing Context File")
    parser.add_argument(
        '--obslist',
        help="path to file of obslist",
        type=str
    )
    parser.add_argument('--nproc', type=int)
    return parser

def main(context, obslist, nproc):
    rank, executor, as_completed_callable = get_exec_env(nproc)
    if rank == 0:
        _main(executor, as_completed_callable, context, obslist)

if __name__ == '__main__':
    main_launcher(main, get_parser)
