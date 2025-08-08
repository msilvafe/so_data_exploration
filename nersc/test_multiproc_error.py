import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

def err_fn(a):
    if a%2 == 0:
        raise ValueError
    else:
        return a

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    nproc = 2
    with ProcessPoolExecutor(nproc) as exe:
        futures = [exe.submit(err_fn, r) for r in [0,1,2,3,4,5,6]]
        for future in as_completed(futures):
            try:
                a = future.result()
                print(a)
            except Exception as e:
                errmsg = f'{type(e)}: {e}'
                tb = ''.join(traceback.format_tb(e.__traceback__))
                print(f"ERROR: future.result()\n{errmsg}\n{tb}")
                continue
            futures.remove(future)  
