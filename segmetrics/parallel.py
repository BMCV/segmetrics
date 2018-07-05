import study
import multiprocessing
import signal
import itertools
import dill


def process(study, get_actual_func, get_expected_func, chunk_ids, num_forks=None, is_actual_unique=True, is_expected_unique=True, callback=None):
    if num_forks is None: num_forks = multiprocessing.cpu_count()
    num_forks = min([num_forks, len(chunk_ids)])
    for chunk_idx, chunk_result in enumerate(fork.imap_unordered(num_forks,
                                                                 _process_chunk,
                                                                 study,
                                                                 dill.dumps(get_actual_func),
                                                                 dill.dumps(get_expected_func),
                                                                 unroll(chunk_ids),
                                                                 is_actual_unique,
                                                                 is_expected_unique)):
        chunk_id, chunk_study = chunk_result
        study.merge(chunk_study, chunk_ids=[chunk_id])
        if callback is not None: callback(chunk_idx + 1, len(chunk_ids))
        yield chunk_id


def process_all(*args, **kwargs):
    for _ in process(*args, **kwargs): pass


def _process_chunk(study, get_actual_func, get_expected_func, chunk_id, is_actual_unique, is_expected_unique):
    actual   = dill.loads(get_actual_func  )(chunk_id)
    expected = dill.loads(get_expected_func)(chunk_id)
    study.set_expected(expected, unique=is_expected_unique)
    study.process(actual, unique=is_actual_unique, chunk_id=chunk_id)
    return chunk_id, study


class _Sequence:
    def __init__(self, val):
            self.val = val


def unroll(seq):
    return _Sequence(seq)


def _get_args_chain(args):
    n = max(len(arg.val) for arg in list(args) if isinstance(arg, _Sequence))
    return n, (arg.val if isinstance(arg, _Sequence) else [arg] * n for arg in args)


class _UnrollArgs:
    def __init__(self, f):
        self.f = f
    
    def __call__(self, args):
        return self.f(*args)


class fork: # namespace

    _forked = False
    DEBUG   = False
    
    @staticmethod
    def map(processes, f, *args):
        assert processes >= 1, 'number of processes must be at least 1'
        assert not fork._forked, 'process was already forked before'

        run_parallel = processes >= 2 and not fork.DEBUG
        n, real_args = _get_args_chain(args)
        real_args = list(zip(*real_args))
    
        # we need to ensure that SIGINT is handled correctly,
        # see for reference: http://stackoverflow.com/a/35134329/1444073
        if run_parallel:
            original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            pool = multiprocessing.Pool(processes=processes)
            signal.signal(signal.SIGINT, original_sigint_handler)
        
        fork._forked = True
        try:
            if run_parallel:
                chunksize = int(round(float(n) / processes))
                result = pool.map(_UnrollArgs(f), real_args, chunksize)
                pool.close()
                return result
            else:
                return map(_UnrollArgs(f), real_args)
        except:
            if run_parallel: pool.terminate()
            raise
        finally:
            fork._forked = False
    
    @staticmethod
    def apply(processes, f, *args):
        fork.map(processes, f, *args)

    @staticmethod
    def imap_unordered(processes, f, *args, **kwargs):
        assert processes >= 1, 'number of processes must be at least 1'
        assert not fork._forked, 'process was already forked before'

        run_parallel = processes >= 2 and not fork.DEBUG
        n, real_args = _get_args_chain(args)
        real_args = list(zip(*real_args))
    
        # we need to ensure that SIGINT is handled correctly,
        # see for reference: http://stackoverflow.com/a/35134329/1444073
        if run_parallel:
            original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            pool = multiprocessing.Pool(processes=processes)
            signal.signal(signal.SIGINT, original_sigint_handler)

        fork._forked = True
        try:
            if run_parallel:
                chunksize = int(round(float(n) / processes)) if kwargs.get('use_chunks') else 1
                for result in pool.imap_unordered(_UnrollArgs(f), real_args, chunksize):
                    yield result
                pool.close()
            else:
                for result in itertools.imap(_UnrollArgs(f), real_args):
                    yield result
        except:
            if run_parallel: pool.terminate()
            raise
        finally:
            fork._forked = False

