import multiprocessing
import signal
from typing import (
    Any,
    Callable,
    Optional,
    Sequence,
)

import dill

from segmetrics.study import Study
from segmetrics.typing import Image


def process(
    study: Study,
    get_actual_func: Callable[[Any], Image],
    get_expected_func: Callable[[Any], Image],
    sample_ids: Sequence[Any],
    num_forks: Optional[int] = None,
    is_actual_unique: bool = True,
    is_expected_unique: bool = True,
    callback: Optional[Callable[[int, int], None]] = None,
):
    if num_forks is None:
        num_forks = multiprocessing.cpu_count()
    num_forks = min([num_forks, len(sample_ids)])
    generator = _fork.imap_unordered(
        num_forks,
        _process_sample,
        study,
        dill.dumps(get_actual_func),
        dill.dumps(get_expected_func),
        _unroll(sample_ids),
        is_actual_unique,
        is_expected_unique,
    )
    for sample_idx, sample_result in enumerate(generator):
        sample_id, sample_study = sample_result

        # This happens when parallelization is off:
        if study is not sample_study:
            study.merge(sample_study, sample_ids=[sample_id])

        if callback is not None:
            callback(sample_idx + 1, len(sample_ids))
        yield sample_id


def process_all(*args, **kwargs):
    for _ in process(*args, **kwargs):
        pass


def _process_sample(
    study: Study,
    get_actual_func: Callable[[Any], Image],
    get_expected_func: Callable[[Any], Image],
    sample_id: Any,
    is_actual_unique: bool,
    is_expected_unique: bool,
):
    actual   = dill.loads(get_actual_func  )(sample_id)
    expected = dill.loads(get_expected_func)(sample_id)
    study.set_expected(expected, unique=is_expected_unique)
    study.process(sample_id, actual, unique=is_actual_unique)
    return sample_id, study


class _Sequence:
    def __init__(self, val: Sequence[Any]) -> None:
        self.val = val


def _unroll(seq: Sequence[Any]) -> _Sequence:
    return _Sequence(seq)


def _get_args_chain(args: Sequence[Any]):
    n = max(len(arg.val) for arg in list(args) if isinstance(arg, _Sequence))
    return n, (
        arg.val if isinstance(arg, _Sequence) else [arg] * n for arg in args
    )


class _UnrollArgs:
    def __init__(self, f):
        self.f = f

    def __call__(self, args):
        return self.f(*args)


class _fork:  # namespace

    _forked = False
    DEBUG   = False

    @staticmethod
    def map(processes, f, *args):
        assert processes >= 1, 'number of processes must be at least 1'
        assert not _fork._forked, 'process was already forked before'

        run_parallel = processes >= 2 and not _fork.DEBUG
        n, real_args = _get_args_chain(args)
        real_args = list(zip(*real_args))

        # we need to ensure that SIGINT is handled correctly,
        # see for reference: http://stackoverflow.com/a/35134329/1444073
        if run_parallel:
            original_sigint_handler = signal.signal(
                signal.SIGINT,
                signal.SIG_IGN,
            )
            pool = multiprocessing.Pool(processes=processes)
            signal.signal(signal.SIGINT, original_sigint_handler)

        _fork._forked = True
        try:
            if run_parallel:
                chunksize = int(round(float(n) / processes))
                result = pool.map(_UnrollArgs(f), real_args, chunksize)
                pool.close()
                return result
            else:
                return map(_UnrollArgs(f), real_args)
        except:  # noqa: E722
            if run_parallel:
                pool.terminate()
            raise
        finally:
            _fork._forked = False

    @staticmethod
    def apply(processes, f, *args):
        _fork.map(processes, f, *args)

    @staticmethod
    def imap_unordered(processes, f, *args, **kwargs):
        assert processes >= 1, 'number of processes must be at least 1'
        assert not _fork._forked, 'process was already forked before'

        run_parallel = processes >= 2 and not _fork.DEBUG
        n, real_args = _get_args_chain(args)
        real_args = list(zip(*real_args))

        # we need to ensure that SIGINT is handled correctly,
        # see for reference: http://stackoverflow.com/a/35134329/1444073
        if run_parallel:
            original_sigint_handler = signal.signal(
                signal.SIGINT,
                signal.SIG_IGN,
            )
            pool = multiprocessing.Pool(processes=processes)
            signal.signal(signal.SIGINT, original_sigint_handler)

        _fork._forked = True
        try:
            if run_parallel:
                chunksize = int(
                    round(float(n) / processes)
                ) if kwargs.get('use_chunks') else 1
                generator = pool.imap_unordered(
                    _UnrollArgs(f),
                    real_args,
                    chunksize,
                )
                for result in generator:
                    yield result
                pool.close()
            else:
                for result in map(_UnrollArgs(f), real_args):
                    yield result
        except:  # noqa: E722
            if run_parallel:
                pool.terminate()
            raise
        finally:
            _fork._forked = False
