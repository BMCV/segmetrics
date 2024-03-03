import numpy as np


def bbox(*masks, margin=0):
    assert len(masks) > 0
    assert len(masks) == 1 or all(
        masks[0].shape == mask.shape for mask in masks[1:]
    )
    _rmin, _rmax = np.inf, -np.inf
    _cmin, _cmax = np.inf, -np.inf
    for mask in masks:
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax  = np.where(rows)[0][[0, -1]]
        cmin, cmax  = np.where(cols)[0][[0, -1]]
        _rmin, _rmax = min((_rmin, rmin)), max((_rmax, rmax))
        _cmin, _cmax = min((_cmin, cmin)), max((_cmax, cmax))
    _rmin -= margin
    _rmax += margin
    _cmin -= margin
    _cmax += margin
    _rmin, _rmax = max((_rmin, 0)), min((_rmax, mask.shape[0] - 1))
    _cmin, _cmax = max((_cmin, 0)), min((_cmax, mask.shape[1] - 1))
    sel = np.s_[_rmin:_rmax + 1, _cmin:_cmax + 1]
    return sel, (_rmin, _rmax), (_cmin, _cmax)
