from .forward_mode import ForwardMode


class AutoDiff:
    """
    TODO
    Implement support for:

    np.ndarray.shape / np.shape
    np.ndarray.size
    np.ndarray.ndim
    np.ndarray.T
    np.ndarray.transpose / np.transpose
    np.ndarray.copy / np.copy
    np.ndarray.astype
    np.ndarray.fill
    np.ndarray.reshape / np.reshape
    np.ndarray.resize / np.resize
    np.ndarray.flat / np.flat
    np.ndarray.flatten / np.flatten
    np.ndarray.ravel / np.ravel
    np.ndarray.repeat / np.repeat
    np.ndarray.sort / np.sort
    np.ndarray.argsort / np.argsort
    np.ndarray.choose / np.choose
    np.ndarray.nonzero / np.nonzero
    np.ndarray.compress / np.compress
    np.ndarray.diagonal / np.diagonal
    np.ndarray.max / np.amax
    np.ndarray.argmax / np.argmax
    np.ndarray.min / np.amin
    np.ndarray.argmin / np.argmin
    np.ndarray.ptp / np.ptp
    np.ndarray.clip / np.clip
    np.ndarray.round / np.around
    np.ndarray.trace / np.trace
    np.ndarray.all / np.all
    np.ndarray.any / np.any
    np.ndarray.__lt__
    np.ndarray.__le__
    np.ndarray.__gt__
    np.ndarray.__ge__
    np.ndarray.__eq__
    np.ndarray.__ne__
    np.ndarray.__bool__
    np.ndarray.__len__
    np.ndarray.__getitem__
    np.ndarray.__setitem__
    np.ndarray.__repr__
    np.where
    np.concatenate
    np.stack
    np.block
    np.vstack
    np.hstack
    np.dstack
    np.column_stack
    np.row_stack
    np.split
    np.array_split
    np.dsplit
    np.hsplit
    np.vsplit
    np.tile
    np.delete
    np.insert
    np.append
    np.flip
    np.fliplr
    np.flipud
    np.roll
    np.rot90
    np.isfinite
    np.isinf
    np.isnan
    np.isneginf
    np.isposinf
    np.maximum
    np.minimum
    np.fmax
    np.fmin
    np.nanmax
    np.nanmin

    np.ndarray.sum / np.sum
    np.ndarray.cumsum / np.cumsum
    np.ndarray.mean / np.mean
    np.ndarray.var / np.var
    np.ndarray.std / np.std
    np.ndarray.prod / np.prod
    np.ndarray.cumprod / np.cumprod
    np.ndarray.__neg__
    np.ndarray.__pos__
    np.ndarray.__abs__
    np.ndarray.__add__
    np.ndarray.__sub__
    np.ndarray.__mul__
    np.ndarray.__truediv__
    np.ndarray.__floordiv__
    np.ndarray.__mod__
    np.ndarray.__divmod__
    np.ndarray.__pow__
    np.ndarray.__and__
    np.ndarray.__or__
    np.ndarray.__xor__
    np.ndarray.__matmul__
    np.trim_zeros
    np.unique
    np.dot
    np.vdot
    np.linalg.multi_dot
    np.inner
    np.outer
    np.matrix_power
    np.linalg.norm
    np.linalg.cond
    np.linalg.det
    np.linalg.matrix_rank
    np.linalg.slogdet
    np.linalg.inv
    np.sin
    np.cos
    np.tan
    np.acrsin
    np.acrcos
    np.acrtan
    np.sinh
    np.cosh
    np.tanh
    np.arcsinh
    np.arccosh
    np.arctanh
    np.prod
    np.sum
    np.nanprod
    np.nansum
    np.cumprod
    np.cumsum
    np.nancumprod
    np.nancumsum
    np.exp
    np.expm1
    np.exp2
    np.log
    np.log10
    np.log2
    np.logp1
    np.sqrt
    np.cbrt
    np.square
    np.absolute
    np.fabs
    np.sign
    np.heavuside
    np.nan_to_num

    """

    forward_mode = ForwardMode()
