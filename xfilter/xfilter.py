import dask
import numpy as np
import warnings
import xarray as xr

from scipy import signal


def _get_num_discard(kwargs, num_discard):

    if num_discard == "auto":
        if "irlen" in kwargs:
            num_discard = kwargs["irlen"]
        else:
            num_discard = estimate_impulse_response_len(b, a)

    return num_discard


def _process_time(time, cycles_per="s"):

    dt = np.nanmedian(
        np.diff(time.values).astype(np.timedelta64) / np.timedelta64(1, cycles_per)
    )
    return dt


def estimate_impulse_response_len(b, a, eps=1e-3):
    """From scipy filtfilt docs.
    Input:
         b, a : filter params
         eps  : How low must the signal drop to? (default 1e-2)
    """

    _, p, _ = signal.tf2zpk(b, a)
    r = np.max(np.abs(p))
    approx_impulse_len = int(np.ceil(np.log(eps) / np.log(r)))

    return approx_impulse_len


def filter_(data, b, a, **kwargs):
    out = signal.filtfilt(b, a, data, **kwargs)
    return out


def _is_datetime_like(da) -> bool:
    import numpy as np

    if np.issubdtype(da.dtype, np.datetime64) or np.issubdtype(
        da.dtype, np.timedelta64
    ):
        return True

    try:
        import cftime

        if isinstance(da.data[0], cftime.datetime):
            return True
    except ImportError:
        pass

    return False


def _wrap_butterworth(
    data, coord, freq, kind, cycles_per="s", order=2, debug=False, gappy=None, **kwargs
):
    """
    Inputs
    ------

    data : xr.DataArray
    coord : coordinate along which to filter
    freq : "frequencies" for filtering
    cycles_per: optional
        Units for frequency
    order : optional
        Butterworth filter order
    kwargs : dict, optional
        passed down to gappy_filter

    Outputs
    -------

    filtered : xr.DataArray
    """

    # if len(data.dims) > 1 and coord is None:
    #     raise ValueError('Specify coordinate along which to filter')
    # else:
    #     coord = data.coords[0]

    if _is_datetime_like(data[coord]):
        dx = _process_time(data[coord], cycles_per)
    else:
        dx = np.diff(data[coord][0:2].values)

    b, a = signal.butter(order, freq * dx / (1 / 2), btype=kind)

    data = data.copy().transpose(..., coord)

    if debug:
        import dcpy.ts
        import matplotlib.pyplot as plt

        f, ax = plt.subplots(2, 1, constrained_layout=True)
        data.plot(x=coord, ax=ax[0])
        dcpy.ts.PlotSpectrum(data, cycles_per=cycles_per, ax=ax[1])

    if data.chunks:
        chunks = dict(zip(data.dims, data.chunks))
        if len(chunks[coord]) > 1:
            use_overlap = True
        else:
            use_overlap = False
    else:
        use_overlap = False

    if gappy is not None:
        warnings.warn(
            UserWarning, "'gappy' kwarg is now deprecated and completely ignored."
        )

    num_discard = kwargs.pop("num_discard", "auto")
    kwargs.setdefault("method", "gust")
    if kwargs["method"] == "gust" and "irlen" not in kwargs:
        kwargs["irlen"] = estimate_impulse_response_len(b, a)
    kwargs.update(b=b, a=a, axis=-1)

    valid = data.notnull()
    if np.issubdtype(data.dtype, np.dtype(complex)):
        filled = data.real.ffill(coord).bfill(coord) + 1j * data.imag.ffill(
            coord
        ).bfill(coord)
    else:
        filled = data.ffill(coord).bfill(coord)

    # I need distance from nearest NaN
    index = np.arange(data.sizes[coord])
    arange = xr.ones_like(data.reset_coords(drop=True), dtype=int) * index
    invalid_arange = (
        arange.where(~valid)
        .interpolate_na(coord, "nearest", fill_value="extrapolate")
        .fillna(-1)  # when all points are valid
    )
    distance = np.abs(arange - invalid_arange).where(valid)

    if not use_overlap:
        filtered = xr.apply_ufunc(
            filter_,
            filled,
            input_core_dims=[[coord]],
            output_core_dims=[[coord]],
            dask="parallelized",
            output_dtypes=[data.dtype],
            kwargs=kwargs,
        )

    else:
        if not isinstance(data, xr.DataArray):
            raise ValueError("map_overlap implemented only for DataArrays.")
        irlen = estimate_impulse_response_len(b, a)
        axis = data.get_axis_num(coord)
        overlap = np.round(2 * irlen).astype(int)
        min_chunksize = 3 * overlap
        actual_chunksize = data.data.chunksize[axis]

        if actual_chunksize < min_chunksize:
            raise ValueError(
                f"Chunksize along {coord} = {actual_chunksize} < {min_chunksize}. Please rechunk"
            )

        depth = dict(zip(range(data.ndim), [0] * data.ndim))
        depth[data.ndim - 1] = overlap
        filtered = data.copy(
            data=dask.array.map_overlap(
                filled.data,
                filter_,
                depth=depth,
                boundary="none",
                meta=filled.data._meta,
                **kwargs,
            )
        )

    # take out the beginning and end if necessary
    mask = xr.DataArray(
        np.ones((filtered.sizes[coord],), dtype=bool),
        dims=[coord],
        name=coord,
        coords={coord: filtered[coord]},
    )
    num_discard = _get_num_discard(kwargs, num_discard)
    if num_discard > 0:
        mask[:num_discard] = False
        mask[-num_discard:] = False

    filtered = filtered.where((distance >= num_discard) & mask)

    if debug:
        filtered.plot(x=coord, ax=ax[0])
        ylim = ax[1].get_ylim()
        dcpy.ts.PlotSpectrum(filtered, cycles_per=cycles_per, ax=ax[1])
        ax[1].set_ylim(ylim)
        for ff in np.array(freq, ndmin=1):
            plt.axvline(ff)

    return filtered


def bandpass(data, coord, freq, **kwargs):
    if len(np.atleast_1d(freq)) != 2:
        raise ValueError(
            f"Expected freq to be a 2 element vector. Received ({freq}) instead."
        )
    return _wrap_butterworth(data, coord, np.sort(freq), kind="bandpass", **kwargs)


def lowpass(data, coord, freq, **kwargs):
    return _wrap_butterworth(data, coord, freq, kind="lowpass", **kwargs)


def highpass(data, coord, freq, **kwargs):
    return _wrap_butterworth(data, coord, freq, kind="highpass", **kwargs)
