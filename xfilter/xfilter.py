import dask
import numpy as np
import warnings
import xarray as xr

from scipy import signal


def _get_num_discard(kwargs):

    num_discard = kwargs.get("num_discard")
    if num_discard == "auto":
        if "irlen" in kwargs:
            num_discard = kwargs["irlen"]
        else:
            num_discard = estimate_impulse_response_len(b, a)

    return num_discard


def _process_time(time, cycles_per="s"):

    time = time.copy()
    dt = np.nanmedian(np.diff(time.values) / np.timedelta64(1, cycles_per))

    time = np.cumsum(time.copy().diff(dim=time.dims[0]) / np.timedelta64(1, cycles_per))

    return dt, time


def estimate_impulse_response_len(b, a, eps=1e-3):
    """ From scipy filtfilt docs.
        Input:
             b, a : filter params
             eps  : How low must the signal drop to? (default 1e-2)
    """

    _, p, _ = signal.tf2zpk(b, a)
    r = np.max(np.abs(p))
    approx_impulse_len = int(np.ceil(np.log(eps) / np.log(r)))

    return approx_impulse_len


def gappy_1d_filter(data, b, a, num_discard, **kwargs):
    out = np.zeros_like(data) * np.nan
    assert data.ndim == 1
    segstart, segend = find_segments(data)
    for index, start in np.ndenumerate(segstart):
        stop = segend[index]
        try:
            out[..., start:stop] = signal.filtfilt(
                b, a, data[..., start:stop], **kwargs
            )
            if num_discard is not None and num_discard > 0:
                out[..., start : start + num_discard] = np.nan
                out[..., stop - num_discard : stop] = np.nan
        except ValueError:
            # segment is not long enough for filtfilt
            pass
    return out


def gappy_filter(data, b, a, gappy=True, **kwargs):

    num_discard = kwargs.pop("num_discard")

    if gappy:
        if isinstance(data, dask.array.Array):
            apply_func = dask.array.apply_along_axis
        else:
            apply_func = np.apply_along_axis
        out = apply_func(gappy_1d_filter, -1, data, b, a, num_discard, **kwargs)

    else:
        assert "axis" not in kwargs
        kwargs["axis"] = -1

        out = signal.filtfilt(b, a, data, **kwargs)
        if num_discard is not None and num_discard > 0:
            out[..., :num_discard] = np.nan
            out[..., -num_discard:] = np.nan

    return out


def find_segments(var):
    """
      Finds and return valid index ranges for the input time series.
      Input:
            var - input time series
      Output:
            start - starting indices of valid ranges
            stop  - ending indices of valid ranges
    """

    NotNans = np.double(~np.isnan(var))
    edges = np.diff(NotNans)
    start = np.where(edges == 1)[0]
    stop = np.where(edges == -1)[0]

    if start.size == 0 and stop.size == 0:
        start = np.array([0])
        stop = np.array([len(var) - 1])

    else:
        start = start + 1
        if ~np.isnan(var[0]):
            start = np.insert(start, 0, 0)

        if ~np.isnan(var[-1]):
            stop = np.append(stop, len(var) - 1)

    return start, stop


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

    if data[coord].dtype.kind == "M":
        dx, x = _process_time(data[coord], cycles_per)
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

    kwargs.setdefault("num_discard", "auto")
    kwargs.setdefault("method", "gust")

    if gappy is not None:
        warnings.warn("The 'gappy' kwarg is now deprecated.")
    else:
        gappy = True

    if kwargs["method"] == "gust" and "irlen" not in kwargs:
        kwargs["irlen"] = estimate_impulse_response_len(b, a)
    num_discard = _get_num_discard(kwargs)
    kwargs["num_discard"] = num_discard

    kwargs.update(b=b, a=a, gappy=gappy)

    valid = data.notnull()
    filled = data.ffill(coord).bfill(coord)
    # I need distance from nearest NaN
    index = np.arange(data.sizes[coord])
    arange = xr.ones_like(data) * index
    invalid_arange = (
        arange.where(~valid)
        .interpolate_na(coord, "nearest", fill_value="extrapolate")
        .fillna(np.inf)  # this is only when there are no NaNs in the data.
    )
    distance = np.abs(arange - invalid_arange).where(data.notnull())

    if not use_overlap:
        filtered = xr.apply_ufunc(
            gappy_filter,
            filled,
            input_core_dims=[[coord]],
            output_core_dims=[[coord]],
            dask="parallelized",
            output_dtypes=[data.dtype],
            kwargs=kwargs,
        )

    else:
        num_discard = kwargs.pop("num_discard")
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
                gappy_filter,
                depth=depth,
                boundary="none",
                num_discard=None,  # don't discard since map_overlap's trimming will do this.
                meta=data.data._meta,
                **kwargs,
            )
        )

    # take out the beginning and end if necessary
    mask = xr.DataArray(
        np.ones((filtered.sizes[coord],), dtype=bool), dims=[coord], name=coord
    )
    if num_discard > 0:
        mask[:num_discard] = False
        mask[-num_discard:] = False
        filtered = filtered.where(mask)

    filtered = filtered.where((distance >= kwargs["num_discard"]) & mask)

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
