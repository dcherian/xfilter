import dask
import numpy as np
import pytest
import xarray as xr

from .xfilter import lowpass, highpass, bandpass


def assert_allclose(a, b, **kwargs):
    xr.testing.assert_allclose(a, b, **kwargs)
    xr.testing._assert_internal_invariants(a)
    xr.testing._assert_internal_invariants(b)


@pytest.fixture
def test_data():
    π = np.pi
    t = np.arange(0, 20001, 4)  # in days
    freqs = dict(zip(["high", "mid", "low"], [5, 100, 1000]))
    data = xr.Dataset()
    for name, f in freqs.items():
        data[name] = xr.DataArray(
            np.sin(2 * π / f * t), dims=["time"], coords={"time": t}
        )

    data["total"] = data.low + data.mid + data.high
    data.attrs["freqs"] = freqs.values()
    return data


@pytest.mark.parametrize(
    "filt, freq, expect",
    [
        (lowpass, 1 / 250, "low"),
        (highpass, 1 / 40, "high"),
        (bandpass, (1 / 40, 1 / 250), "mid"),
    ],
)
def test_filters(test_data, filt, freq, expect):
    actual = filt(test_data.total, coord="time", freq=freq, order=4)
    expected = test_data[expect].where(~np.isnan(actual))
    assert_allclose(actual, expected, atol=1e-2)


@pytest.mark.xfail(reason="use_overlap needs to be fixed.")
@pytest.mark.parametrize(
    "filt, freq", [(lowpass, 1 / 50), (highpass, 1 / 50), (bandpass, (1 / 40, 1 / 250))]
)
def test_map_overlap(test_data, filt, freq):
    actual = filt(
        test_data.total.chunk({"time": 1001}), coord="time", freq=freq
    ).compute()
    expected = filt(test_data.total, coord="time", freq=freq)

    assert (np.isnan(actual) == np.isnan(expected)).all()
    assert_allclose(actual, expected)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "filt, freq", [(lowpass, 1 / 50), (highpass, 1 / 50), (bandpass, (1 / 40, 1 / 250))]
)
def test_gappy_filter(test_data, filt, freq):
    da = test_data.total.copy()
    da[500:1000] = np.nan
    da[3000:3100] = np.nan
    da = da.expand_dims(x=10)

    chunked = da.chunk({"time": -1, "x": 1})

    kwargs = dict(coord="time", freq=freq)

    numpy_ans = filt(da, coord="time", freq=freq, gappy=False)
    numpy_ans_1d = filt(da, coord="time", freq=freq, gappy=True)
    dask_ans = filt(chunked, coord="time", freq=freq, gappy=False)
    dask_ans_1d = filt(chunked, coord="time", freq=freq, gappy=True)

    assert isinstance(dask_ans.data, dask.array.Array)
    xr.testing.assert_allclose(numpy_ans, numpy_ans_1d)
    xr.testing.assert_equal(numpy_ans, dask_ans.compute())
    xr.testing.assert_equal(numpy_ans_1d, dask_ans_1d.compute())
    xr.testing.assert_allclose(dask_ans.compute(), dask_ans_1d.compute())
