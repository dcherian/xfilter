# xfilter: A package for filtering xarray objects

Simple wrapping for scipy's Butterworth filter and `scipy.filtfilt`.

Works with gappy datasets by filling in values with a constant, filtering and then NaNing out gaps.

Automatically determines length of regions with edge effects by estimating the filter's impulse response length and NaNs those out.

Takes input frequency in dimensional units (time only): for e.g. `freq=1/10, cycles_per="D"` for filtering at the 10-day period.

Another experimental aspect is that it attempts to filter along chunked dimensions using map_overlap. Because filters are "local" --- influence of a point is restricted to the impulse response length (approximately!) --- chunks that are large enough can be filtered independently of other chunks. `xfilter` uses dask's `map_overlap` to apply a filter such that edge effects are taken care of. This is currently working approximately.

This software is alpha quality.
