def _get_data_and_dtype_name(data: ArrayLike):
    """
    Convert the passed data into a storable form and a dtype string.
    """
    if isinstance(data, Categorical):
        data = data.codes

    if isinstance(data.dtype, DatetimeTZDtype):
        # For datetime64tz we need to drop the TZ in tests TODO: why?
        dtype_name = f"datetime64[{data.dtype.unit}]"
    else:
        dtype_name = data.dtype.name

    if data.dtype.kind in "mM":
        data = np.asarray(data.view("i8"))
        # TODO: we used to reshape for the dt64tz case, but no longer
        #  doing that doesn't seem to break anything.  why?

    elif isinstance(data, PeriodIndex):
        data = data.asi8

    data = np.asarray(data)
    return data, dtype_name
