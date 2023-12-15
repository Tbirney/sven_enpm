    def test_is_valid_na_for_dtype_interval(self):
        dtype = IntervalDtype("int64", "left")
        assert not is_valid_na_for_dtype(NaT, dtype)

        dtype = IntervalDtype("datetime64[ns]", "both")
        assert not is_valid_na_for_dtype(NaT, dtype)