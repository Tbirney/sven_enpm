def test_concat_empty_arrow_backed_series(dtype):
    # GH#51734
    ser = pd.Series([], dtype=dtype)
    expected = ser.copy()
    result = pd.concat([ser[np.array([], dtype=np.bool_)]])
    tm.assert_series_equal(result, expected)