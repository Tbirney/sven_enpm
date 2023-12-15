    def test_autocorrelation_plot(self):
        from pandas.plotting import autocorrelation_plot

        ser = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )
        # Ensure no UserWarning when making plot
        with tm.assert_produces_warning(None):
            _check_plot_works(autocorrelation_plot, series=ser)
            _check_plot_works(autocorrelation_plot, series=ser.values)

            ax = autocorrelation_plot(ser, label="Test")
        _check_legend_labels(ax, labels=["Test"])