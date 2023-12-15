    def applymap_index(
        self,
        func: Callable,
        axis: AxisInt | str = 0,
        level: Level | list[Level] | None = None,
        **kwargs,
    ) -> Styler:
        """
        Apply a CSS-styling function to the index or column headers, elementwise.

        .. deprecated:: 2.1.0

           Styler.applymap_index has been deprecated. Use Styler.map_index instead.

        Parameters
        ----------
        func : function
            ``func`` should take a scalar and return a string.
        axis : {{0, 1, "index", "columns"}}
            The headers over which to apply the function.
        level : int, str, list, optional
            If index is MultiIndex the level(s) over which to apply the function.
        **kwargs : dict
            Pass along to ``func``.

        Returns
        -------
        Styler
        """
        warnings.warn(
            "Styler.applymap_index has been deprecated. Use Styler.map_index instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self.map_index(func, axis, level, **kwargs)