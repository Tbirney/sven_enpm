    def _ensure_valid_index(self, value) -> None:
        """
        Ensure that if we don't have an index, that we can create one from the
        passed value.
        """
        # GH5632, make sure that we are a Series convertible
        if not len(self.index) and is_list_like(value) and len(value):
            if not isinstance(value, DataFrame):
                try:
                    value = Series(value)
                except (ValueError, NotImplementedError, TypeError) as err:
                    raise ValueError(
                        "Cannot set a frame with no defined index "
                        "and a value that cannot be converted to a Series"
                    ) from err

            # GH31368 preserve name of index
            index_copy = value.index.copy()
            if self.index.name is not None:
                index_copy.name = self.index.name

            self._mgr = self._mgr.reindex_axis(index_copy, axis=1, fill_value=np.nan)