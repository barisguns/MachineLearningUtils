
class ArraysTransformer:
    """
    A class to convert sampler output arrays to their original types.
    Modification of imblearn.utils._validation.ArrayTransformer to validate only X if y is None.
    """

    def __init__(self, X, y=None):
        self.x_props = self._gets_props(X)
        if y is not None:
            self.y_props = self._gets_props(y)

    def transform(self, X, y=None):
        X = self._transfrom_one(X, self.x_props)
        if y is not None:
            self.y_props = self._gets_props(y)
            y = self._transfrom_one(y, self.y_props)
            if self.x_props["type"].lower() == "dataframe" and self.y_props[
                "type"
            ].lower() in {"series", "dataframe"}:
                # We lost the y.index during resampling. We can safely use X.index to align
                # them.
                y.index = X.index
            return X, y
        return X

    def _gets_props(self, array):
        props = {}
        props["type"] = array.__class__.__name__
        props["columns"] = getattr(array, "columns", None)
        props["name"] = getattr(array, "name", None)
        props["dtypes"] = getattr(array, "dtypes", None)
        return props

    def _transfrom_one(self, array, props):
        type_ = props["type"].lower()
        if type_ == "list":
            ret = array.tolist()
        elif type_ == "dataframe":
            import pandas as pd

            ret = pd.DataFrame(array, columns=props["columns"])
            ret = ret.astype(props["dtypes"])
        elif type_ == "series":
            import pandas as pd

            ret = pd.Series(array, dtype=props["dtypes"], name=props["name"])
        else:
            ret = array
        return ret