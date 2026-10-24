import numpy as np

def _validate_2d_array(X):
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError(
            f"Expected a 2D array, got {X.ndim}D array instead."
        )

    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("Input array must not be empty.")

    return X
class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance."""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.n_features_in_ = None

    def fit(self, X):
        """Learn column means and population scales from a nonempty (n, p) array. Return self."""
        X = _validate_2d_array(X)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_ = np.where(
            self.scale_ == 0,
            1.0,
            self.scale_
        )#avoid division by zero
        self.n_features_in_ = X.shape[1]
        return self
    def transform(self, X):

        """Return a new (n, p) array using the fitted means and scales.

        Raises RuntimeError before fit and ValueError for a feature mismatch.
        Time and output storage are O(n * p)."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError(
            "StandardScaler must be fitted before calling transform()."
            )#not fitted
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, "
                f"but got {X.shape[1]}."
            )#shape mismatch
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        """Fit on X, then return its standardised values with the same shape."""
        return self.fit(X).transform(X)


class MinMaxScaler:

    """Scale each numeric feature to feature_range using fitted extrema.

    Inputs have shape (n_samples, n_features). Constant columns use a range
    of one. Fit uses O(n * p) time and O(p) fitted storage."""
    def __init__(self, feature_range=(0.0, 1.0)):
        self.feature_range = feature_range

        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
        self.n_features_in_ = None

    def fit(self, X):
        """Learn per-column minima and ranges from a nonempty 2-D array. Return self."""
        X = _validate_2d_array(X)

        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)

        self.data_range_ = self.data_max_ - self.data_min_

        self.data_range_ = np.where(
            self.data_range_ == 0,
            1.0,
            self.data_range_
        )
        self.n_features_in_ = X.shape[1]

        return self
    def transform(self, X):
        """Return an (n, p) array scaled with the fitted extrema.

        Raises RuntimeError before fit and ValueError for invalid dimensions
        or feature count. Time and output storage are O(n * p)."""
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError(
                "MinMaxScaler must be fitted before calling transform()."
            )

        X = _validate_2d_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, "
                f"but got {X.shape[1]}."
            )
        low, high = self.feature_range

        X_std = (X - self.data_min_) / self.data_range_

        return X_std * (high - low) + low

    def fit_transform(self, X):
            """Fit on X, then return its values mapped to feature_range."""
            return self.fit(X).transform(X)

class OneHotEncoder:

    """Encode fitted categories as integer indicator columns.

    Input shape is (n, p); output shape is (n, sum of category counts).
    handle_unknown selects an error or an all-zero block for unknown values."""
    def __init__(self, handle_unknown="error"):
        if handle_unknown not in ("error", "ignore"):
            raise ValueError(
                "handle_unknown must be 'error' or 'ignore'."
            )

        self.handle_unknown = handle_unknown
        self.categories_ = None
        self.n_features_in_ = None

    def fit(self, X):
        """Learn sorted categories from each column of a nonempty 2-D array. Return self."""
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(
                f"Expected a 2D array, got {X.ndim}D array instead."
            )

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("Input array must not be empty.")

        self.categories_ = [
            np.unique(X[:, j])
            for j in range(X.shape[1])
        ]

        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X):
        """Encode X using the fitted categories, without changing the fitted state.

        Raises RuntimeError before fit and ValueError for shape mismatches
        or unknown categories in error mode. For K total categories, time
        and output storage are O(n * K)."""
        if self.categories_ is None:
            raise RuntimeError(
                "OneHotEncoder must be fitted before calling transform()."
            )

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(
                f"Expected a 2D array, got {X.ndim}D array instead."
            )

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, "
                f"but got {X.shape[1]}."
            )

        encoded_columns = []

        for j, categories in enumerate(self.categories_):
            column = X[:, j]

            matches = (
                column[:, None] == categories[None, :]
            )

            if self.handle_unknown == "error":
                known = np.any(matches, axis=1)

                if not np.all(known):
                    unknown = np.unique(column[~known])
                    raise ValueError(
                        f"Unknown categories in feature {j}: {unknown}"
                    )

            encoded_columns.append(matches.astype(int))

        return np.concatenate(
            encoded_columns,
            axis=1
        )

    def fit_transform(self, X):
        """Learn categories from X and return its indicator matrix."""
        return self.fit(X).transform(X)

class SimpleImputer:

    """Replace NaNs in a 2-D numeric array with the constant fill_value.

    Fit records the feature count. Transform preserves shape and does not
    mutate the input; it takes O(n * p) time and output storage."""
    def __init__(self, fill_value=0.0):
        self.fill_value = fill_value
        self.n_features_in_ = None
        self.is_fitted_ = False

    def fit(self, X):
        """Record the feature count of a nonempty 2-D array and return self."""
        X = _validate_2d_array(X)

        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True

        return self

    def transform(self, X):
        """Return an (n, p) array with NaNs replaced by fill_value.

        Raises RuntimeError before fit and ValueError for invalid dimensions
        or a feature-count mismatch."""
        if not self.is_fitted_:
            raise RuntimeError(
                "SimpleImputer must be fitted before calling transform()."
            )

        X = _validate_2d_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, "
                f"but got {X.shape[1]}."
            )

        return np.where(
            np.isnan(X),
            self.fill_value,
            X
        )

    def fit_transform(self, X):
        """Fit the feature count and return X with NaNs replaced."""
        return self.fit(X).transform(X)


# Short public name used by the project specification.  Keep
# ``SimpleImputer`` for compatibility with Hongyu Liu's preprocessing module.
Imputer = SimpleImputer


__all__ = [
    "Imputer",
    "MinMaxScaler",
    "OneHotEncoder",
    "SimpleImputer",
    "StandardScaler",
]
