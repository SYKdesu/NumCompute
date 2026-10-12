"""Small, dependency-free transformer/estimator pipeline."""

from __future__ import annotations

import inspect
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from typing import Any


def _call_with_optional_y(method: Any, X: Any, y: Any, params: dict[str, Any]) -> Any:
    """Pass y only when the bound method's signature accepts it."""

    if y is None:
        return method(X, **params)
    try:
        inspect.signature(method).bind(X, y, **params)
    except (TypeError, ValueError):
        return method(X, **params)
    return method(X, y, **params)


class Pipeline:
    """Chain transformers and finish with a transformer or estimator.

    Intermediate steps must implement ``fit`` and ``transform``. The final
    step must implement ``fit`` and may expose ``predict``, ``transform``,
    ``predict_proba`` or ``score``; matching methods are delegated by the
    pipeline after applying all earlier transforms.
    """

    def __init__(self, steps: Sequence[tuple[str, Any]]):
        self.steps = list(steps)
        self._validate_steps()
        self._is_fitted = False

    def _validate_steps(self) -> None:
        if not self.steps:
            raise ValueError("steps must contain at least one (name, component) pair")
        names: list[str] = []
        for item in self.steps:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("each pipeline step must be a (name, component) tuple")
            name, component = item
            if not isinstance(name, str) or not name or "__" in name:
                raise ValueError("step names must be non-empty strings without '__'")
            if name in names:
                raise ValueError(f"duplicate step name: {name!r}")
            if component is None:
                raise ValueError(f"step {name!r} must not be None")
            names.append(name)

        for name, transformer in self.steps[:-1]:
            if not callable(getattr(transformer, "fit", None)):
                raise TypeError(f"intermediate step {name!r} must implement fit")
            if not callable(getattr(transformer, "transform", None)):
                raise TypeError(f"intermediate step {name!r} must implement transform")
        last_name, final = self.steps[-1]
        if not callable(getattr(final, "fit", None)):
            raise TypeError(f"final step {last_name!r} must implement fit")

    @property
    def named_steps(self) -> OrderedDict[str, Any]:
        return OrderedDict(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, index: int | slice | str) -> Any:
        if isinstance(index, str):
            try:
                return self.named_steps[index]
            except KeyError as error:
                raise KeyError(f"unknown pipeline step {index!r}") from error
        if isinstance(index, slice):
            return Pipeline(self.steps[index])
        return self.steps[index][1]

    def _split_fit_params(self, fit_params: dict[str, Any]) -> dict[str, dict[str, Any]]:
        routed = {name: {} for name, _ in self.steps}
        for key, value in fit_params.items():
            if "__" not in key:
                raise ValueError(
                    f"fit parameter {key!r} must use the 'step__parameter' form"
                )
            name, parameter = key.split("__", 1)
            if name not in routed:
                raise ValueError(f"fit parameter targets unknown step {name!r}")
            if not parameter:
                raise ValueError(f"fit parameter {key!r} has an empty parameter name")
            routed[name][parameter] = value
        return routed

    def _fit_transformers(
        self,
        X: Any,
        y: Any,
        routed_params: dict[str, dict[str, Any]],
    ) -> Any:
        transformed = X
        for name, transformer in self.steps[:-1]:
            params = routed_params[name]
            fit_transform = getattr(transformer, "fit_transform", None)
            if callable(fit_transform):
                transformed = _call_with_optional_y(fit_transform, transformed, y, params)
            else:
                _call_with_optional_y(transformer.fit, transformed, y, params)
                transformed = transformer.transform(transformed)
        return transformed

    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> "Pipeline":
        """Fit each transformer in sequence, then fit the final component."""

        routed = self._split_fit_params(fit_params)
        transformed = self._fit_transformers(X, y, routed)
        final_name, final = self.steps[-1]
        _call_with_optional_y(final.fit, transformed, y, routed[final_name])
        self._is_fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Pipeline is not fitted; call fit before inference")

    def _transform_before_final(self, X: Any) -> Any:
        self._require_fitted()
        transformed = X
        for _, transformer in self.steps[:-1]:
            transformed = transformer.transform(transformed)
        return transformed

    def predict(self, X: Any) -> Any:
        transformed = self._transform_before_final(X)
        final_name, final = self.steps[-1]
        if not callable(getattr(final, "predict", None)):
            raise AttributeError(f"final step {final_name!r} does not implement predict")
        return final.predict(transformed)

    def predict_proba(self, X: Any) -> Any:
        transformed = self._transform_before_final(X)
        final_name, final = self.steps[-1]
        if not callable(getattr(final, "predict_proba", None)):
            raise AttributeError(f"final step {final_name!r} does not implement predict_proba")
        return final.predict_proba(transformed)

    def transform(self, X: Any) -> Any:
        self._require_fitted()
        transformed = X
        for name, component in self.steps:
            if not callable(getattr(component, "transform", None)):
                raise AttributeError(f"step {name!r} does not implement transform")
            transformed = component.transform(transformed)
        return transformed

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> Any:
        routed = self._split_fit_params(fit_params)
        transformed = self._fit_transformers(X, y, routed)
        final_name, final = self.steps[-1]
        params = routed[final_name]
        fit_transform = getattr(final, "fit_transform", None)
        if callable(fit_transform):
            result = _call_with_optional_y(fit_transform, transformed, y, params)
        else:
            _call_with_optional_y(final.fit, transformed, y, params)
            if not callable(getattr(final, "transform", None)):
                raise AttributeError(
                    f"final step {final_name!r} does not implement transform"
                )
            result = final.transform(transformed)
        self._is_fitted = True
        return result

    def score(self, X: Any, y: Any, **params: Any) -> float:
        transformed = self._transform_before_final(X)
        final_name, final = self.steps[-1]
        if not callable(getattr(final, "score", None)):
            raise AttributeError(f"final step {final_name!r} does not implement score")
        return float(final.score(transformed, y, **params))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params: dict[str, Any] = {"steps": self.steps}
        for name, component in self.steps:
            params[name] = component
            if deep:
                getter = getattr(component, "get_params", None)
                nested = (
                    getter(deep=True)
                    if callable(getter)
                    else dict(getattr(component, "__dict__", {}))
                )
                params.update({f"{name}__{key}": value for key, value in nested.items()})
        return params

    def set_params(self, **params: Any) -> "Pipeline":
        if not params:
            return self
        if "steps" in params:
            self.steps = list(params.pop("steps"))
            self._validate_steps()
        direct: dict[str, Any] = {}
        nested: dict[str, dict[str, Any]] = {}
        valid_names = {name for name, _ in self.steps}
        for key, value in params.items():
            if "__" in key:
                name, parameter = key.split("__", 1)
                if name not in valid_names:
                    raise ValueError(f"unknown pipeline step {name!r}")
                nested.setdefault(name, {})[parameter] = value
            elif key in valid_names:
                direct[key] = value
            else:
                raise ValueError(f"unknown pipeline parameter {key!r}")

        self.steps = [(name, direct.get(name, component)) for name, component in self.steps]
        for name, component in self.steps:
            if name not in nested:
                continue
            setter = getattr(component, "set_params", None)
            if callable(setter):
                setter(**nested[name])
            else:
                for parameter, value in nested[name].items():
                    if not hasattr(component, parameter):
                        raise ValueError(
                            f"step {name!r} has no parameter {parameter!r}"
                        )
                    setattr(component, parameter, value)
        self._validate_steps()
        self._is_fitted = False
        return self


def make_pipeline(*components: Any) -> Pipeline:
    """Construct a pipeline with deterministic names derived from class names."""

    if not components:
        raise ValueError("make_pipeline requires at least one component")
    counts: dict[str, int] = {}
    steps: list[tuple[str, Any]] = []
    for component in components:
        base = component.__class__.__name__.lower()
        counts[base] = counts.get(base, 0) + 1
        name = base if counts[base] == 1 else f"{base}{counts[base]}"
        steps.append((name, component))
    return Pipeline(steps)


__all__ = ["Pipeline", "make_pipeline"]
