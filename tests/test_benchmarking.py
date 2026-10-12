import pytest

from numcompute.benchmarking import benchmark, benchmark_many, format_results


def test_benchmark_records_requested_calls_and_value():
    calls = []

    def function():
        calls.append(1)
        return 42

    result = benchmark(function, repeat=3, number=2, warmup=1)
    assert len(calls) == 7
    assert result.calls == 6
    assert result.value == 42
    assert len(result.durations) == 3


def test_benchmark_many_preserves_case_order():
    results = benchmark_many({"first": lambda: 1, "second": lambda: 2}, repeat=1, warmup=0)
    assert [result.name for result in results] == ["first", "second"]


def test_format_results_contains_headers_and_names():
    result = benchmark(lambda: None, name="no-op", repeat=1, warmup=0)
    table = format_results([result], unit="us")
    assert "name" in table
    assert "no-op" in table
    assert "us" in table


@pytest.mark.parametrize("option", [{"repeat": 0}, {"number": 0}, {"warmup": -1}])
def test_invalid_benchmark_counts_are_rejected(option):
    with pytest.raises(ValueError):
        benchmark(lambda: None, **option)
