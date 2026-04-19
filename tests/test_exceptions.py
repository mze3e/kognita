"""Tests for the Kognita exception hierarchy."""
import pytest

from kognita import ConfigError, KognitaError, ProviderError
from kognita.exceptions import extract_api_error


def test_kognita_error_is_exception():
    assert issubclass(KognitaError, Exception)


def test_config_error_hierarchy():
    assert issubclass(ConfigError, KognitaError)
    err = ConfigError("bad config")
    assert str(err) == "bad config"


def test_provider_error_hierarchy():
    assert issubclass(ProviderError, KognitaError)


def test_provider_error_message():
    err = ProviderError("rate limited")
    assert str(err) == "rate limited"
    assert err.status_code is None


def test_provider_error_with_status_code():
    err = ProviderError("unauthorized", status_code=401)
    assert err.status_code == 401
    assert "unauthorized" in str(err)


def test_provider_error_catchable_as_kognita_error():
    with pytest.raises(KognitaError):
        raise ProviderError("api failure", status_code=500)


def test_extract_api_error_generic():
    exc = ValueError("something went wrong")
    result = extract_api_error(exc)
    assert "something went wrong" in result


def test_extract_api_error_with_status_code():
    class FakeAPIError(Exception):
        status_code = 429
        body = None
        message = "too many requests"

    result = extract_api_error(FakeAPIError())
    assert "429" in result
    assert "too many requests" in result


def test_extract_api_error_with_body():
    class FakeError(Exception):
        status_code = 400
        body = {"error": {"message": "invalid request"}}

    result = extract_api_error(FakeError())
    assert "400" in result
