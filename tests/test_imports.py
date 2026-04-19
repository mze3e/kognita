"""Verify all public exports are importable."""
import kognita


def test_version():
    assert kognita.__version__ == "0.1.0"


def test_public_classes():
    assert hasattr(kognita, "Kognita")
    assert hasattr(kognita, "KognitaConfig")
    assert hasattr(kognita, "LLMConfig")
    assert hasattr(kognita, "EmbedderConfig")


def test_public_types():
    assert hasattr(kognita, "Node")
    assert hasattr(kognita, "Edge")
    assert hasattr(kognita, "SearchResult")
    assert hasattr(kognita, "EpisodeResult")
    assert hasattr(kognita, "GraphSnapshot")


def test_public_functions():
    assert callable(kognita.list_models)
    assert callable(kognita.execute_cypher)
    assert callable(kognita.load_snapshot)
    assert callable(kognita.save_snapshot)
    assert callable(kognita.content_hash)


def test_public_exceptions():
    assert issubclass(kognita.KognitaError, Exception)
    assert issubclass(kognita.ProviderError, kognita.KognitaError)
    assert issubclass(kognita.ConfigError, kognita.KognitaError)


def test_all_exports_present():
    for name in kognita.__all__:
        assert hasattr(kognita, name), f"Missing export: {name}"
