"""Tests for Kognita dataclasses: Node, Edge, SearchResult, EpisodeResult."""
import pytest

from kognita import Edge, EpisodeResult, Node, SearchResult


class TestNode:
    def test_basic_creation(self):
        node = Node(uuid="abc-123", name="Einstein")
        assert node.uuid == "abc-123"
        assert node.name == "Einstein"
        assert node.summary == ""
        assert node.labels == []

    def test_with_all_fields(self):
        node = Node(uuid="abc-123", name="Einstein", summary="Physicist", labels=["Person", "Scientist"])
        assert node.summary == "Physicist"
        assert "Person" in node.labels

    def test_is_frozen(self):
        node = Node(uuid="abc-123", name="Einstein")
        with pytest.raises((TypeError, AttributeError)):
            node.name = "Newton"  # type: ignore[misc]

    def test_from_graphiti(self):
        class FakeNode:
            uuid = "xyz-456"
            name = "Relativity"
            summary = "Theory of special relativity"
            labels = ["Concept"]

        node = Node.from_graphiti(FakeNode())
        assert node.uuid == "xyz-456"
        assert node.name == "Relativity"
        assert node.labels == ["Concept"]

    def test_from_graphiti_missing_attrs(self):
        class EmptyNode:
            pass

        node = Node.from_graphiti(EmptyNode())
        assert node.uuid == ""
        assert node.name == ""
        assert node.labels == []


class TestEdge:
    def test_basic_creation(self):
        edge = Edge(uuid="e-001", source_uuid="n-001", target_uuid="n-002")
        assert edge.uuid == "e-001"
        assert edge.source_uuid == "n-001"
        assert edge.fact == ""
        assert edge.name == ""

    def test_with_all_fields(self):
        edge = Edge(
            uuid="e-001",
            source_uuid="n-001",
            target_uuid="n-002",
            fact="Einstein published relativity in 1905",
            name="PUBLISHED",
        )
        assert edge.fact == "Einstein published relativity in 1905"
        assert edge.name == "PUBLISHED"

    def test_is_frozen(self):
        edge = Edge(uuid="e-001", source_uuid="n-001", target_uuid="n-002")
        with pytest.raises((TypeError, AttributeError)):
            edge.fact = "changed"  # type: ignore[misc]

    def test_from_graphiti(self):
        class FakeEdge:
            uuid = "e-999"
            source_node_uuid = "n-001"
            target_node_uuid = "n-002"
            fact = "discovered by"
            name = "DISCOVERED_BY"

        edge = Edge.from_graphiti(FakeEdge())
        assert edge.uuid == "e-999"
        assert edge.source_uuid == "n-001"
        assert edge.fact == "discovered by"


class TestSearchResult:
    def test_basic(self):
        result = SearchResult(fact="Einstein discovered relativity")
        assert result.fact == "Einstein discovered relativity"
        assert result.source_node is None
        assert result.target_node is None
        assert result.score is None

    def test_with_nodes(self):
        src = Node(uuid="n-001", name="Einstein")
        tgt = Node(uuid="n-002", name="Relativity")
        result = SearchResult(fact="discovered", source_node=src, target_node=tgt, score=0.95)
        assert result.source_node.name == "Einstein"
        assert result.score == 0.95


class TestEpisodeResult:
    def test_basic(self):
        ep = EpisodeResult(chunk_index=0, preview="Einstein published…", node_count=3, edge_count=2)
        assert ep.chunk_index == 0
        assert ep.node_count == 3
        assert ep.edge_count == 2
        assert ep.error is None

    def test_with_error(self):
        ep = EpisodeResult(chunk_index=5, preview="Failed chunk…", node_count=0, edge_count=0, error="API timeout")
        assert ep.error == "API timeout"
        assert ep.node_count == 0
