"""OpenTelemetry export. Optional extra: ``kognita[otel]``.

The decision engine does not import this package. :func:`attach` refuses to
install a hook when the extra is missing or ``OTEL_EXPORTER_OTLP_ENDPOINT``
is unset. A hook that fails raises; :class:`~kognita.evidence.EvidenceWriter`
catches that, leaves the decision event as it was appended, and records the
failure.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Mapping

from kognita.evidence import EvidenceWriter
from kognita.exceptions import KognitaError

CITE_OTEL = "Flight recorder: OpenTelemetry export"


class OtelError(KognitaError):
    """OpenTelemetry export could not be installed or did not succeed."""


def attach(writer: EvidenceWriter) -> str:
    """Point ``writer.telemetry`` at an OTLP exporter.

    The endpoint is ``OTEL_EXPORTER_OTLP_ENDPOINT``. Without it the hook is
    not installed: a process that looks monitored and is not is refused.
    """
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        raise OtelError(
            "OpenTelemetry export has no OTEL_EXPORTER_OTLP_ENDPOINT "
            f"({CITE_OTEL})"
        )
    writer.telemetry = build_hook()
    return f"OpenTelemetry export attached ({CITE_OTEL})"


def build_hook() -> Callable[[Mapping[str, Any]], None]:
    """A hook that exports each snapshot and raises when the export fails."""
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
    except ImportError as exc:
        raise OtelError(
            f"OpenTelemetry is not installed (kognita[otel]) ({CITE_OTEL})"
        ) from exc

    class _Visible:
        """Remember an export that did not succeed. The processor swallows it."""

        def __init__(self, inner: Any) -> None:
            self.inner = inner
            self.error: str = ""

        def export(self, spans: Any) -> Any:
            self.error = ""
            try:
                result = self.inner.export(spans)
            except Exception as exc:
                self.error = str(exc)
                return SpanExportResult.FAILURE
            if result is not SpanExportResult.SUCCESS:
                self.error = "the exporter did not accept the span"
            return result

        def shutdown(self) -> None:
            self.inner.shutdown()

    exporter = _Visible(OTLPSpanExporter())
    provider = TracerProvider(resource=Resource.create({"service.name": "kognita"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("kognita")

    def hook(snapshot: Mapping[str, Any]) -> None:
        from kognita.recorder import telemetry_signals

        for signal in telemetry_signals(snapshot):
            span = tracer.start_span(f"kognita.{signal['name']}")
            try:
                span.set_attribute("kognita.measured", bool(signal["measured"]))
                span.set_attribute("kognita.citation", str(signal["citation"]))
                span.set_attribute("kognita.detail", str(signal["detail"]))
                if signal["measured"] and isinstance(signal["value"], (int, float)):
                    span.set_attribute("kognita.value", float(signal["value"]))
                attributes = signal.get("attributes") or {}
                if isinstance(attributes, dict):
                    for key, value in attributes.items():
                        if isinstance(value, (str, int, float, bool)):
                            span.set_attribute(f"kognita.{key}", value)
            finally:
                span.end()
            if exporter.error:
                raise OtelError(f"{exporter.error} ({CITE_OTEL})")

    return hook


__all__ = ["CITE_OTEL", "OtelError", "attach", "build_hook"]
