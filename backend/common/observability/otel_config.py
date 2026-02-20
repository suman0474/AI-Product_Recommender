"""
OpenTelemetry Configuration for Unified Tracing

Provides centralized observability configuration for distributed tracing
across the application. Integrates with Azure Monitor, Jaeger, or any
OTLP-compatible backend.

Environment Variables:
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP exporter endpoint (e.g., http://localhost:4317)
    OTEL_SERVICE_NAME: Service name for traces (default: aipr-backend)
    APPLICATIONINSIGHTS_CONNECTION_STRING: Azure Monitor connection string
"""
import os
import logging
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Track initialization state
_tracer = None
_otel_initialized = False


def configure_opentelemetry() -> Optional[object]:
    """
    Configure OpenTelemetry with OTLP exporter.
    
    Supports:
    - OTLP gRPC exporter (Jaeger, Zipkin, etc.)
    - Azure Monitor exporter (if connection string provided)
    
    Returns:
        Tracer instance if configured, None otherwise
    """
    global _tracer, _otel_initialized
    
    if _otel_initialized:
        return _tracer
    
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    azure_connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    service_name = os.getenv("OTEL_SERVICE_NAME", "aipr-backend")
    
    if not endpoint and not azure_connection_string:
        logger.info("[OTEL] No endpoint configured, tracing disabled")
        _otel_initialized = True
        return None
    
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME
        
        # Create resource with service name
        resource = Resource.create({SERVICE_NAME: service_name})
        provider = TracerProvider(resource=resource)
        
        # Configure exporter based on available configuration
        if azure_connection_string:
            try:
                from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
                exporter = AzureMonitorTraceExporter(connection_string=azure_connection_string)
                logger.info("[OTEL] Using Azure Monitor exporter")
            except ImportError:
                logger.warning("[OTEL] azure-monitor-opentelemetry-exporter not installed")
                exporter = None
        elif endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                exporter = OTLPSpanExporter(endpoint=endpoint)
                logger.info(f"[OTEL] Using OTLP exporter: {endpoint}")
            except ImportError:
                logger.warning("[OTEL] opentelemetry-exporter-otlp not installed")
                exporter = None
        else:
            exporter = None
        
        if exporter:
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)
            trace.set_tracer_provider(provider)
            _tracer = trace.get_tracer(service_name)
            logger.info(f"[OTEL] ✅ OpenTelemetry configured for service: {service_name}")
        else:
            logger.warning("[OTEL] No exporter available, tracing disabled")
        
        _otel_initialized = True
        return _tracer
        
    except ImportError as e:
        logger.info(f"[OTEL] OpenTelemetry not installed: {e}")
        _otel_initialized = True
        return None
    except Exception as e:
        logger.error(f"[OTEL] Configuration failed: {e}")
        _otel_initialized = True
        return None


def get_tracer(name: str = None):
    """
    Get or create a tracer instance.
    
    Args:
        name: Tracer name (defaults to service name)
        
    Returns:
        Tracer instance or None if not configured
    """
    global _tracer
    
    if not _otel_initialized:
        configure_opentelemetry()
    
    if _tracer is None:
        return None
    
    if name:
        try:
            from opentelemetry import trace
            return trace.get_tracer(name)
        except ImportError:
            return None
    
    return _tracer


@contextmanager
def trace_span(name: str, attributes: dict = None):
    """
    Context manager for creating trace spans.
    
    Works as a no-op if OpenTelemetry is not configured.
    
    Args:
        name: Span name
        attributes: Optional span attributes
        
    Usage:
        with trace_span("process_request", {"user_id": "123"}):
            do_something()
    """
    tracer = get_tracer()
    
    if tracer is None:
        yield None
        return
    
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        yield span


def get_otel_status() -> dict:
    """
    Get OpenTelemetry configuration status.
    
    Returns:
        Dictionary with configuration status
    """
    return {
        "initialized": _otel_initialized,
        "tracer_available": _tracer is not None,
        "endpoint": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "not set"),
        "azure_monitor": bool(os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")),
        "service_name": os.getenv("OTEL_SERVICE_NAME", "aipr-backend")
    }
