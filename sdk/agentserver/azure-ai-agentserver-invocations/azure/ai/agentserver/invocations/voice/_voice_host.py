# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Typed event relay over the existing Invocations WebSocket transport."""

from __future__ import annotations

import asyncio  # pylint: disable=do-not-import-asyncio
import contextvars
import inspect
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any, NoReturn, TypeVar, cast

from opentelemetry import (
    baggage as _otel_baggage,
    context as _otel_context,
    trace as _otel_trace,
)
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.propagators.textmap import Getter
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from starlette.routing import Host, Match, Mount, Router, WebSocketRoute
from starlette.types import Receive, Scope, Send
from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

from azure.ai.agentserver.core import (
    FoundryAgentRequestContext,
    experimental,
    get_request_context,
    set_request_context,
)
from azure.ai.agentserver.core._platform_headers import (  # pylint: disable=import-error,no-name-in-module
    FOUNDRY_CALL_ID,
    REQUEST_ID,
    SERVER_VERSION,
    USER_ID,
)

from .._constants import InvocationsWSConstants
from .._invocation import InvocationAgentServerHost
from . import _session as _session_transport
from ._codec import MAX_FRAME_BYTES, VoiceProtocolError, decode_inbound_message
from ._models import (
    BargeIn,
    InboundVoiceMessage,
    ResponseAccepted,
    ResponseCancelled,
    ResponseDropped,
    ResponseTimeout,
    SessionDisconnected,
    SessionEnd,
    SessionStart,
    UserMessage,
    UserNoInput,
    UserSpeechStarted,
)
from ._session import Session

SessionStartCallback = Callable[[Session, SessionStart], Awaitable[None]]
UserMessageCallback = Callable[[Session, UserMessage], Awaitable[None]]
UserNoInputCallback = Callable[[Session, UserNoInput], Awaitable[None]]
UserSpeechStartedCallback = Callable[[Session, UserSpeechStarted], Awaitable[None]]
BargeInCallback = Callable[[Session, BargeIn], Awaitable[None]]
ResponseAcceptedCallback = Callable[[Session, ResponseAccepted], Awaitable[None]]
ResponseDroppedCallback = Callable[[Session, ResponseDropped], Awaitable[None]]
ResponseCancelledCallback = Callable[[Session, ResponseCancelled], Awaitable[None]]
ResponseTimeoutCallback = Callable[[Session, ResponseTimeout], Awaitable[None]]
SessionEndCallback = Callable[[Session, SessionEnd], Awaitable[None]]
DisconnectCallback = Callable[[Session, SessionDisconnected], Awaitable[None]]
ConnectionTerminatingCallback = Callable[[Session], None]

_CallbackT = TypeVar("_CallbackT", bound=Callable[..., Awaitable[None]])
_AwaitedT = TypeVar("_AwaitedT")
_VoiceCallback = Callable[[Session, Any], Awaitable[None]]
logger = logging.getLogger("azure.ai.agentserver")
_TRACER = _otel_trace.get_tracer(__name__)
_VOICE_AUTHORITY_ROUTE = object()
_VOICE_CLOSE_CODE = _session_transport._VOICE_CLOSE_CODE_SCOPE_KEY  # pylint: disable=protected-access
_VOICE_DISCONNECT_EVENT = _session_transport._VOICE_DISCONNECT_EVENT_SCOPE_KEY  # pylint: disable=protected-access
_VOICE_TERMINATION_DEADLINE = "azure.ai.agentserver.invocations.voice.termination_deadline"
_VOICE_LOCAL_PROTOCOL_ERROR = "azure.ai.agentserver.invocations.voice.local_protocol_error"
_VOICE_TRACING_ENABLED = "azure.ai.agentserver.invocations.voice.tracing_enabled"
_VOICE_ROUTE_CONFLICT = "VoiceAgentServerHost cannot own /invocations_ws because the route is already registered"
_GEN_AI_SYSTEM = "azure.ai.agentserver"
_GEN_AI_PROVIDER = "AzureAI Hosted Agents"
_SESSION_BAGGAGE_KEY = "azure.ai.agentserver.session_id"
_VOICE_TRACE_PROPAGATOR = TraceContextTextMapPropagator()
_VOICE_BAGGAGE_PROPAGATOR = W3CBaggagePropagator()
_VOICE_ALLOWED_OPAQUE_BAGGAGE = frozenset(
    {
        _SESSION_BAGGAGE_KEY,
        "azure.ai.agentserver.conversation_id",
        "azure.ai.agentserver.invocation_id",
    }
)
_VOICE_LEAF_SPAN_BAGGAGE_KEY = "leaf_customer_span_id"
_MAX_CORRELATION_ID_BYTES = 256
_VALID_CORRELATION_ID = re.compile(r"^[A-Za-z0-9_.:-]+$")
_VALID_SIGNED_DECIMAL = re.compile(r"^-?[0-9]{1,19}$")


def _run_voice_observability(operation: Callable[[], _AwaitedT]) -> _AwaitedT:
    try:
        fallback_context = _otel_context.get_current()
    except BaseException:  # pylint: disable=broad-exception-caught
        fallback_context = None
    try:
        return contextvars.copy_context().run(operation)
    finally:
        _restore_voice_context(fallback_context)


def _log_tracing_failure(message: str) -> None:
    try:
        _run_voice_observability(lambda: logger.debug(message))
    except BaseException:  # pylint: disable=broad-exception-caught
        pass


def _start_voice_span(
    name: str,
    *,
    kind: _otel_trace.SpanKind,
    attributes: dict[str, Any],
) -> Any:
    try:
        return _run_voice_observability(lambda: _TRACER.start_span(name, kind=kind, attributes=attributes))
    except BaseException:  # pylint: disable=broad-exception-caught
        _log_tracing_failure("Voice span creation failed")
        return None


def _restore_voice_context(fallback_context: Any) -> None:
    if fallback_context is None:
        return
    try:
        if _otel_context.get_current() == fallback_context:
            return
    except BaseException:  # pylint: disable=broad-exception-caught
        _log_tracing_failure("Voice current context verification failed")
    try:
        _otel_context.attach(fallback_context)
    except BaseException:  # pylint: disable=broad-exception-caught
        _log_tracing_failure("Voice prior context restoration failed")


def _attach_voice_span(span: Any, fallback_context: Any) -> Any:
    if span is None:
        return None
    try:
        span_context = _otel_trace.set_span_in_context(span)
    except BaseException:  # pylint: disable=broad-exception-caught
        _log_tracing_failure("Voice span context construction failed")
        _restore_voice_context(fallback_context)
        return None
    return _attach_voice_context(
        span_context,
        fallback_context,
        failure_message="Voice span context attachment failed",
    )


def _attach_voice_context(
    context: Any,
    fallback_context: Any,
    *,
    failure_message: str = "Voice parent context attachment failed",
) -> Any:
    try:
        return _otel_context.attach(context)
    except BaseException:  # pylint: disable=broad-exception-caught
        _log_tracing_failure(failure_message)
        _restore_voice_context(fallback_context)
        return None


def _get_current_voice_context() -> Any:
    try:
        return _otel_context.get_current()
    except BaseException:  # pylint: disable=broad-exception-caught
        _log_tracing_failure("Voice current context lookup failed")
        return None


def _detach_voice_context(token: Any, fallback_context: Any) -> None:
    if token is not None:
        try:
            _otel_context.detach(token)
        except BaseException:  # pylint: disable=broad-exception-caught
            _log_tracing_failure("Voice span context detachment failed")
    _restore_voice_context(fallback_context)


def _end_voice_span(span: Any) -> None:
    if span is None:
        return
    try:
        _run_voice_observability(span.end)
    except BaseException:  # pylint: disable=broad-exception-caught
        _log_tracing_failure("Voice span completion failed")


def _set_voice_span_attributes(span: Any, attributes: dict[str, Any]) -> None:
    if span is None:
        return

    def set_attributes() -> None:
        for name, value in attributes.items():
            span.set_attribute(name, value)

    try:
        _run_voice_observability(set_attributes)
    except BaseException:  # pylint: disable=broad-exception-caught
        _log_tracing_failure("Voice span enrichment failed")


def _set_voice_span_error(span: Any, error_type: str, *, bridge_outcome: str | None = None) -> None:
    if span is None:
        return

    def set_error() -> None:
        if bridge_outcome is not None:
            span.set_attribute("bridge.outcome", bridge_outcome)
        span.set_attribute("error.type", error_type)
        span.set_status(_otel_trace.StatusCode.ERROR)

    try:
        _run_voice_observability(set_error)
    except BaseException:  # pylint: disable=broad-exception-caught
        _log_tracing_failure("Voice span error recording failed")


def _classify_voice_connection_outcome(
    *,
    close_code: int,
    error_code: str | None,
    span_error_code: str | None,
    local_protocol_error: bool,
    peer_disconnect: bool,
) -> tuple[str, bool]:
    selected_error_code = span_error_code or error_code
    if selected_error_code is not None:
        outcome = (
            selected_error_code
            if selected_error_code in {"accept_failed", "cancelled", "internal_error", "transport_error"}
            else "internal_error"
        )
        return outcome, True
    if local_protocol_error:
        return "protocol_error", True
    if peer_disconnect:
        return ("completed", False) if close_code == 1000 else ("transport_error", True)
    if close_code in {1000, 1001}:
        return "completed", False
    if close_code in {1002, 1003, 1007, 1008, 1009}:
        return "protocol_error", True
    return "transport_error", True


def _set_voice_cleanup_error(span: Any, *, primary_error: bool) -> None:
    _set_voice_span_attributes(
        span,
        {"azure.ai.agentserver.connection.cleanup_error": True},
    )
    if not primary_error:
        _set_voice_span_error(span, "cleanup_error")


def _validated_voice_correlation_id(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        return None
    if len(encoded) > _MAX_CORRELATION_ID_BYTES or _VALID_CORRELATION_ID.fullmatch(value) is None:
        return None
    return value


def _validated_voice_leaf_span_id(value: Any) -> str | None:
    if not isinstance(value, str) or _VALID_SIGNED_DECIMAL.fullmatch(value) is None:
        return None
    parsed = int(value)
    if parsed == 0 or parsed < -(2**63) or parsed > 2**63 - 1:
        return None
    return value


def _copy_approved_voice_baggage(source_context: Any, target_context: Any) -> Any:
    entries = _otel_baggage.get_all(context=source_context)
    for key in _VOICE_ALLOWED_OPAQUE_BAGGAGE:
        value = _validated_voice_correlation_id(entries.get(key))
        if value is not None:
            target_context = _otel_baggage.set_baggage(key, value, context=target_context)
    leaf_span_id = _validated_voice_leaf_span_id(entries.get(_VOICE_LEAF_SPAN_BAGGAGE_KEY))
    if leaf_span_id is not None:
        target_context = _otel_baggage.set_baggage(
            _VOICE_LEAF_SPAN_BAGGAGE_KEY,
            leaf_span_id,
            context=target_context,
        )
    return target_context


def _is_async_callable(callback: Callable[..., Any]) -> bool:
    for candidate in (callback, getattr(callback, "__call__", None)):
        if candidate is None:
            continue
        unwrapped = inspect.unwrap(candidate)
        if inspect.iscoroutinefunction(unwrapped) or inspect.isasyncgenfunction(unwrapped):
            return True
    return False


class _VoiceWebSocketRoute(WebSocketRoute):
    """Reserved Voice route that preserves matching Host and Mount authority."""

    def __init__(self, endpoint: Callable[..., Any], *, router: Router) -> None:
        super().__init__(InvocationsWSConstants.ROUTE_PATH, endpoint, name="invocations_ws")
        self._router = router

    def matches(self, scope: Scope) -> tuple[Match, Scope]:
        match, child_scope = super().matches(scope)
        if match is not Match.FULL:
            return match, child_scope
        for route in tuple(self._router.routes):
            if route is self or not isinstance(route, (Host, Mount)):
                continue
            authority_match, authority_scope = route.matches(scope)
            if authority_match is Match.FULL:
                selected_scope: dict[Any, Any] = dict(authority_scope)
                selected_scope[_VOICE_AUTHORITY_ROUTE] = route
                return Match.FULL, cast(Scope, selected_scope)
        return match, child_scope

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        authority_route = scope.pop(cast(Any, _VOICE_AUTHORITY_ROUTE), None)
        if isinstance(authority_route, (Host, Mount)):
            await authority_route.handle(scope, receive, send)
            return
        await super().handle(scope, receive, send)


class _VoiceHeaderGetter(Getter[list[tuple[bytes, bytes]]]):
    """Read raw Voice upgrade headers with W3C multi-value semantics."""

    def get(self, carrier: list[tuple[bytes, bytes]], key: str) -> list[str] | None:
        normalized_key = key.lower().encode("latin-1")
        values = [value.decode("latin-1") for name, value in carrier if name.lower() == normalized_key]
        if not values:
            return None
        if key.lower() == "traceparent":
            return values if len(values) == 1 else None
        if key.lower() in ("baggage", "tracestate"):
            return [",".join(values)]
        return values

    def keys(self, carrier: list[tuple[bytes, bytes]]) -> list[str]:
        return list(dict.fromkeys(name.decode("latin-1").lower() for name, _ in carrier))


_VOICE_HEADER_GETTER = _VoiceHeaderGetter()


def _extract_voice_websocket_context(websocket: WebSocket) -> Any:
    try:
        raw_headers: list[tuple[bytes, bytes]] = websocket.scope.get("headers", [])
        context = _VOICE_TRACE_PROPAGATOR.extract(carrier=raw_headers, getter=_VOICE_HEADER_GETTER)
        baggage_context = _VOICE_BAGGAGE_PROPAGATOR.extract(
            carrier=raw_headers,
            getter=_VOICE_HEADER_GETTER,
            context=context,
        )
        context = _copy_approved_voice_baggage(baggage_context, context)
        request_ids = _VOICE_HEADER_GETTER.get(raw_headers, REQUEST_ID) or []
        request_id = next(
            (validated for value in request_ids if (validated := _validated_voice_correlation_id(value)) is not None),
            None,
        )
        if request_id:
            context = _otel_baggage.set_baggage("x_request_id", request_id, context=context)
        return context
    except BaseException:  # pylint: disable=broad-exception-caught
        return None


def _selected_voice_close_code(websocket: WebSocket, default_code: int) -> int:
    scope = getattr(websocket, "scope", None)
    if not isinstance(scope, MutableMapping):
        return default_code
    selected = scope.get(_VOICE_CLOSE_CODE)
    return int(selected) if isinstance(selected, int) else default_code


def _select_voice_close_code(websocket: WebSocket, code: int) -> bool:
    scope = getattr(websocket, "scope", None)
    if not isinstance(scope, MutableMapping) or _VOICE_CLOSE_CODE in scope:
        return False
    scope[_VOICE_CLOSE_CODE] = code
    return True


def _raise_voice_disconnect(websocket: WebSocket, code: int, reason: str) -> NoReturn:
    scope = getattr(websocket, "scope", None)
    if _select_voice_close_code(websocket, code) and isinstance(scope, MutableMapping):
        scope[_VOICE_LOCAL_PROTOCOL_ERROR] = True
    raise WebSocketDisconnect(code=code, reason=reason)


def _raise_wrapped_cancellation(
    error: BaseException,
    cancellation_requests: int | None,
) -> None:
    _session_transport._raise_task_cancellation(error, cancellation_requests)  # pylint: disable=protected-access


def _task_cancellation_requests() -> int | None:
    return _session_transport._task_cancellation_requests()  # pylint: disable=protected-access


def _begin_voice_termination(websocket: WebSocket, session: Session) -> float:
    session._begin_termination()  # pylint: disable=protected-access
    loop = asyncio.get_running_loop()
    scope = getattr(websocket, "scope", None)
    if isinstance(scope, MutableMapping):
        selected = scope.get(_VOICE_TERMINATION_DEADLINE)
        if isinstance(selected, (int, float)):
            return float(selected)
        deadline = loop.time() + _session_transport.CLOSE_TIMEOUT_SECONDS
        scope[_VOICE_TERMINATION_DEADLINE] = deadline
        return deadline
    return loop.time() + _session_transport.CLOSE_TIMEOUT_SECONDS


def _selected_voice_termination_deadline(websocket: WebSocket) -> float:
    scope = getattr(websocket, "scope", None)
    if isinstance(scope, MutableMapping):
        selected = scope.get(_VOICE_TERMINATION_DEADLINE)
        if isinstance(selected, (int, float)):
            return float(selected)
    return asyncio.get_running_loop().time() + _session_transport.CLOSE_TIMEOUT_SECONDS


def _peek_voice_disconnect_event(websocket: WebSocket) -> SessionDisconnected | None:
    scope = getattr(websocket, "scope", None)
    if not isinstance(scope, MutableMapping):
        return None
    event = scope.get(_VOICE_DISCONNECT_EVENT)
    return event if isinstance(event, SessionDisconnected) else None


def _take_voice_disconnect_event(websocket: WebSocket) -> SessionDisconnected | None:
    scope = getattr(websocket, "scope", None)
    if not isinstance(scope, MutableMapping):
        return None
    event = scope.pop(_VOICE_DISCONNECT_EVENT, None)
    return event if isinstance(event, SessionDisconnected) else None


async def _raise_pending_cancellation() -> None:
    await asyncio.sleep(0)


async def _raise_pending_or_consumed_cancellation(cancellation_requests: int | None) -> None:
    await _raise_pending_cancellation()
    current_requests = _task_cancellation_requests()
    if cancellation_requests is not None and current_requests is not None and current_requests > cancellation_requests:
        raise asyncio.CancelledError()


async def _await_with_cancellation_guard(
    awaitable: Awaitable[_AwaitedT],
    *,
    on_success: Callable[[], object] | None = None,
) -> _AwaitedT:
    cancellation_requests = _task_cancellation_requests()
    result = await awaitable
    if on_success is not None:
        on_success()
    await _raise_pending_or_consumed_cancellation(cancellation_requests)
    return result


async def _receive_voice_transport_message(websocket: WebSocket) -> MutableMapping[str, Any]:
    message = await _session_transport._run_transport_operation(websocket.receive())  # pylint: disable=protected-access
    await _raise_pending_cancellation()
    return message


@experimental
class VoiceAgentServerHost(InvocationAgentServerHost):
    """Invocations host with typed Voice event decorators.

    The host performs only per-frame decoding and static callback dispatch.
    Agent code owns IDs, application tasks, response lifecycle, terminal-event
    correlation, cancellation, history, and reconnect restoration.

    :param openapi_spec: Optional OpenAPI document inherited from Invocations.
    :param asyncapi_spec_json: Optional AsyncAPI JSON document.
    :param asyncapi_spec_yaml: Optional AsyncAPI YAML document.
    :param kwargs: Remaining :class:`InvocationAgentServerHost` options.
    """

    def __init__(
        self,
        *,
        openapi_spec: dict[str, Any] | None = None,
        asyncapi_spec_json: dict[str, Any] | None = None,
        asyncapi_spec_yaml: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._voice_callbacks: dict[str, _VoiceCallback] = {}
        self._connection_terminating_callback: ConnectionTerminatingCallback | None = None
        self._voice_route: _VoiceWebSocketRoute | None = None
        super().__init__(
            openapi_spec=openapi_spec,
            asyncapi_spec_json=asyncapi_spec_json,
            asyncapi_spec_yaml=asyncapi_spec_yaml,
            **kwargs,
        )
        super().ws_handler(self._handle_voice_connection)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":
            scope = dict(scope)
            self._ensure_ws_route_registered()
        await super().__call__(scope, receive, send)

    def _ensure_ws_route_registered(self) -> None:
        voice_route = self._voice_route
        if voice_route is None:
            voice_route = _VoiceWebSocketRoute(self._ws_endpoint, router=self.router)
            self._voice_route = voice_route
        routes = self.router.routes
        for route in routes:
            if (
                route is not voice_route
                and isinstance(route, WebSocketRoute)
                and getattr(route, "path", None) == InvocationsWSConstants.ROUTE_PATH
            ):
                raise RuntimeError(_VOICE_ROUTE_CONFLICT)
        if voice_route in routes:
            routes.remove(voice_route)
        routes.insert(0, voice_route)

    async def _ws_endpoint(self, websocket: WebSocket) -> None:
        session_id = _validated_voice_correlation_id(self.config.session_id) or str(uuid.uuid4())
        start_ns = time.monotonic_ns()
        calling_context = _get_current_voice_context()
        connection_parent_context = calling_context
        trace_token = None
        connection_span = None
        connection_span_token = None
        tracing_enabled = False
        scope = getattr(websocket, "scope", None)
        try:
            if calling_context is not None:
                extracted_context = _extract_voice_websocket_context(websocket)
                if extracted_context is not None:
                    try:
                        extracted_context = _otel_baggage.set_baggage(
                            _SESSION_BAGGAGE_KEY,
                            session_id,
                            context=extracted_context,
                        )
                        trace_token = _attach_voice_context(extracted_context, calling_context)
                    except BaseException:  # pylint: disable=broad-exception-caught
                        trace_token = None
            if trace_token is not None:
                connection_parent_context = _get_current_voice_context()
            if connection_parent_context is not None and trace_token is not None:
                connection_attributes = self._voice_span_attributes(session_id=session_id)
                connection_attributes["network.protocol.name"] = "websocket"
                connection_span = _start_voice_span(
                    "agentserver.connection",
                    kind=_otel_trace.SpanKind.SERVER,
                    attributes=connection_attributes,
                )
                connection_span_token = _attach_voice_span(connection_span, connection_parent_context)
                tracing_enabled = connection_span is not None and connection_span_token is not None
            if isinstance(scope, MutableMapping):
                scope[_VOICE_TRACING_ENABLED] = tracing_enabled
            platform_token = set_request_context(
                FoundryAgentRequestContext(
                    call_id=websocket.headers.get(FOUNDRY_CALL_ID) or None,
                    user_id=websocket.headers.get(USER_ID) or None,
                    session_id=session_id,
                )
            )
            try:
                await self._run_voice_endpoint(websocket, session_id, start_ns, connection_span)
            finally:
                platform_token.var.reset(platform_token)
        finally:
            if isinstance(scope, MutableMapping):
                scope.pop(_VOICE_TRACING_ENABLED, None)
                scope.pop(_VOICE_LOCAL_PROTOCOL_ERROR, None)
            _detach_voice_context(connection_span_token, connection_parent_context)
            _end_voice_span(connection_span)
            _detach_voice_context(trace_token, calling_context)

    def _voice_span_attributes(self, *, session_id: str) -> dict[str, Any]:
        attributes: dict[str, Any] = {
            "service.name": "azure.ai.agentserver",
            "gen_ai.system": _GEN_AI_SYSTEM,
            "gen_ai.provider.name": _GEN_AI_PROVIDER,
            "microsoft.session.id": session_id,
        }
        if self.config.agent_name:
            attributes["gen_ai.agent.name"] = self.config.agent_name
        if self.config.agent_version:
            attributes["gen_ai.agent.version"] = self.config.agent_version
        if self.config.agent_id:
            attributes["gen_ai.agent.id"] = self.config.agent_id
        if self.config.project_id:
            attributes["microsoft.foundry.project.id"] = self.config.project_id
        return attributes

    def _turn_span_name(self) -> str:
        return f"invoke_agent {self.config.agent_id}" if self.config.agent_id else "invoke_agent"

    async def _invoke_turn_callback(
        self,
        callback: _VoiceCallback,
        session: Session,
        event: UserMessage | UserNoInput | ResponseAccepted,
    ) -> None:
        parent_context = _get_current_voice_context()
        if parent_context is None:
            await _await_with_cancellation_guard(callback(session, event))
            return
        attributes = self._voice_span_attributes(
            session_id=get_request_context().session_id or self.config.session_id or ""
        )
        attributes["gen_ai.operation.name"] = "invoke_agent"
        if isinstance(event, ResponseAccepted):
            attributes.update(
                {
                    "bridge.input.count": 0,
                    "turn.origin": "proactive",
                    "gen_ai.response.id": event.response_id,
                }
            )
        else:
            attributes.update(
                {
                    "bridge.input.count": 1,
                    "turn.origin": ("user" if isinstance(event, UserMessage) else "no_input"),
                }
            )
        span = _start_voice_span(
            self._turn_span_name(),
            kind=_otel_trace.SpanKind.INTERNAL,
            attributes=attributes,
        )
        token = _attach_voice_span(span, parent_context)
        if span is None or token is None:
            _end_voice_span(span)
            await _await_with_cancellation_guard(callback(session, event))
            return
        cancellation_requests = _task_cancellation_requests()
        try:
            await callback(session, event)
        except asyncio.CancelledError:
            _set_voice_span_error(span, "cancelled", bridge_outcome="cancelled")
            raise
        except BaseException as exc:  # pylint: disable=broad-exception-caught
            try:
                _raise_wrapped_cancellation(exc, cancellation_requests)
            except asyncio.CancelledError:
                _set_voice_span_error(span, "cancelled", bridge_outcome="cancelled")
                raise
            _set_voice_span_error(span, "callback_error", bridge_outcome="error")
            raise
        finally:
            _detach_voice_context(token, parent_context)
            _end_voice_span(span)
        await _raise_pending_or_consumed_cancellation(cancellation_requests)

    async def _run_voice_endpoint(
        self,
        websocket: WebSocket,
        session_id: str,
        start_ns: int,
        connection_span: Any,
    ) -> None:
        try:
            accept_error, voice_session, close_code, handler_exc, pending_error = (
                await self._run_voice_connection_context(
                    websocket,
                    session_id,
                )
            )
        except asyncio.CancelledError:
            self._emit_voice_close_event(
                session_id=session_id,
                start_ns=start_ns,
                close_code=InvocationsWSConstants.CLOSE_INTERNAL_ERROR,
                error_code="cancelled",
                span=connection_span,
            )
            raise

        if accept_error is not None:
            if voice_session is not None:
                await self._complete_voice_endpoint(
                    websocket=websocket,
                    voice_session=voice_session,
                    session_id=session_id,
                    start_ns=start_ns,
                    close_code=close_code,
                    handler_exc=None,
                    pending_error=None,
                    error_code_override="accept_failed",
                    connection_span=connection_span,
                )
            self._report_voice_accept_failure(
                session_id,
                start_ns,
                emit_event=voice_session is None,
                connection_span=connection_span,
            )
            return

        if voice_session is None:
            raise RuntimeError("Voice WebSocket accepted without a Session")
        await self._complete_voice_endpoint(
            websocket=websocket,
            voice_session=voice_session,
            session_id=session_id,
            start_ns=start_ns,
            close_code=close_code,
            handler_exc=handler_exc,
            pending_error=pending_error,
            connection_span=connection_span,
        )

    async def _run_voice_connection_context(
        self,
        websocket: WebSocket,
        session_id: str,
    ) -> tuple[Exception | None, Session | None, int, BaseException | None, BaseException | None]:
        accept_error: Exception | None = None
        voice_session: Session | None = None
        close_code = InvocationsWSConstants.CLOSE_NORMAL
        handler_exc: BaseException | None = None
        pending_error: BaseException | None = None
        try:
            accept_error = await self._accept_voice_websocket(websocket)
            if accept_error is None:
                voice_session, close_code, handler_exc, pending_error = await self._run_accepted_voice_handler(
                    websocket,
                    session_id,
                )
            elif websocket.application_state == WebSocketState.CONNECTED:
                voice_session = Session._create(websocket)  # pylint: disable=protected-access
                _begin_voice_termination(websocket, voice_session)
                close_code = InvocationsWSConstants.CLOSE_INTERNAL_ERROR
        finally:
            if voice_session is not None:
                Session._release(websocket, voice_session)  # pylint: disable=protected-access
        return accept_error, voice_session, close_code, handler_exc, pending_error

    async def _accept_voice_websocket(
        self,
        websocket: WebSocket,
    ) -> Exception | None:
        try:
            await _session_transport._run_transport_operation(  # pylint: disable=protected-access
                websocket.accept(
                    headers=[
                        (
                            SERVER_VERSION.encode("latin-1"),
                            self._build_server_version().encode("latin-1"),  # pylint: disable=protected-access
                        )
                    ]
                )
            )
            await _raise_pending_cancellation()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return exc
        return None

    async def _run_accepted_voice_handler(
        self,
        websocket: WebSocket,
        session_id: str,
    ) -> tuple[Session, int, BaseException | None, BaseException | None]:
        voice_session = Session._create(websocket)  # pylint: disable=protected-access
        close_code = InvocationsWSConstants.CLOSE_NORMAL
        handler_exc: BaseException | None = None
        pending_error: BaseException | None = None
        cancellation_requests = _task_cancellation_requests()
        try:
            close_code, handler_exc = await self._invoke_user_handler(websocket, session_id)
            await _raise_pending_cancellation()
            close_code = _selected_voice_close_code(websocket, close_code)
        except BaseException as exc:  # pylint: disable=broad-exception-caught
            if isinstance(exc, Exception):
                try:
                    _raise_wrapped_cancellation(exc, cancellation_requests)
                except asyncio.CancelledError as cancellation:
                    exc = cancellation
            close_code = _selected_voice_close_code(websocket, InvocationsWSConstants.CLOSE_INTERNAL_ERROR)
            handler_exc = exc
            pending_error = exc
        return voice_session, close_code, handler_exc, pending_error

    async def _complete_voice_endpoint(
        self,
        *,
        websocket: WebSocket,
        voice_session: Session,
        session_id: str,
        start_ns: int,
        close_code: int,
        handler_exc: BaseException | None,
        pending_error: BaseException | None,
        error_code_override: str | None = None,
        connection_span: Any = None,
    ) -> None:
        deadline = _selected_voice_termination_deadline(websocket)
        cleanup_cancelled = False
        disconnect_event = _peek_voice_disconnect_event(websocket)
        peer_disconnect = disconnect_event is not None
        scope = getattr(websocket, "scope", None)
        local_protocol_error = bool(isinstance(scope, MutableMapping) and scope.get(_VOICE_LOCAL_PROTOCOL_ERROR))
        close_attempt: asyncio.Task[None] | None = None
        close_error: Exception | None = None
        if error_code_override is not None:
            error_code = error_code_override
        elif handler_exc is None:
            error_code = None
        elif isinstance(handler_exc, asyncio.CancelledError):
            error_code = "cancelled"
        else:
            error_code = "internal_error"
        if (
            not isinstance(handler_exc, asyncio.CancelledError)
            and disconnect_event is None
            and close_code
            not in {
                1005,
                1006,
                1015,
            }
        ):
            reason = "Internal server error" if close_code == InvocationsWSConstants.CLOSE_INTERNAL_ERROR else ""
            try:
                close_attempt = voice_session._start_close(close_code, reason)  # pylint: disable=protected-access
            except Exception as exc:  # pylint: disable=broad-exception-caught
                close_error = exc

        termination_error = self._notify_connection_terminating(voice_session)
        disconnect_error: BaseException | None = None
        try:
            await _raise_pending_cancellation()
            if close_attempt is not None:
                try:
                    await voice_session._wait_close(close_attempt, deadline)  # pylint: disable=protected-access
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    close_error = exc
            disconnect_event = _take_voice_disconnect_event(websocket)
            peer_disconnect = peer_disconnect or disconnect_event is not None
            disconnect_error = await self._notify_peer_disconnect(
                voice_session,
                disconnect_event,
            )
            await _raise_pending_cancellation()
        except asyncio.CancelledError:
            cleanup_cancelled = True
            raise
        finally:
            span_error_code = (
                "transport_error"
                if error_code is None and close_error is not None and close_code in {1000, 1001}
                else None
            )
            primary_error = self._emit_voice_close_event(
                session_id=session_id,
                start_ns=start_ns,
                close_code=close_code,
                error_code=error_code,
                span_error_code=span_error_code,
                span=connection_span,
                local_protocol_error=local_protocol_error,
                peer_disconnect=peer_disconnect,
                cleanup_error=termination_error is not None or disconnect_error is not None,
                cleanup_cancelled=cleanup_cancelled,
            )
            if termination_error is not None or disconnect_error is not None or cleanup_cancelled:
                _set_voice_cleanup_error(connection_span, primary_error=primary_error)

        if pending_error is not None:
            raise pending_error
        self._report_voice_endpoint_errors(
            session_id=session_id,
            handler_error=handler_exc if isinstance(handler_exc, Exception) else None,
            termination_error=termination_error,
            disconnect_error=disconnect_error,
            close_error=close_error,
        )

    @staticmethod
    def _report_voice_endpoint_errors(
        *,
        session_id: str,
        handler_error: Exception | None,
        termination_error: BaseException | None,
        disconnect_error: BaseException | None,
        close_error: Exception | None,
    ) -> None:
        if handler_error is not None:
            try:
                _run_voice_observability(
                    lambda: logger.error("Voice WebSocket handler raised for session %s", session_id)
                )
            except BaseException:  # pylint: disable=broad-exception-caught
                pass
        if termination_error is not None:
            try:
                _run_voice_observability(lambda: logger.error("Voice connection termination callback failed"))
            except BaseException:  # pylint: disable=broad-exception-caught
                pass
        if disconnect_error is not None:
            try:
                _run_voice_observability(lambda: logger.error("Voice disconnect callback failed"))
            except BaseException:  # pylint: disable=broad-exception-caught
                pass
        if close_error is not None:
            try:
                _run_voice_observability(lambda: logger.debug("Error closing Voice WebSocket session %s", session_id))
            except BaseException:  # pylint: disable=broad-exception-caught
                pass

    def _report_voice_accept_failure(
        self,
        session_id: str,
        start_ns: int,
        *,
        emit_event: bool,
        connection_span: Any,
    ) -> None:
        if emit_event:
            self._emit_voice_close_event(
                session_id=session_id,
                start_ns=start_ns,
                close_code=InvocationsWSConstants.CLOSE_INTERNAL_ERROR,
                error_code="accept_failed",
                span=connection_span,
            )
        try:
            _run_voice_observability(lambda: logger.error("Voice WebSocket accept failed for session %s", session_id))
        except BaseException:  # pylint: disable=broad-exception-caught
            pass

    async def _invoke_user_handler(
        self,
        websocket: WebSocket,
        session_id: str,
    ) -> tuple[int, BaseException | None]:
        ws_fn = self._ws_fn
        if ws_fn is None:
            raise RuntimeError("_invoke_user_handler called with no registered ws_handler")
        cancellation_requests = _task_cancellation_requests()
        try:
            await ws_fn(websocket)
            return InvocationsWSConstants.CLOSE_NORMAL, None
        except WebSocketDisconnect as exc:
            _raise_wrapped_cancellation(exc, cancellation_requests)
            return int(exc.code) if exc.code else InvocationsWSConstants.CLOSE_NORMAL, None
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _raise_wrapped_cancellation(exc, cancellation_requests)
            return InvocationsWSConstants.CLOSE_INTERNAL_ERROR, exc

    def _emit_voice_close_event(
        self,
        *,
        session_id: str,
        start_ns: int,
        close_code: int,
        error_code: str | None,
        span_error_code: str | None = None,
        span: Any = None,
        local_protocol_error: bool = False,
        peer_disconnect: bool = False,
        cleanup_error: bool = False,
        cleanup_cancelled: bool = False,
    ) -> bool:
        duration_ms = (time.monotonic_ns() - start_ns) // 1_000_000
        outcome, is_error = _classify_voice_connection_outcome(
            close_code=close_code,
            error_code=error_code,
            span_error_code=span_error_code,
            local_protocol_error=local_protocol_error,
            peer_disconnect=peer_disconnect,
        )
        _set_voice_span_attributes(span, {"azure.ai.agentserver.connection.outcome": outcome})
        if is_error:
            _set_voice_span_error(span, outcome)
        try:
            # The close code remains the first terminal. The existing structured
            # error tag independently reports a cleanup callback failure, while
            # the connection span retains the primary terminal outcome above.
            diagnostic_error_code = (
                "internal_error" if cleanup_error else "cancelled" if cleanup_cancelled else error_code
            )
            _run_voice_observability(
                lambda: self._emit_close_event(
                    session_id,
                    close_code,
                    duration_ms,
                    error_code=diagnostic_error_code,
                )
            )
        except BaseException:  # pylint: disable=broad-exception-caught
            pass
        return is_error

    def ws_handler(self, fn: Any) -> NoReturn:
        """Reject raw-handler registration on the typed Voice host.

        :param fn: Raw handler supplied by the caller.
        :type fn: Any
        :raises RuntimeError: Always; the typed host owns ``/invocations_ws``.
        """
        del fn
        raise RuntimeError("VoiceAgentServerHost owns /invocations_ws; use on_<event> decorators")

    def on_session_start(self, callback: SessionStartCallback) -> SessionStartCallback:
        """Register the ``session.start`` callback.

        :param callback: Async callback receiving the thin Session and event.
        :type callback: SessionStartCallback
        :return: The callback, unchanged.
        :rtype: SessionStartCallback
        """
        return self._register_voice_callback(SessionStart.type, callback)

    def on_user_message(self, callback: UserMessageCallback) -> UserMessageCallback:
        """Register the ``user.message`` callback.

        :param callback: Async callback receiving the thin Session and event.
        :type callback: UserMessageCallback
        :return: The callback, unchanged.
        :rtype: UserMessageCallback
        """
        return self._register_voice_callback(UserMessage.type, callback)

    def on_user_no_input(self, callback: UserNoInputCallback) -> UserNoInputCallback:
        """Register the ``user.no_input`` callback.

        :param callback: Async callback receiving the thin Session and event.
        :type callback: UserNoInputCallback
        :return: The callback, unchanged.
        :rtype: UserNoInputCallback
        """
        return self._register_voice_callback(UserNoInput.type, callback)

    def on_user_speech_started(self, callback: UserSpeechStartedCallback) -> UserSpeechStartedCallback:
        """Register the ``user.speech_started`` callback.

        :param callback: Async callback receiving the thin Session and event.
        :type callback: UserSpeechStartedCallback
        :return: The callback, unchanged.
        :rtype: UserSpeechStartedCallback
        """
        return self._register_voice_callback(UserSpeechStarted.type, callback)

    def on_barge_in(self, callback: BargeInCallback) -> BargeInCallback:
        """Register the ``barge_in`` callback.

        :param callback: Async callback receiving the thin Session and event.
        :type callback: BargeInCallback
        :return: The callback, unchanged.
        :rtype: BargeInCallback
        """
        return self._register_voice_callback(BargeIn.type, callback)

    def on_response_accepted(self, callback: ResponseAcceptedCallback) -> ResponseAcceptedCallback:
        """Register the ``response.accepted`` callback.

        :param callback: Async callback receiving the thin Session and event.
        :type callback: ResponseAcceptedCallback
        :return: The callback, unchanged.
        :rtype: ResponseAcceptedCallback
        """
        return self._register_voice_callback(ResponseAccepted.type, callback)

    def on_response_dropped(self, callback: ResponseDroppedCallback) -> ResponseDroppedCallback:
        """Register the ``response.dropped`` callback.

        :param callback: Async callback receiving the thin Session and event.
        :type callback: ResponseDroppedCallback
        :return: The callback, unchanged.
        :rtype: ResponseDroppedCallback
        """
        return self._register_voice_callback(ResponseDropped.type, callback)

    def on_response_cancelled(self, callback: ResponseCancelledCallback) -> ResponseCancelledCallback:
        """Register the ``response.cancelled`` callback.

        :param callback: Async callback receiving the thin Session and event.
        :type callback: ResponseCancelledCallback
        :return: The callback, unchanged.
        :rtype: ResponseCancelledCallback
        """
        return self._register_voice_callback(ResponseCancelled.type, callback)

    def on_response_timeout(self, callback: ResponseTimeoutCallback) -> ResponseTimeoutCallback:
        """Register the ``response.timeout`` callback.

        :param callback: Async callback receiving the thin Session and event.
        :type callback: ResponseTimeoutCallback
        :return: The callback, unchanged.
        :rtype: ResponseTimeoutCallback
        """
        return self._register_voice_callback(ResponseTimeout.type, callback)

    def on_session_end(self, callback: SessionEndCallback) -> SessionEndCallback:
        """Register the ``session.end`` callback.

        :param callback: Async callback receiving the thin Session and event.
        :type callback: SessionEndCallback
        :return: The callback, unchanged.
        :rtype: SessionEndCallback
        """
        return self._register_voice_callback(SessionEnd.type, callback)

    def on_disconnect(self, callback: DisconnectCallback) -> DisconnectCallback:
        """Register the peer transport-disconnect callback.

        :param callback: Async callback receiving the thin Session and transport event.
        :type callback: DisconnectCallback
        :return: The callback, unchanged.
        :rtype: DisconnectCallback
        """
        return self._register_voice_callback("disconnect", callback)

    def on_connection_terminating(self, callback: ConnectionTerminatingCallback) -> ConnectionTerminatingCallback:
        """Register a synchronous signal that the connection handler is exiting.

        The host invokes the callback once whenever the connection handler
        unwinds in process. The callback must return promptly, be idempotent,
        and must not send frames. Applications can use it to synchronously
        cancel their connection-owned tasks or set their own stop signals. The
        SDK does not wait for those tasks to finish.

        :param callback: Synchronous callback receiving the thin Session.
        :type callback: ConnectionTerminatingCallback
        :return: The callback, unchanged.
        :rtype: ConnectionTerminatingCallback
        :raises TypeError: If the callback is async or cannot accept Session.
        :raises RuntimeError: If a callback is already registered.
        """
        try:
            is_async = _is_async_callable(callback)
        except ValueError as exc:
            raise TypeError("on_connection_terminating expects a synchronous function") from exc
        if is_async:
            raise TypeError("on_connection_terminating expects a synchronous function")
        try:
            inspect.signature(callback).bind(None)
        except TypeError as exc:
            raise TypeError("Connection terminating callback must accept Session") from exc
        if self._connection_terminating_callback is not None:
            raise RuntimeError("A callback is already registered for connection termination")
        self._connection_terminating_callback = callback
        return callback

    def _register_voice_callback(self, message_type: str, callback: _CallbackT) -> _CallbackT:
        if not inspect.iscoroutinefunction(callback):
            raise TypeError(f"on_{message_type.replace('.', '_')} expects an async function")
        try:
            inspect.signature(callback).bind(None, None)
        except TypeError as exc:
            raise TypeError("Voice callbacks must accept Session and event positional arguments") from exc
        if message_type in self._voice_callbacks:
            raise RuntimeError(f"A callback is already registered for {message_type}")
        self._voice_callbacks[message_type] = cast(_VoiceCallback, callback)
        return cast(_CallbackT, callback)

    def _notify_connection_terminating(self, session: Session) -> BaseException | None:
        terminating_callback = self._connection_terminating_callback
        if terminating_callback is None:
            return None
        try:
            result = terminating_callback(session)
            if result is not None:
                raise TypeError("Connection terminating callback must return None")
        except BaseException as exc:  # pylint: disable=broad-exception-caught
            return exc
        return None

    async def _notify_peer_disconnect(
        self,
        session: Session,
        event: SessionDisconnected | None,
    ) -> BaseException | None:
        callback = self._voice_callbacks.get("disconnect")
        if callback is None or event is None:
            return None
        cancellation_requests = _task_cancellation_requests()
        try:
            await callback(session, event)
        except asyncio.CancelledError as exc:
            current_requests = _task_cancellation_requests()
            if (
                cancellation_requests is not None
                and current_requests is not None
                and current_requests > cancellation_requests
            ):
                raise
            return exc
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _raise_wrapped_cancellation(exc, cancellation_requests)
            return exc
        except BaseException as exc:  # pylint: disable=broad-exception-caught
            _raise_wrapped_cancellation(exc, cancellation_requests)
            return exc
        await _raise_pending_or_consumed_cancellation(cancellation_requests)
        return None

    async def _dispatch_voice_callback(
        self,
        websocket: WebSocket,
        session: Session,
        event: InboundVoiceMessage,
        callback: _VoiceCallback,
    ) -> None:
        if isinstance(event, (UserMessage, UserNoInput, ResponseAccepted)):
            scope = getattr(websocket, "scope", None)
            if isinstance(scope, MutableMapping) and scope.get(_VOICE_TRACING_ENABLED) is True:
                await self._invoke_turn_callback(callback, session, event)
            else:
                await _await_with_cancellation_guard(callback(session, event))
            return
        await _await_with_cancellation_guard(
            callback(session, event),
            on_success=(
                (lambda: _begin_voice_termination(websocket, session)) if isinstance(event, SessionEnd) else None
            ),
        )

    async def _handle_voice_connection(self, websocket: WebSocket) -> None:
        bound_session = Session._current(websocket)  # pylint: disable=protected-access
        session = bound_session or Session._create(websocket)  # pylint: disable=protected-access
        try:
            while True:
                raw_message = await _receive_voice_transport_message(websocket)
                raw_type = raw_message.get("type")
                if raw_type == "websocket.disconnect":
                    code = int(raw_message.get("code") or 1000)
                    raw_reason = raw_message.get("reason")
                    reason = raw_reason if isinstance(raw_reason, str) else None
                    _select_voice_close_code(websocket, code)
                    _begin_voice_termination(websocket, session)
                    websocket.scope.setdefault(
                        _VOICE_DISCONNECT_EVENT,
                        SessionDisconnected(code=code, reason=reason),
                    )
                    raise WebSocketDisconnect(
                        code=code,
                        reason=reason,
                    )
                if raw_type != "websocket.receive":
                    reason = "Invalid Voice WebSocket event"
                    _raise_voice_disconnect(websocket, 1002, reason)
                frame = raw_message.get("text")
                if frame is None:
                    reason = "Voice messages must be text frames"
                    _raise_voice_disconnect(websocket, 1003, reason)
                try:
                    event = decode_inbound_message(frame)
                except VoiceProtocolError as exc:
                    reason = "Invalid Voice message"
                    try:
                        _raise_voice_disconnect(websocket, exc.close_code, reason)
                    except WebSocketDisconnect as disconnect:
                        raise disconnect from exc
                if event is None:
                    continue
                callback = self._voice_callbacks.get(event.type)
                if callback is not None:
                    await self._dispatch_voice_callback(
                        websocket,
                        session,
                        cast(InboundVoiceMessage, event),
                        callback,
                    )
                if isinstance(event, SessionEnd):
                    return
        finally:
            _begin_voice_termination(websocket, session)
            if bound_session is None:
                Session._release(websocket, session)  # pylint: disable=protected-access
                termination_error = self._notify_connection_terminating(session)
                disconnect_error = await self._notify_peer_disconnect(
                    session,
                    _take_voice_disconnect_event(websocket),
                )
                if termination_error is not None:
                    try:
                        logger.error("Voice connection termination callback failed")
                    except BaseException:  # pylint: disable=broad-exception-caught
                        pass
                if disconnect_error is not None:
                    try:
                        logger.error("Voice disconnect callback failed")
                    except BaseException:  # pylint: disable=broad-exception-caught
                        pass

    def _build_hypercorn_config(self, host: str, port: int) -> object:
        """Create a Hypercorn config with the Voice frame admission limit.

        :param host: Network interface to bind.
        :type host: str
        :param port: Port to bind.
        :type port: int
        :return: Configured Hypercorn config.
        :rtype: hypercorn.config.Config
        """
        config = super()._build_hypercorn_config(host, port)
        setattr(config, "websocket_max_message_size", MAX_FRAME_BYTES)
        return config
