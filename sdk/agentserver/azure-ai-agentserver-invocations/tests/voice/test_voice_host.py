# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Tracing tests for the typed Voice relay ownership boundary."""

import asyncio
import json
import logging

import pytest
import opentelemetry.propagate as otel_propagate
from opentelemetry import baggage as otel_baggage, context as otel_context, trace
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.propagate import inject
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags, TraceState
from starlette.websockets import WebSocket, WebSocketDisconnect

from azure.ai.agentserver.core import get_request_context
from azure.ai.agentserver.core._tracing import _BaggageLogRecordProcessor
from azure.ai.agentserver.invocations.voice import (
    InputTextPart,
    ResponseAccepted,
    ResponseDropped,
    SessionEnd,
    SessionReady,
    UserMessage,
    UserNoInput,
    UserSpeechStarted,
    VoiceAgentServerHost,
)
from azure.ai.agentserver.invocations.voice import _voice_host as voice_host_module


def _frame(message_type, message_id, **fields):
    return {
        "type": message_type,
        "id": message_id,
        "ts": "2026-08-14T00:00:00Z",
        **fields,
    }


class _VoiceSocket:
    def __init__(
        self,
        headers=None,
        *,
        fail_accept=False,
        fail_close=False,
        block_close=False,
        peer_send_close_code=None,
    ):
        self._incoming = asyncio.Queue()
        self._incoming.put_nowait({"type": "websocket.connect"})
        self._fail_accept = fail_accept
        self._fail_close = fail_close
        self._block_close = block_close
        self._peer_send_close_code = peer_send_close_code
        self.close_started = asyncio.Event()
        self.release_close = asyncio.Event()
        self.sent = []
        self.websocket = WebSocket(
            {
                "type": "websocket",
                "asgi": {"version": "3.0", "spec_version": "2.4"},
                "scheme": "ws",
                "path": "/invocations_ws",
                "raw_path": b"/invocations_ws",
                "query_string": b"",
                "headers": headers or [],
                "client": ("test", 1),
                "server": ("testserver", 80),
                "subprotocols": [],
                "state": {},
            },
            self._incoming.get,
            self._send,
        )

    async def _send(self, message):
        if self._fail_accept and message["type"] == "websocket.accept":
            self._fail_accept = False
            raise OSError("private accept detail")
        if self._fail_close and message["type"] == "websocket.close":
            self._fail_close = False
            raise OSError("private close detail")
        if self._block_close and message["type"] == "websocket.close":
            self.close_started.set()
            await self.release_close.wait()
        if self._peer_send_close_code is not None and message["type"] == "websocket.send":
            code = self._peer_send_close_code
            self._peer_send_close_code = None
            raise WebSocketDisconnect(code=code)
        self.sent.append(message)

    async def send_frame(self, payload):
        await self._incoming.put(
            {
                "type": "websocket.receive",
                "text": json.dumps(payload),
            }
        )

    async def disconnect(self, code=1000):
        await self._incoming.put({"type": "websocket.disconnect", "code": code})

    def outbound_frames(self):
        return [json.loads(message["text"]) for message in self.sent if message["type"] == "websocket.send"]


@pytest.fixture
def spans(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        voice_host_module,
        "_TRACER",
        provider.get_tracer("azure.ai.agentserver.invocations.voice"),
        raising=False,
    )
    return provider, exporter


def _trace_headers(provider):
    caller = provider.get_tracer("test.caller").start_span("hosted_agents.invoke_agent")
    carrier = {}
    inject(carrier, context=trace.set_span_in_context(caller))
    return caller, [(name.encode("latin-1"), value.encode("latin-1")) for name, value in carrier.items()]


def _remote_trace_headers(*, sampled, trace_state):
    parent = NonRecordingSpan(
        SpanContext(
            trace_id=0x11111111111111111111111111111111,
            span_id=0x2222222222222222,
            is_remote=True,
            trace_flags=TraceFlags.SAMPLED if sampled else TraceFlags.DEFAULT,
            trace_state=TraceState.from_header([trace_state]),
        )
    )
    carrier = {}
    inject(carrier, context=trace.set_span_in_context(parent))
    return parent, [(name.encode("latin-1"), value.encode("latin-1")) for name, value in carrier.items()]


def _span_by_name(exporter, name):
    matches = [span for span in exporter.get_finished_spans() if span.name == name]
    assert len(matches) == 1, [span.name for span in exporter.get_finished_spans()]
    return matches[0]


async def _wait_for_span(exporter, name):
    for _ in range(10):
        matches = [span for span in exporter.get_finished_spans() if span.name == name]
        if matches:
            assert len(matches) == 1
            return matches[0]
        await asyncio.sleep(0)
    return _span_by_name(exporter, name)


@pytest.mark.asyncio
async def test_voice_input_callback_spans_preserve_parenting_without_owning_background_task(monkeypatch, spans):
    provider, exporter = spans
    monkeypatch.setenv("FOUNDRY_AGENT_NAME", "weather-agent")
    monkeypatch.setenv("FOUNDRY_AGENT_VERSION", "7")
    monkeypatch.setenv("FOUNDRY_AGENT_SESSION_ID", "session-42")
    caller, headers = _trace_headers(provider)
    socket = _VoiceSocket(headers)
    app = VoiceAgentServerHost(configure_observability=None)
    customer_tracer = provider.get_tracer("customer.agent")
    callback_returned = asyncio.Event()
    background_started = asyncio.Event()
    release_background = asyncio.Event()
    background_task = None

    @app.on_session_start
    async def on_session_start(session, _event):
        await session.send(SessionReady())

    @app.on_user_message
    async def on_user_message(_session, event):
        nonlocal background_task
        assert isinstance(event, UserMessage)
        with customer_tracer.start_as_current_span("customer.model.sync"):
            pass

        async def background_work():
            with customer_tracer.start_as_current_span("customer.model.background"):
                background_started.set()
                await release_background.wait()

        background_task = asyncio.create_task(background_work())
        callback_returned.set()

    endpoint = asyncio.create_task(app._ws_endpoint(socket.websocket))  # pylint: disable=protected-access
    await socket.send_frame(
        _frame(
            "session.start",
            "m_start",
            protocol_version="1.0",
            reconnect=False,
            response_timeouts={
                "first_output_ms": 1,
                "idle_ms": 2,
                "max_duration_ms": 3,
            },
        )
    )
    await socket.send_frame(
        _frame(
            "user.message",
            "m_user",
            item_id="in_1",
            content=[{"type": "input_text", "text": "private transcript"}],
        )
    )
    await asyncio.wait_for(callback_returned.wait(), timeout=1)
    await asyncio.wait_for(background_started.wait(), timeout=1)

    turn = await _wait_for_span(exporter, "invoke_agent weather-agent:7")
    assert not endpoint.done()
    assert background_task is not None and not background_task.done()

    release_background.set()
    await asyncio.wait_for(background_task, timeout=1)
    await socket.disconnect()
    await asyncio.wait_for(endpoint, timeout=1)
    caller.end()

    connection = _span_by_name(exporter, "agentserver.connection")
    synchronous = _span_by_name(exporter, "customer.model.sync")
    background = _span_by_name(exporter, "customer.model.background")
    assert connection.parent is not None and connection.parent.span_id == caller.context.span_id
    assert turn.parent is not None and turn.parent.span_id == connection.context.span_id
    assert synchronous.parent is not None and synchronous.parent.span_id == turn.context.span_id
    assert background.parent is not None and background.parent.span_id == turn.context.span_id
    assert turn.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert turn.attributes["bridge.input.count"] == 1
    assert turn.attributes["turn.origin"] == "user"
    assert connection.attributes["network.protocol.name"] == "websocket"
    assert connection.attributes["microsoft.session.id"] == "session-42"
    assert connection.attributes["azure.ai.agentserver.connection.outcome"] == "completed"
    assert "private transcript" not in repr(connection.attributes)
    assert "private transcript" not in repr(turn.attributes)


@pytest.mark.asyncio
@pytest.mark.parametrize("sampled", [False, True])
async def test_voice_connection_preserves_remote_sampling_and_tracestate(spans, sampled):
    _, exporter = spans
    parent, headers = _remote_trace_headers(sampled=sampled, trace_state="vendor=value")
    socket = _VoiceSocket(headers)
    app = VoiceAgentServerHost(configure_observability=None)

    await socket.disconnect()
    await app._ws_endpoint(socket.websocket)  # pylint: disable=protected-access

    finished = exporter.get_finished_spans()
    if not sampled:
        assert finished == ()
        return
    connection = _span_by_name(exporter, "agentserver.connection")
    assert connection.parent is not None
    assert connection.parent.span_id == parent.get_span_context().span_id
    assert connection.context.trace_id == parent.get_span_context().trace_id
    assert connection.context.trace_flags == parent.get_span_context().trace_flags
    assert connection.context.trace_state == parent.get_span_context().trace_state


@pytest.mark.asyncio
@pytest.mark.parametrize("sampled", [False, True])
async def test_voice_w3c_parent_is_independent_of_global_propagator(spans, sampled):
    _, exporter = spans
    parent, headers = _remote_trace_headers(sampled=sampled, trace_state="vendor=value")
    original_propagator = otel_propagate.get_global_textmap()
    otel_propagate.set_global_textmap(CompositePropagator([]))
    try:
        socket = _VoiceSocket(headers)
        app = VoiceAgentServerHost(configure_observability=None)
        await socket.disconnect()
        await app._ws_endpoint(socket.websocket)  # pylint: disable=protected-access
    finally:
        otel_propagate.set_global_textmap(original_propagator)

    finished = exporter.get_finished_spans()
    if not sampled:
        assert finished == ()
        return
    connection = _span_by_name(exporter, "agentserver.connection")
    assert connection.parent is not None
    assert connection.parent.span_id == parent.get_span_context().span_id
    assert connection.context.trace_id == parent.get_span_context().trace_id
    assert connection.context.trace_flags == parent.get_span_context().trace_flags
    assert connection.context.trace_state == parent.get_span_context().trace_state


@pytest.mark.asyncio
async def test_voice_signal_callbacks_remain_under_connection_without_turn_span(spans):
    provider, exporter = spans
    caller, headers = _trace_headers(provider)
    socket = _VoiceSocket(headers)
    app = VoiceAgentServerHost(configure_observability=None)
    customer_tracer = provider.get_tracer("customer.agent")

    @app.on_session_start
    async def on_session_start(session, _event):
        await session.send(SessionReady())

    @app.on_user_speech_started
    async def on_speech_started(_session, event):
        assert isinstance(event, UserSpeechStarted)
        with customer_tracer.start_as_current_span("customer.signal"):
            pass

    @app.on_session_end
    async def on_session_end(_session, event):
        assert isinstance(event, SessionEnd)

    endpoint = asyncio.create_task(app._ws_endpoint(socket.websocket))  # pylint: disable=protected-access
    await socket.send_frame(
        _frame(
            "session.start",
            "m_start",
            protocol_version="1.0",
            reconnect=False,
            response_timeouts={
                "first_output_ms": 1,
                "idle_ms": 2,
                "max_duration_ms": 3,
            },
        )
    )
    await socket.send_frame(_frame("user.speech_started", "m_speech"))
    await socket.send_frame(_frame("session.end", "m_end", reason="completed"))
    await asyncio.wait_for(endpoint, timeout=1)
    caller.end()

    connection = _span_by_name(exporter, "agentserver.connection")
    signal = _span_by_name(exporter, "customer.signal")
    assert signal.parent is not None and signal.parent.span_id == connection.context.span_id
    assert [span for span in exporter.get_finished_spans() if span.name.startswith("invoke_agent")] == []


@pytest.mark.asyncio
async def test_voice_turn_detach_failure_restores_connection_context(monkeypatch, spans):
    provider, exporter = spans
    socket = _VoiceSocket()
    app = VoiceAgentServerHost(configure_observability=None)
    customer_tracer = provider.get_tracer("customer.agent")
    runtime_context = voice_host_module._otel_context._RUNTIME_CONTEXT  # pylint: disable=protected-access
    original_detach = runtime_context.detach
    detach_calls = 0

    def fail_first_detach(token):
        nonlocal detach_calls
        detach_calls += 1
        if detach_calls == 1:
            raise RuntimeError("turn detach failed")
        original_detach(token)

    monkeypatch.setattr(runtime_context, "detach", fail_first_detach)

    @app.on_user_message
    async def on_user_message(_session, _event):
        return None

    @app.on_user_speech_started
    async def on_user_speech_started(_session, _event):
        with customer_tracer.start_as_current_span("customer.signal.after_detach"):
            pass

    endpoint = asyncio.create_task(app._ws_endpoint(socket.websocket))  # pylint: disable=protected-access
    await socket.send_frame(
        _frame(
            "user.message",
            "m_user",
            item_id="in_1",
            content=[{"type": "input_text", "text": "secret"}],
        )
    )
    await socket.send_frame(_frame("user.speech_started", "m_speech"))
    await socket.disconnect()
    await asyncio.wait_for(endpoint, timeout=1)

    connection = _span_by_name(exporter, "agentserver.connection")
    turn = _span_by_name(exporter, "invoke_agent")
    signal = _span_by_name(exporter, "customer.signal.after_detach")
    assert signal.parent is not None and signal.parent.span_id == connection.context.span_id
    assert signal.parent.span_id != turn.context.span_id
    assert detach_calls >= 3


@pytest.mark.asyncio
async def test_voice_connection_detach_failure_restores_calling_context(monkeypatch, spans):
    _, exporter = spans
    baseline_context = voice_host_module._otel_context.get_current()  # pylint: disable=protected-access
    socket = _VoiceSocket()
    app = VoiceAgentServerHost(configure_observability=None)

    runtime_context = voice_host_module._otel_context._RUNTIME_CONTEXT  # pylint: disable=protected-access

    def fail_detach(_token):
        raise RuntimeError("connection detach failed")

    monkeypatch.setattr(runtime_context, "detach", fail_detach)
    await socket.disconnect()
    try:
        await app._ws_endpoint(socket.websocket)  # pylint: disable=protected-access
        connection = _span_by_name(exporter, "agentserver.connection")
        assert voice_host_module._otel_context.get_current() is baseline_context  # pylint: disable=protected-access
        assert trace.get_current_span().get_span_context() != connection.context
    finally:
        voice_host_module._otel_context.attach(baseline_context)  # pylint: disable=protected-access


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["parent", "connection"])
@pytest.mark.parametrize("failure_mode", ["before", "after"])
async def test_voice_attach_failure_does_not_create_turn_or_modify_ambient_span(
    monkeypatch, spans, failure_stage, failure_mode
):
    provider, exporter = spans
    baseline = provider.get_tracer("customer.agent").start_span("customer.baseline")
    customer_tracer = provider.get_tracer("customer.agent")
    baseline_context = trace.set_span_in_context(baseline)
    baseline_token = voice_host_module._otel_context.attach(baseline_context)  # pylint: disable=protected-access
    original_attach = voice_host_module._otel_context.attach  # pylint: disable=protected-access
    attach_calls = 0

    def fail_selected_attach(context):
        nonlocal attach_calls
        attach_calls += 1
        if attach_calls == (1 if failure_stage == "parent" else 2):
            if failure_mode == "after":
                original_attach(context)
            raise RuntimeError(f"{failure_stage} attach failed")
        return original_attach(context)

    monkeypatch.setattr(
        voice_host_module._otel_context, "attach", fail_selected_attach
    )  # pylint: disable=protected-access
    _, headers = _remote_trace_headers(sampled=True, trace_state="vendor=value")
    socket = _VoiceSocket(headers)
    app = VoiceAgentServerHost(configure_observability=None)

    @app.on_user_message
    async def on_user_message(_session, _event):
        with customer_tracer.start_as_current_span("customer.after_attach_failure"):
            pass

    try:
        await socket.send_frame(
            _frame(
                "user.message",
                "m_user",
                item_id="in_1",
                content=[{"type": "input_text", "text": "secret"}],
            )
        )
        await socket.disconnect()
        await app._ws_endpoint(socket.websocket)  # pylint: disable=protected-access

        assert baseline.status.status_code is trace.StatusCode.UNSET
        assert "azure.ai.agentserver.connection.outcome" not in baseline.attributes
        assert voice_host_module._otel_context.get_current() is baseline_context  # pylint: disable=protected-access
        assert [span for span in exporter.get_finished_spans() if span.name == "invoke_agent"] == []
        connection_spans = [span for span in exporter.get_finished_spans() if span.name == "agentserver.connection"]
        assert len(connection_spans) == (0 if failure_stage == "parent" else 1)
        customer = _span_by_name(exporter, "customer.after_attach_failure")
        if connection_spans:
            assert customer.parent is not None
            assert customer.parent.span_id != connection_spans[0].context.span_id
    finally:
        baseline.end()
        voice_host_module._otel_context.detach(baseline_token)  # pylint: disable=protected-access


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_source", ["initial_context", "attached_context", "w3c_extract"])
async def test_voice_parent_context_failure_disables_spans_without_changing_dispatch(monkeypatch, failure_source):
    start_calls = []

    class RecordingTracer:
        @staticmethod
        def start_span(*args, **kwargs):
            start_calls.append((args, kwargs))
            return None

    monkeypatch.setattr(voice_host_module, "_TRACER", RecordingTracer())
    if failure_source in {"initial_context", "attached_context"}:
        original_get_current = voice_host_module._otel_context.get_current  # pylint: disable=protected-access
        get_current_calls = 0

        def fail_selected_current_lookup():
            nonlocal get_current_calls
            get_current_calls += 1
            if get_current_calls == (1 if failure_source == "initial_context" else 2):
                raise RuntimeError("current context failed")
            return original_get_current()

        monkeypatch.setattr(
            voice_host_module._otel_context,  # pylint: disable=protected-access
            "get_current",
            fail_selected_current_lookup,
        )
    else:
        monkeypatch.setattr(
            voice_host_module._VOICE_TRACE_PROPAGATOR,  # pylint: disable=protected-access
            "extract",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("W3C extract failed")),
        )
    socket = _VoiceSocket()
    app = VoiceAgentServerHost(configure_observability=None)
    callback_events = []

    @app.on_session_start
    async def on_session_start(session, event):
        callback_events.append(event.type)
        await session.send(SessionReady())

    endpoint = asyncio.create_task(app._ws_endpoint(socket.websocket))  # pylint: disable=protected-access
    await socket.send_frame(
        _frame(
            "session.start",
            "m_start",
            protocol_version="1.0",
            reconnect=False,
            response_timeouts={
                "first_output_ms": 1,
                "idle_ms": 2,
                "max_duration_ms": 3,
            },
        )
    )
    await socket.disconnect()
    await asyncio.wait_for(endpoint, timeout=1)

    assert callback_events == ["session.start"]
    assert [frame["type"] for frame in socket.outbound_frames()] == ["session.ready"]
    assert start_calls == []


@pytest.mark.asyncio
async def test_voice_callback_failure_records_only_sanitized_trace_data(spans, caplog):
    provider, exporter = spans
    caller, headers = _trace_headers(provider)
    socket = _VoiceSocket(headers)
    app = VoiceAgentServerHost(configure_observability=None)

    @app.on_user_message
    async def on_user_message(_session, _event):
        raise RuntimeError("private transcript in exception")

    with caplog.at_level(logging.ERROR, logger="azure.ai.agentserver"):
        endpoint = asyncio.create_task(app._ws_endpoint(socket.websocket))  # pylint: disable=protected-access
        await socket.send_frame(
            _frame(
                "user.message",
                "m_user",
                item_id="in_1",
                content=[{"type": "input_text", "text": "private transcript"}],
            )
        )
        await asyncio.wait_for(endpoint, timeout=1)
    caller.end()

    turn = _span_by_name(exporter, "invoke_agent")
    connection = _span_by_name(exporter, "agentserver.connection")
    assert turn.status.status_code is trace.StatusCode.ERROR
    assert turn.attributes["bridge.outcome"] == "error"
    assert turn.attributes["error.type"] == "callback_error"
    assert connection.attributes["azure.ai.agentserver.connection.outcome"] == "internal_error"
    assert "bridge.outcome" not in connection.attributes
    assert "private transcript" not in repr(turn.attributes)
    assert "private transcript" not in repr(turn.events)
    assert "private transcript" not in (turn.status.description or "")
    assert "private transcript" not in caplog.text
    assert "Voice WebSocket handler raised" in caplog.text


@pytest.mark.asyncio
async def test_voice_no_input_span_uses_generated_connection_session_id(monkeypatch, spans):
    _, exporter = spans
    monkeypatch.delenv("FOUNDRY_AGENT_SESSION_ID", raising=False)
    socket = _VoiceSocket()
    app = VoiceAgentServerHost(configure_observability=None)
    callback_finished = asyncio.Event()

    @app.on_user_no_input
    async def on_user_no_input(_session, event):
        assert isinstance(event, UserNoInput)
        callback_finished.set()

    endpoint = asyncio.create_task(app._ws_endpoint(socket.websocket))  # pylint: disable=protected-access
    await socket.send_frame(_frame("user.no_input", "m_no_input", item_id="in_1", count=2))
    await asyncio.wait_for(callback_finished.wait(), timeout=1)
    await socket.disconnect()
    await asyncio.wait_for(endpoint, timeout=1)

    connection = _span_by_name(exporter, "agentserver.connection")
    turn = _span_by_name(exporter, "invoke_agent")
    assert turn.attributes["turn.origin"] == "no_input"
    assert connection.attributes["microsoft.session.id"]
    assert turn.attributes["microsoft.session.id"] == connection.attributes["microsoft.session.id"]


@pytest.mark.asyncio
async def test_voice_owner_cancellation_ends_turn_and_connection_spans_once(spans):
    _, exporter = spans
    socket = _VoiceSocket()
    app = VoiceAgentServerHost(configure_observability=None)
    callback_entered = asyncio.Event()

    @app.on_user_message
    async def on_user_message(_session, _event):
        callback_entered.set()
        await asyncio.Future()

    endpoint = asyncio.create_task(app._ws_endpoint(socket.websocket))  # pylint: disable=protected-access
    await socket.send_frame(
        _frame(
            "user.message",
            "m_user",
            item_id="in_1",
            content=[{"type": "input_text", "text": "secret"}],
        )
    )
    await asyncio.wait_for(callback_entered.wait(), timeout=1)
    endpoint.cancel("test shutdown")
    with pytest.raises(asyncio.CancelledError):
        await endpoint

    turn = _span_by_name(exporter, "invoke_agent")
    connection = _span_by_name(exporter, "agentserver.connection")
    assert turn.attributes["bridge.outcome"] == "cancelled"
    assert turn.attributes["error.type"] == "cancelled"
    assert turn.status.status_code is trace.StatusCode.ERROR
    assert connection.attributes["azure.ai.agentserver.connection.outcome"] == "cancelled"
    assert len([span for span in exporter.get_finished_spans() if span.name == "agentserver.connection"]) == 1
    assert len([span for span in exporter.get_finished_spans() if span.name == "invoke_agent"]) == 1


@pytest.mark.asyncio
async def test_voice_cancellation_after_callback_return_does_not_reclassify_turn(spans):
    _, exporter = spans
    socket = _VoiceSocket()
    app = VoiceAgentServerHost(configure_observability=None)
    callback_returned = asyncio.Event()

    @app.on_user_message
    async def on_user_message(_session, _event):
        owner = asyncio.current_task()
        assert owner is not None
        asyncio.get_running_loop().call_soon(owner.cancel, "after callback")
        callback_returned.set()

    endpoint = asyncio.create_task(app._ws_endpoint(socket.websocket))  # pylint: disable=protected-access
    await socket.send_frame(
        _frame(
            "user.message",
            "m_user",
            item_id="in_1",
            content=[{"type": "input_text", "text": "secret"}],
        )
    )
    await asyncio.wait_for(callback_returned.wait(), timeout=1)
    with pytest.raises(asyncio.CancelledError):
        await endpoint

    turn = _span_by_name(exporter, "invoke_agent")
    connection = _span_by_name(exporter, "agentserver.connection")
    assert turn.status.status_code is trace.StatusCode.UNSET
    assert "bridge.outcome" not in turn.attributes
    assert "error.type" not in turn.attributes
    assert connection.attributes["azure.ai.agentserver.connection.outcome"] == "cancelled"
    assert len([span for span in exporter.get_finished_spans() if span.name == "agentserver.connection"]) == 1
    assert len([span for span in exporter.get_finished_spans() if span.name == "invoke_agent"]) == 1


@pytest.mark.asyncio
async def test_voice_accept_failure_ends_connection_span_without_turn(spans, caplog):
    _, exporter = spans
    socket = _VoiceSocket(fail_accept=True)
    app = VoiceAgentServerHost(configure_observability=None)

    with caplog.at_level(logging.ERROR, logger="azure.ai.agentserver"):
        await app._ws_endpoint(socket.websocket)  # pylint: disable=protected-access

    connection = _span_by_name(exporter, "agentserver.connection")
    assert connection.status.status_code is trace.StatusCode.ERROR
    assert connection.attributes["azure.ai.agentserver.connection.outcome"] == "accept_failed"
    assert connection.attributes["error.type"] == "accept_failed"
    assert "bridge.outcome" not in connection.attributes
    assert [span for span in exporter.get_finished_spans() if span.name.startswith("invoke_agent")] == []
    assert "private accept detail" not in caplog.text


@pytest.mark.asyncio
async def test_voice_sdk_close_failure_marks_connection_transport_error(spans, caplog):
    _, exporter = spans
    socket = _VoiceSocket(fail_close=True)
    app = VoiceAgentServerHost(configure_observability=None)
    endpoint = asyncio.create_task(app._ws_endpoint(socket.websocket))  # pylint: disable=protected-access

    with caplog.at_level(logging.DEBUG, logger="azure.ai.agentserver"):
        await socket.send_frame(_frame("session.end", "m_end", reason="completed"))
        await asyncio.wait_for(endpoint, timeout=1)

    connection = _span_by_name(exporter, "agentserver.connection")
    assert connection.status.status_code is trace.StatusCode.ERROR
    assert connection.attributes["azure.ai.agentserver.connection.outcome"] == "transport_error"
    assert connection.attributes["error.type"] == "transport_error"
    assert "private close detail" not in repr(connection.attributes)
    assert "private close detail" not in repr(connection.events)
    assert "private close detail" not in caplog.text
    assert len([span for span in exporter.get_finished_spans() if span.name == "agentserver.connection"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("hook", ["terminating", "disconnect"])
async def test_voice_teardown_failure_logs_are_content_free(spans, caplog, hook):
    _, exporter = spans
    socket = _VoiceSocket()
    app = VoiceAgentServerHost(configure_observability=None)

    if hook == "terminating":

        @app.on_connection_terminating
        def on_connection_terminating(_session):
            raise RuntimeError("private termination detail")

    else:

        @app.on_disconnect
        async def on_disconnect(_session, _event):
            raise RuntimeError("private disconnect detail")

    with caplog.at_level(logging.ERROR, logger="azure.ai.agentserver"):
        endpoint = asyncio.create_task(app._ws_endpoint(socket.websocket))  # pylint: disable=protected-access
        await socket.disconnect(code=1006)
        await asyncio.wait_for(endpoint, timeout=1)

    connection = _span_by_name(exporter, "agentserver.connection")
    assert connection.attributes["azure.ai.agentserver.connection.outcome"] == "transport_error"
    assert connection.attributes["azure.ai.agentserver.connection.cleanup_error"] is True
    assert connection.attributes["error.type"] == "transport_error"
    assert "private termination detail" not in caplog.text
    assert "private disconnect detail" not in caplog.text
    assert f"Voice {hook if hook == 'disconnect' else 'connection termination'} callback failed" in caplog.text


@pytest.mark.asyncio
async def test_voice_sdk_close_timeout_marks_connection_transport_error(monkeypatch, spans):
    _, exporter = spans
    session_module = voice_host_module._session_transport  # pylint: disable=protected-access
    monkeypatch.setattr(session_module, "CLOSE_TIMEOUT_SECONDS", 0.01)
    baseline_attempts = set(session_module._CLOSE_ATTEMPTS)  # pylint: disable=protected-access
    socket = _VoiceSocket(block_close=True)
    app = VoiceAgentServerHost(configure_observability=None)
    endpoint = asyncio.create_task(app._ws_endpoint(socket.websocket))  # pylint: disable=protected-access

    try:
        await socket.send_frame(_frame("session.end", "m_end", reason="completed"))
        await asyncio.wait_for(socket.close_started.wait(), timeout=1)
        await asyncio.wait_for(endpoint, timeout=1)

        connection = _span_by_name(exporter, "agentserver.connection")
        assert connection.status.status_code is trace.StatusCode.ERROR
        assert connection.attributes["azure.ai.agentserver.connection.outcome"] == "transport_error"
        assert connection.attributes["error.type"] == "transport_error"
        assert len([span for span in exporter.get_finished_spans() if span.name == "agentserver.connection"]) == 1
    finally:
        socket.release_close.set()
        outstanding = set(session_module._CLOSE_ATTEMPTS) - baseline_attempts  # pylint: disable=protected-access
        if outstanding:
            await asyncio.wait_for(asyncio.gather(*outstanding, return_exceptions=True), timeout=1)
        await asyncio.sleep(0)
        assert set(session_module._CLOSE_ATTEMPTS) == baseline_attempts  # pylint: disable=protected-access


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("incoming", "expected_outcome"),
    [
        ({"type": "websocket.receive", "text": "not-json"}, "protocol_error"),
        ({"type": "websocket.disconnect", "code": 1002}, "transport_error"),
        ({"type": "websocket.disconnect", "code": 1006}, "transport_error"),
    ],
)
async def test_voice_connection_failure_has_sanitized_outcome(spans, incoming, expected_outcome):
    _, exporter = spans
    socket = _VoiceSocket()
    app = VoiceAgentServerHost(configure_observability=None)
    endpoint = asyncio.create_task(app._ws_endpoint(socket.websocket))  # pylint: disable=protected-access

    await socket._incoming.put(incoming)  # pylint: disable=protected-access
    await asyncio.wait_for(endpoint, timeout=1)

    connection = _span_by_name(exporter, "agentserver.connection")
    assert connection.status.status_code is trace.StatusCode.ERROR
    assert connection.attributes["azure.ai.agentserver.connection.outcome"] == expected_outcome
    assert connection.attributes["error.type"] == expected_outcome
    assert "bridge.outcome" not in connection.attributes
    assert voice_host_module._VOICE_TRACING_ENABLED not in socket.websocket.scope  # pylint: disable=protected-access
    assert (
        voice_host_module._VOICE_LOCAL_PROTOCOL_ERROR not in socket.websocket.scope
    )  # pylint: disable=protected-access


@pytest.mark.asyncio
async def test_voice_accepted_proactive_callback_is_a_turn_but_dropped_callback_is_not(
    spans,
):
    provider, exporter = spans
    socket = _VoiceSocket()
    app = VoiceAgentServerHost(configure_observability=None)
    customer_tracer = provider.get_tracer("customer.agent")

    @app.on_response_accepted
    async def on_response_accepted(_session, event):
        assert isinstance(event, ResponseAccepted)
        with customer_tracer.start_as_current_span("customer.proactive"):
            pass

    @app.on_response_dropped
    async def on_response_dropped(_session, event):
        assert isinstance(event, ResponseDropped)
        with customer_tracer.start_as_current_span("customer.dropped"):
            pass

    endpoint = asyncio.create_task(app._ws_endpoint(socket.websocket))  # pylint: disable=protected-access
    await socket.send_frame(_frame("response.accepted", "m_accept", response_id="r_proactive"))
    await socket.send_frame(
        _frame(
            "response.dropped",
            "m_drop",
            response_id="r_dropped",
            reason="no_barge_safe_window",
        )
    )
    await socket.disconnect()
    await asyncio.wait_for(endpoint, timeout=1)

    connection = _span_by_name(exporter, "agentserver.connection")
    turn = _span_by_name(exporter, "invoke_agent")
    proactive = _span_by_name(exporter, "customer.proactive")
    dropped = _span_by_name(exporter, "customer.dropped")
    assert turn.attributes["turn.origin"] == "proactive"
    assert turn.attributes["bridge.input.count"] == 0
    assert turn.attributes["gen_ai.response.id"] == "r_proactive"
    assert proactive.parent is not None and proactive.parent.span_id == turn.context.span_id
    assert dropped.parent is not None and dropped.parent.span_id == connection.context.span_id
    assert len([span for span in exporter.get_finished_spans() if span.name == "invoke_agent"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["start", "span_context", "lifecycle"])
async def test_voice_tracer_failure_does_not_change_dispatch(monkeypatch, failure_stage):
    class FailingSpan:
        @staticmethod
        def set_attribute(*_args, **_kwargs):
            raise RuntimeError("span attribute failed")

        @staticmethod
        def set_status(*_args, **_kwargs):
            raise RuntimeError("span status failed")

        @staticmethod
        def end(*_args, **_kwargs):
            raise RuntimeError("span end failed")

    class FailingTracer:
        @staticmethod
        def start_span(*_args, **_kwargs):
            if failure_stage == "start":
                raise RuntimeError("tracer failed")
            return FailingSpan()

    monkeypatch.setattr(voice_host_module, "_TRACER", FailingTracer(), raising=False)
    if failure_stage == "span_context":
        monkeypatch.setattr(
            voice_host_module._otel_trace,  # pylint: disable=protected-access
            "set_span_in_context",
            lambda _span: (_ for _ in ()).throw(RuntimeError("span context failed")),
        )
    socket = _VoiceSocket()
    app = VoiceAgentServerHost(configure_observability=None)
    callback_events = []

    @app.on_session_start
    async def on_session_start(session, event):
        callback_events.append(event.type)
        await session.send(SessionReady())

    endpoint = asyncio.create_task(app._ws_endpoint(socket.websocket))  # pylint: disable=protected-access
    await socket.send_frame(
        _frame(
            "session.start",
            "m_start",
            protocol_version="1.0",
            reconnect=False,
            response_timeouts={
                "first_output_ms": 1,
                "idle_ms": 2,
                "max_duration_ms": 3,
            },
        )
    )
    await socket.disconnect()
    await asyncio.wait_for(endpoint, timeout=1)

    assert callback_events == ["session.start"]
    assert [frame["type"] for frame in socket.outbound_frames()] == ["session.ready"]


@pytest.mark.asyncio
async def test_voice_span_start_failure_cannot_poison_callback_context(monkeypatch):
    baseline_context = otel_context.get_current()
    observed_poison = []

    class PoisoningTracer:
        @staticmethod
        def start_span(*_args, **_kwargs):
            poisoned = otel_baggage.set_baggage(
                "external.poison",
                "span-start",
                context=otel_context.get_current(),
            )
            otel_context.attach(poisoned)
            raise RuntimeError("span start failed")

    monkeypatch.setattr(voice_host_module, "_TRACER", PoisoningTracer())
    socket = _VoiceSocket()
    app = VoiceAgentServerHost(configure_observability=None)

    @app.on_session_start
    async def on_session_start(session, _event):
        observed_poison.append(otel_baggage.get_baggage("external.poison"))
        await session.send(SessionReady())

    try:
        endpoint = asyncio.create_task(app._ws_endpoint(socket.websocket))  # pylint: disable=protected-access
        await socket.send_frame(
            _frame(
                "session.start",
                "m_start",
                protocol_version="1.0",
                reconnect=False,
                response_timeouts={
                    "first_output_ms": 1,
                    "idle_ms": 2,
                    "max_duration_ms": 3,
                },
            )
        )
        await socket.disconnect()
        await asyncio.wait_for(endpoint, timeout=1)
    finally:
        voice_host_module._restore_voice_context(baseline_context)  # pylint: disable=protected-access

    assert observed_poison == [None]
    assert [frame["type"] for frame in socket.outbound_frames()] == ["session.ready"]
    assert otel_context.get_current() == baseline_context


@pytest.mark.asyncio
async def test_voice_span_end_failure_cannot_poison_later_callback_context(monkeypatch):
    baseline_context = otel_context.get_current()
    observed_poison = []

    class PoisoningSpan:
        @staticmethod
        def set_attribute(*_args, **_kwargs):
            return None

        @staticmethod
        def set_status(*_args, **_kwargs):
            return None

        @staticmethod
        def end(*_args, **_kwargs):
            poisoned = otel_baggage.set_baggage(
                "external.poison",
                "span-end",
                context=otel_context.get_current(),
            )
            otel_context.attach(poisoned)
            raise RuntimeError("span end failed")

    class PoisoningTracer:
        @staticmethod
        def start_span(*_args, **_kwargs):
            return PoisoningSpan()

    monkeypatch.setattr(voice_host_module, "_TRACER", PoisoningTracer())
    socket = _VoiceSocket()
    app = VoiceAgentServerHost(configure_observability=None)

    @app.on_user_message
    async def on_user_message(_session, _event):
        return None

    @app.on_user_speech_started
    async def on_user_speech_started(_session, _event):
        observed_poison.append(otel_baggage.get_baggage("external.poison"))

    try:
        endpoint = asyncio.create_task(app._ws_endpoint(socket.websocket))  # pylint: disable=protected-access
        await socket.send_frame(
            _frame(
                "user.message",
                "m_user",
                item_id="in_1",
                content=[{"type": "input_text", "text": "private transcript"}],
            )
        )
        await socket.send_frame(_frame("user.speech_started", "m_speech"))
        await socket.disconnect()
        await asyncio.wait_for(endpoint, timeout=1)
    finally:
        voice_host_module._restore_voice_context(baseline_context)  # pylint: disable=protected-access

    assert observed_poison == [None]
    assert otel_context.get_current() == baseline_context


def test_voice_observability_failure_restores_non_contextvars_runtime(monkeypatch):
    baseline_context = object()
    poisoned_context = object()
    runtime = {"current": baseline_context}

    monkeypatch.setattr(
        voice_host_module._otel_context,  # pylint: disable=protected-access
        "get_current",
        lambda: runtime["current"],
    )

    def attach(context):
        runtime["current"] = context
        return object()

    monkeypatch.setattr(
        voice_host_module._otel_context,  # pylint: disable=protected-access
        "attach",
        attach,
    )

    def poison_runtime():
        runtime["current"] = poisoned_context
        raise RuntimeError("observability callback failed")

    with pytest.raises(RuntimeError, match="observability callback failed"):
        voice_host_module._run_voice_observability(poison_runtime)  # pylint: disable=protected-access

    assert runtime["current"] is baseline_context


class _FakeLogRecord:
    def __init__(self):
        self.attributes = {}


class _FakeLogData:
    def __init__(self):
        self.log_record = _FakeLogRecord()


def test_voice_upgrade_filters_baggage_before_core_log_export():
    websocket = _VoiceSocket(
        [
            (
                b"traceparent",
                b"00-11111111111111111111111111111111-2222222222222222-01",
            ),
            (
                b"baggage",
                b"leaf_customer_span_id=123,transcript=private-transcript",
            ),
            (
                b"baggage",
                b"azure.ai.agentserver.conversation_id=conversation-1,heard_text=private-heard",
            ),
            (b"x-request-id", b"x" * 257),
        ]
    ).websocket
    context = voice_host_module._extract_voice_websocket_context(websocket)  # pylint: disable=protected-access
    assert context is not None
    log_data = _FakeLogData()
    processor = _BaggageLogRecordProcessor()
    token = otel_context.attach(context)
    try:
        processor.on_emit(log_data)
    finally:
        otel_context.detach(token)

    attributes = log_data.log_record.attributes
    assert attributes["leaf_customer_span_id"] == "123"
    assert attributes["azure.ai.agentserver.conversation_id"] == "conversation-1"
    assert "transcript" not in attributes
    assert "heard_text" not in attributes
    assert "x_request_id" not in attributes


def _voice_close_records(records):
    return [record for record in records if hasattr(record, "azure.ai.agentserver.invocations_ws.close_code")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("terminal_kind", "expected_code", "expected_outcome", "expected_error_code"),
    [
        ("local_protocol", 1002, "protocol_error", "internal_error"),
        ("peer_normal", 1000, "completed", "internal_error"),
        ("peer_abnormal", 1006, "transport_error", "internal_error"),
        ("accept_failure", 1011, "accept_failed", "internal_error"),
    ],
)
async def test_voice_cleanup_failure_does_not_override_first_terminal(
    spans,
    caplog,
    terminal_kind,
    expected_code,
    expected_outcome,
    expected_error_code,
):
    _, exporter = spans
    socket = _VoiceSocket(fail_accept=terminal_kind == "accept_failure")
    app = VoiceAgentServerHost(configure_observability=None)

    if terminal_kind == "peer_abnormal":

        @app.on_disconnect
        async def on_disconnect(_session, _event):
            raise RuntimeError("private disconnect detail")

        expected_cleanup_log = "Voice disconnect callback failed"
    else:

        @app.on_connection_terminating
        def on_connection_terminating(_session):
            raise RuntimeError("private termination detail")

        expected_cleanup_log = "Voice connection termination callback failed"

    with caplog.at_level(logging.INFO, logger="azure.ai.agentserver"):
        endpoint = asyncio.create_task(app._ws_endpoint(socket.websocket))  # pylint: disable=protected-access
        if terminal_kind == "local_protocol":
            await socket._incoming.put(  # pylint: disable=protected-access
                {"type": "websocket.receive", "text": "not-json"}
            )
        elif terminal_kind == "peer_normal":
            await socket.disconnect(code=1000)
        elif terminal_kind == "peer_abnormal":
            await socket.disconnect(code=1006)
        await asyncio.wait_for(endpoint, timeout=1)

    connection = _span_by_name(exporter, "agentserver.connection")
    assert connection.attributes["azure.ai.agentserver.connection.outcome"] == expected_outcome
    assert connection.attributes["azure.ai.agentserver.connection.cleanup_error"] is True
    if expected_outcome == "completed":
        assert connection.status.status_code is trace.StatusCode.ERROR
        assert connection.attributes["error.type"] == "cleanup_error"
    else:
        assert connection.attributes["error.type"] == expected_outcome
    close_records = _voice_close_records(caplog.records)
    assert len(close_records) == 1
    assert getattr(close_records[0], "azure.ai.agentserver.invocations_ws.close_code") == expected_code
    assert getattr(close_records[0], "azure.ai.agentserver.invocations_ws.error.code", None) == expected_error_code
    assert expected_cleanup_log in caplog.text
    assert "private termination detail" not in caplog.text
    assert "private disconnect detail" not in caplog.text


@pytest.mark.asyncio
async def test_voice_peer_terminal_survives_cleanup_cancellation(spans, caplog):
    _, exporter = spans
    socket = _VoiceSocket()
    app = VoiceAgentServerHost(configure_observability=None)
    disconnect_entered = asyncio.Event()

    @app.on_disconnect
    async def on_disconnect(_session, _event):
        disconnect_entered.set()
        await asyncio.Future()

    with caplog.at_level(logging.INFO, logger="azure.ai.agentserver"):
        endpoint = asyncio.create_task(app._ws_endpoint(socket.websocket))  # pylint: disable=protected-access
        await socket.disconnect(code=1006)
        await asyncio.wait_for(disconnect_entered.wait(), timeout=1)
        endpoint.cancel("cleanup cancellation")
        with pytest.raises(asyncio.CancelledError):
            await endpoint

    connection = _span_by_name(exporter, "agentserver.connection")
    assert connection.attributes["azure.ai.agentserver.connection.outcome"] == "transport_error"
    assert connection.attributes["azure.ai.agentserver.connection.cleanup_error"] is True
    assert connection.attributes["error.type"] == "transport_error"
    close_records = _voice_close_records(caplog.records)
    assert len(close_records) == 1
    assert getattr(close_records[0], "azure.ai.agentserver.invocations_ws.close_code") == 1006
    assert getattr(close_records[0], "azure.ai.agentserver.invocations_ws.error.code", None) == "cancelled"


@pytest.mark.asyncio
async def test_voice_local_protocol_terminal_survives_close_cleanup_cancellation(spans, caplog):
    _, exporter = spans
    session_module = voice_host_module._session_transport  # pylint: disable=protected-access
    baseline_attempts = set(session_module._CLOSE_ATTEMPTS)  # pylint: disable=protected-access
    socket = _VoiceSocket(block_close=True)
    app = VoiceAgentServerHost(configure_observability=None)

    try:
        with caplog.at_level(logging.INFO, logger="azure.ai.agentserver"):
            endpoint = asyncio.create_task(app._ws_endpoint(socket.websocket))  # pylint: disable=protected-access
            await socket._incoming.put(  # pylint: disable=protected-access
                {"type": "websocket.receive", "text": "not-json"}
            )
            await asyncio.wait_for(socket.close_started.wait(), timeout=1)
            endpoint.cancel("close cleanup cancellation")
            with pytest.raises(asyncio.CancelledError):
                await endpoint

        connection = _span_by_name(exporter, "agentserver.connection")
        assert connection.attributes["azure.ai.agentserver.connection.outcome"] == "protocol_error"
        assert connection.attributes["azure.ai.agentserver.connection.cleanup_error"] is True
        assert connection.attributes["error.type"] == "protocol_error"
        close_records = _voice_close_records(caplog.records)
        assert len(close_records) == 1
        assert getattr(close_records[0], "azure.ai.agentserver.invocations_ws.close_code") == 1002
        assert getattr(close_records[0], "azure.ai.agentserver.invocations_ws.error.code", None) == "cancelled"
    finally:
        socket.release_close.set()
        outstanding = set(session_module._CLOSE_ATTEMPTS) - baseline_attempts  # pylint: disable=protected-access
        if outstanding:
            await asyncio.wait_for(
                asyncio.gather(*outstanding, return_exceptions=True),
                timeout=1,
            )
        await asyncio.sleep(0)
        assert set(session_module._CLOSE_ATTEMPTS) == baseline_attempts  # pylint: disable=protected-access


@pytest.mark.asyncio
async def test_voice_peer_terminal_wins_same_code_local_late_race(spans):
    _, exporter = spans
    socket = _VoiceSocket(peer_send_close_code=1002)
    app = VoiceAgentServerHost(configure_observability=None)
    disconnect_codes = []

    @app.on_user_message
    async def on_user_message(session, _event):
        try:
            await session.send(SessionReady())
        except WebSocketDisconnect:
            pass

    @app.on_disconnect
    async def on_disconnect(_session, event):
        disconnect_codes.append(event.code)

    endpoint = asyncio.create_task(app._ws_endpoint(socket.websocket))  # pylint: disable=protected-access
    await socket.send_frame(
        _frame(
            "user.message",
            "m_user",
            item_id="in_1",
            content=[{"type": "input_text", "text": "private transcript"}],
        )
    )
    await socket._incoming.put({"type": "websocket.receive", "text": "not-json"})  # pylint: disable=protected-access
    await asyncio.wait_for(endpoint, timeout=1)

    connection = _span_by_name(exporter, "agentserver.connection")
    assert disconnect_codes == [1002]
    assert connection.attributes["azure.ai.agentserver.connection.outcome"] == "transport_error"
    assert connection.attributes["error.type"] == "transport_error"
    assert (
        voice_host_module._VOICE_LOCAL_PROTOCOL_ERROR not in socket.websocket.scope
    )  # pylint: disable=protected-access


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "configured_session_id",
    ["invalid session", "x" * 257, "é" * 129],
)
async def test_voice_invalid_configured_session_id_uses_safe_fallback(
    monkeypatch,
    spans,
    configured_session_id,
):
    _, exporter = spans
    monkeypatch.setenv("FOUNDRY_AGENT_SESSION_ID", configured_session_id)
    socket = _VoiceSocket()
    app = VoiceAgentServerHost(configure_observability=None)
    observed_session_ids = []

    @app.on_session_start
    async def on_session_start(session, _event):
        observed_session_ids.append(
            (
                get_request_context().session_id,
                otel_baggage.get_baggage("azure.ai.agentserver.session_id"),
            )
        )
        await session.send(SessionReady())

    endpoint = asyncio.create_task(app._ws_endpoint(socket.websocket))  # pylint: disable=protected-access
    await socket.send_frame(
        _frame(
            "session.start",
            "m_start",
            protocol_version="1.0",
            reconnect=False,
            response_timeouts={
                "first_output_ms": 1,
                "idle_ms": 2,
                "max_duration_ms": 3,
            },
        )
    )
    await socket.disconnect()
    await asyncio.wait_for(endpoint, timeout=1)

    connection = _span_by_name(exporter, "agentserver.connection")
    safe_session_id = connection.attributes["microsoft.session.id"]
    assert safe_session_id != configured_session_id
    assert len(safe_session_id.encode("utf-8")) <= 256
    assert (
        voice_host_module._VALID_CORRELATION_ID.fullmatch(safe_session_id) is not None
    )  # pylint: disable=protected-access
    assert observed_session_ids == [(safe_session_id, safe_session_id)]


@pytest.mark.asyncio
async def test_voice_wrapped_owner_cancellation_marks_turn_cancelled(spans):
    _, exporter = spans
    socket = _VoiceSocket()
    app = VoiceAgentServerHost(configure_observability=None)
    callback_started = asyncio.Event()

    @app.on_user_message
    async def on_user_message(_session, _event):
        callback_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError as cancellation:
            raise RuntimeError("wrapped cancellation") from cancellation

    endpoint = asyncio.create_task(app._ws_endpoint(socket.websocket))  # pylint: disable=protected-access
    await socket.send_frame(
        _frame(
            "user.message",
            "m_user",
            item_id="in_1",
            content=[{"type": "input_text", "text": "private transcript"}],
        )
    )
    await asyncio.wait_for(callback_started.wait(), timeout=1)
    endpoint.cancel("owner cancellation")
    task_counter_available = (
        voice_host_module._task_cancellation_requests() is not None
    )  # pylint: disable=protected-access
    if task_counter_available:
        with pytest.raises(asyncio.CancelledError):
            await endpoint
    else:
        await endpoint

    turn = _span_by_name(exporter, "invoke_agent")
    connection = _span_by_name(exporter, "agentserver.connection")
    expected_outcome = "cancelled" if task_counter_available else "error"
    expected_error_type = "cancelled" if task_counter_available else "callback_error"
    expected_connection = "cancelled" if task_counter_available else "internal_error"
    assert turn.attributes["bridge.outcome"] == expected_outcome
    assert turn.attributes["error.type"] == expected_error_type
    assert connection.attributes["azure.ai.agentserver.connection.outcome"] == expected_connection


@pytest.mark.asyncio
async def test_voice_wrapped_cancellation_without_task_counter_is_callback_error(
    monkeypatch,
    spans,
):
    _, exporter = spans
    monkeypatch.setattr(
        voice_host_module,
        "_task_cancellation_requests",
        lambda: None,
    )
    socket = _VoiceSocket()
    app = VoiceAgentServerHost(configure_observability=None)
    callback_started = asyncio.Event()

    @app.on_user_message
    async def on_user_message(_session, _event):
        callback_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError as cancellation:
            raise RuntimeError("wrapped cancellation") from cancellation

    endpoint = asyncio.create_task(app._ws_endpoint(socket.websocket))  # pylint: disable=protected-access
    await socket.send_frame(
        _frame(
            "user.message",
            "m_user",
            item_id="in_1",
            content=[{"type": "input_text", "text": "private transcript"}],
        )
    )
    await asyncio.wait_for(callback_started.wait(), timeout=1)
    endpoint.cancel("owner cancellation")
    await endpoint

    turn = _span_by_name(exporter, "invoke_agent")
    connection = _span_by_name(exporter, "agentserver.connection")
    assert turn.attributes["bridge.outcome"] == "error"
    assert turn.attributes["error.type"] == "callback_error"
    assert connection.attributes["azure.ai.agentserver.connection.outcome"] == "internal_error"
