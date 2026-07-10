"""Asyncio HTTP server for coding-guardrails proxy.

Reuses Forge's raw HTTP server pattern with OpenAI-compatible routing.
Adds guardrail-specific endpoints and wires our Layer 2 into the handler.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from forge.clients.base import LLMClient
from forge.context.manager import ContextManager

from coding_guardrails.middleware import CodingGuardrails
from coding_guardrails.proxy.handler import handle_chat_completions

logger = logging.getLogger("coding_guardrails.server")

_MAX_BODY = 16 * 1024 * 1024


@dataclass
class _QueueItem:
    """A request waiting to be processed."""

    body: dict[str, Any]
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())
    cancelled: bool = False


class GuardrailProxyServer:
    """OpenAI-compatible proxy with Forge + coding guardrails.

    Single merged proxy:
    - Layer 1 (Forge): rescue parsing, retries, validation
    - Layer 2 (our rules): read-before-edit, path safety, command blocking,
      secret masking, test-after-change, tool resolution
    """

    def __init__(
        self,
        client: LLMClient,
        context_manager: ContextManager,
        guardrails: CodingGuardrails,
        host: str = "127.0.0.1",
        port: int = 8081,
        serialize_requests: bool = False,
        max_retries: int = 3,
        rescue_enabled: bool = True,
        model_name: str = "coding-guardrails",
        backend_manager=None,  # optional coding_guardrails.server.manager.BackendManager
    ) -> None:
        self._client = client
        self._context_manager = context_manager
        self._guardrails = guardrails
        self._host = host
        self._port = port
        self._model_name = model_name
        self._max_retries = max_retries
        self._rescue_enabled = rescue_enabled
        self._serialize = serialize_requests
        self._backend_manager = backend_manager
        self._server: asyncio.Server | None = None
        self._queue: asyncio.Queue[_QueueItem] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start listening."""
        if self._serialize:
            self._worker_task = asyncio.create_task(self._inference_worker())
        self._server = await asyncio.start_server(
            self._handle_connection, self._host, self._port,
        )
        logger.info("Proxy listening on %s:%d", self._host, self._port)

    async def stop(self) -> None:
        """Stop the server."""
        if self._backend_manager is not None:
            await self._backend_manager.unload_now()
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    # ── Internal ──

    async def _inference_worker(self) -> None:
        """Single-worker for serialized GPU access."""
        while True:
            item = await self._queue.get()
            try:
                if item.cancelled or item.future.cancelled():
                    continue
                result = await self._run_handler(item.body)
                if not item.future.done():
                    item.future.set_result(result)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not item.future.done():
                    item.future.set_result(exc)
            finally:
                self._queue.task_done()

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single HTTP connection."""
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=30.0)
            if not request_line:
                return

            parts = request_line.decode("utf-8", errors="replace").strip().split(" ", 2)
            if len(parts) < 2:
                await self._send_error(writer, 400, "Bad request")
                return

            method, path = parts[0], parts[1]
            logger.info(">> %s %s", method, path)

            headers = await self._read_headers(reader)
            content_length = int(headers.get("content-length", "0"))

            body_bytes = b""
            if content_length > 0:
                if content_length > _MAX_BODY:
                    await self._send_error(writer, 413, "Request too large")
                    return
                body_bytes = await asyncio.wait_for(
                    reader.readexactly(content_length), timeout=60.0,
                )

            if method == "GET" and path == "/health":
                await self._send_json(writer, 200, json.dumps({"status": "ok"}))
            elif method == "GET" and path == "/v1/models":
                await self._handle_models(writer)
            elif method == "POST" and path == "/v1/chat/completions":
                await self._handle_completions(writer, body_bytes)
            elif method == "OPTIONS":
                await self._send_cors_preflight(writer)
            else:
                await self._send_error(writer, 404, "Not found")

        except (asyncio.TimeoutError, asyncio.IncompleteReadError, ConnectionError):
            pass
        except Exception:
            logger.exception("Unhandled error")
            try:
                await self._send_error(writer, 500, "Internal server error")
            except Exception:
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _read_headers(self, reader: asyncio.StreamReader) -> dict[str, str]:
        headers: dict[str, str] = {}
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=30.0)
            decoded = line.decode("utf-8", errors="replace").strip()
            if not decoded:
                break
            if ":" in decoded:
                key, value = decoded.split(":", 1)
                headers[key.strip().lower()] = value.strip()
        return headers

    async def _handle_models(self, writer: asyncio.StreamWriter) -> None:
        """GET /v1/models — returns model info."""
        model_info: dict[str, Any] = {
            "id": self._model_name,
            "object": "model",
            "owned_by": "coding-guardrails",
        }

        # Proxy the backend's model metadata (includes n_ctx)
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as http:
                resp = await http.get(f"{self._client.base_url}/models")
                if resp.status_code == 200:
                    data = resp.json()
                    for m in data.get("data", []):
                        if "meta" in m:
                            model_info["meta"] = m["meta"]
                            break
        except Exception:
            pass

        body = json.dumps({
            "object": "list",
            "data": [model_info],
        })
        await self._send_json(writer, 200, body)

    async def _handle_completions(
        self,
        writer: asyncio.StreamWriter,
        body_bytes: bytes,
    ) -> None:
        """POST /v1/chat/completions."""
        try:
            body = json.loads(body_bytes)
        except json.JSONDecodeError:
            await self._send_error(writer, 400, "Invalid JSON")
            return

        is_stream = body.get("stream", False)
        msg_count = len(body.get("messages", []))
        tool_count = len(body.get("tools", []))
        logger.info("")
        logger.info("-" * 60)
        logger.info(">> POST /v1/chat/completions")
        logger.info("   msgs=%d tools=%d stream=%s model=%s",
                    msg_count, tool_count, is_stream, body.get("model", "?"))

        if self._serialize:
            item = _QueueItem(body=body)
            if is_stream:
                await self._send_sse_header(writer)
            self._queue.put_nowait(item)
            result = await self._await_with_disconnect(item, writer)
        else:
            if is_stream:
                await self._send_sse_header(writer)
            result = await self._run_handler(body)

        if result is None:
            return

        if isinstance(result, Exception):
            if is_stream:
                await self._send_sse_body(writer, [{"error": str(result)}])
            else:
                await self._send_error(writer, 502, str(result))
            return

        if is_stream:
            await self._send_sse_body(writer, result)
        else:
            await self._send_json(writer, 200, json.dumps(result))

    async def _await_with_disconnect(
        self, item: _QueueItem, writer: asyncio.StreamWriter,
    ) -> Any:
        while not item.future.done():
            if writer.is_closing():
                item.cancelled = True
                return None
            try:
                await asyncio.wait_for(asyncio.shield(item.future), timeout=1.0)
            except asyncio.TimeoutError:
                continue
        return item.future.result()

    async def _run_handler(self, body: dict[str, Any]) -> Any:
        # Managed backend: ensure the GPU model is loaded (VRAM-gate + queue) before
        # inference, then release → idle-unload timer. Acquire may raise
        # BackendUnavailable (queue-timeout) → surfaced as 503 → fleet L2 fallback.
        acquired = False
        if self._backend_manager is not None:
            try:
                await self._backend_manager.acquire()
                acquired = True
            except Exception as exc:
                logger.warning("backend unavailable: %s", exc)
                return exc
        try:
            return await handle_chat_completions(
                body=body,
                client=self._client,
                context_manager=self._context_manager,
                guardrails=self._guardrails,
                max_retries=self._max_retries,
                rescue_enabled=self._rescue_enabled,
            )
        except Exception as exc:
            logger.exception("Handler error")
            return exc
        finally:
            if acquired:
                await self._backend_manager.release()

    # ── HTTP helpers ──

    async def _send_json(self, writer: asyncio.StreamWriter, status: int, body: str) -> None:
        response = (
            f"HTTP/1.1 {status} {_status_text(status)}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body.encode())}\r\n"
            f"Connection: close\r\n"
            f"Access-Control-Allow-Origin: *\r\n"
            f"\r\n"
            f"{body}"
        )
        writer.write(response.encode())
        await writer.drain()

    async def _send_sse_header(self, writer: asyncio.StreamWriter) -> None:
        header = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Transfer-Encoding: chunked\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Connection: keep-alive\r\n"
            "\r\n"
        )
        writer.write(header.encode())
        await writer.drain()

    async def _send_sse_body(self, writer: asyncio.StreamWriter, events: list[dict[str, Any]]) -> None:
        for event in events:
            if writer.is_closing():
                return
            data = f"data: {json.dumps(event)}\n\n".encode()
            writer.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            await writer.drain()

        done = b"data: [DONE]\n\n"
        writer.write(f"{len(done):x}\r\n".encode() + done + b"\r\n")
        writer.write(b"0\r\n\r\n")
        await writer.drain()

    async def _send_error(self, writer: asyncio.StreamWriter, status: int, message: str) -> None:
        body = json.dumps({"error": {"message": message, "type": "proxy_error"}})
        await self._send_json(writer, status, body)

    async def _send_cors_preflight(self, writer: asyncio.StreamWriter) -> None:
        response = (
            "HTTP/1.1 204 No Content\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
            "Connection: close\r\n"
            "\r\n"
        )
        writer.write(response.encode())
        await writer.drain()


def _status_text(code: int) -> str:
    return {
        200: "OK", 204: "No Content", 400: "Bad Request",
        404: "Not Found", 413: "Payload Too Large",
        500: "Internal Server Error", 502: "Bad Gateway",
    }.get(code, "Error")
