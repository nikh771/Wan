"""Fan-out job progress to WebSocket subscribers (thread-safe → asyncio)."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


class ProgressHub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._queues: dict[str, list[asyncio.Queue[dict[str, Any]]]] = defaultdict(list)
        self._loop: asyncio.AbstractEventLoop | None = None

    def register_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        with self._lock:
            self._loop = loop

    def subscribe(self, job_id: str) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=64)
        with self._lock:
            self._queues[job_id].append(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            subs = self._queues.get(job_id, [])
            if queue in subs:
                subs.remove(queue)
            if not subs:
                self._queues.pop(job_id, None)

    def publish(self, job_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            subs = list(self._queues.get(job_id, []))
            loop = self._loop
        if not subs:
            return

        def _deliver() -> None:
            for q in subs:
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    try:
                        _ = q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    try:
                        q.put_nowait(event)
                    except asyncio.QueueFull:
                        logger.debug("Dropping progress event for %s (queue full)", job_id)

        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(_deliver)
        else:
            _deliver()


progress_hub = ProgressHub()


async def stream_job_events(
    websocket,
    job_id: str,
    *,
    poll_fallback,
) -> None:
    """Relay hub events to the client; poll when idle so progress never stalls."""
    queue = progress_hub.subscribe(job_id)
    try:
        await websocket.send_text(
            json.dumps({"type": "subscribed", "job_id": job_id})
        )
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
                await websocket.send_text(json.dumps(event))
                if event.get("done"):
                    break
            except asyncio.TimeoutError:
                snap = poll_fallback()
                if snap is not None:
                    await websocket.send_text(json.dumps(snap))
                    if snap.get("done"):
                        break
    finally:
        progress_hub.unsubscribe(job_id, queue)
