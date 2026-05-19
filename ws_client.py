"""Python 3.7-compatible drop-in replacement for
`openpi_client.websocket_client_policy.WebsocketClientPolicy`.

The upstream client imports `websockets.sync.client`, which only exists on
`websockets >= 12` (and therefore Python >= 3.8). The `franka` conda env on
katef004 is Python 3.7 and is capped at `websockets <= 11`, so we re-implement
the same protocol on top of the async API (available since websockets 10).

Wire-protocol parity: msgpack frames serialized via `openpi_client.msgpack_numpy`,
identical to what the openpi WebsocketPolicyServer expects.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional

import websockets  # async API; works on both 10/11 and 12+

from openpi_client import msgpack_numpy


class WebsocketClientPolicy:
    """Synchronous facade over an async websockets connection.

    Mirrors the upstream `WebsocketClientPolicy` API:
      - `__init__(host, port, api_key=None)`
      - `infer(obs: dict) -> dict`
      - `get_server_metadata() -> dict`
      - `reset()` (no-op, server is stateless)
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"

        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        # Dedicated event loop so we can drive async I/O from blocking call sites.
        self._loop = asyncio.new_event_loop()
        self._ws, self._server_metadata = self._loop.run_until_complete(
            self._wait_for_server()
        )

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    async def _wait_for_server(self):
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                extra_headers = (
                    [("Authorization", f"Api-Key {self._api_key}")]
                    if self._api_key
                    else None
                )
                conn = await websockets.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    extra_headers=extra_headers,
                    ping_interval=60,
                    ping_timeout=300,
                )
                metadata = msgpack_numpy.unpackb(await conn.recv())
                return conn, metadata
            except (ConnectionRefusedError, OSError) as e:
                logging.info(f"Still waiting for server... ({e})")
                await asyncio.sleep(5)

    def infer(self, obs: Dict) -> Dict:
        data = self._packer.pack(obs)
        return self._loop.run_until_complete(self._infer_async(data))

    async def _infer_async(self, data: bytes) -> Dict:
        await self._ws.send(data)
        response = await self._ws.recv()
        if isinstance(response, str):
            # Server sends bytes on success; a str payload signals an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def reset(self) -> None:
        pass
