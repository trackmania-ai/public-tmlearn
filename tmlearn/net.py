# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/net.ipynb.

# %% auto 0
__all__ = ['ThreadServer', 'WebsocketServer']

# %% ../nbs/net.ipynb 1
import asyncio
import json
import logging
import queue
from threading import Thread

import torch.multiprocessing as mp
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from uvicorn import Config, Server
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

# %% ../nbs/net.ipynb 3
class ThreadServer:
    def __init__(self, app, **kwargs):
        self.server = Server(Config(app, **kwargs))
        self.thread = Thread(target=self.server.run)
        self.thread.start()
        while not self.server.started:
            pass

    @property
    def port(self):
        return (
            self.server.config.port
            or self.server.servers[0].sockets[0].getsockname()[1]
        )

    def __del__(self):
        self.server.should_exit = True
        self.thread.join()

# %% ../nbs/net.ipynb 4
class WebsocketServer(ThreadServer):
    def __init__(self, port):
        self.publishers = dict()
        self.subscribers = dict()
        self.last_object = dict()

        app = FastAPI()

        @app.websocket("/ws/{name}")
        async def websocket_endpoint(websocket: WebSocket, name: str):
            if name not in self.publishers:
                logging.info(f"Publisher {name} does not exist..")
                return
            await websocket.accept()
            logging.debug(f"WebSocket connection to {name}.")
            q = queue.Queue(2)
            self.subscribers[name].append(q)
            try:
                if name in self.last_object:
                    await websocket.send_json(self.last_object[name])
                while True:
                    try:
                        o = q.get(False)
                    except queue.Empty:
                        await asyncio.sleep(0.001)
                        continue
                    await websocket.send_json(o)
            except (WebSocketDisconnect, ConnectionClosedError, ConnectionClosedOK):
                logging.debug(f"WebSocket disconnection from {name}.")
                self.subscribers[name].remove(q)

        super().__init__(app, host="127.0.0.1", port=port, log_level="error")

    def _put_to_subscribers(self, name):
        while True:
            o = self.publishers[name].get()
            for q in self.subscribers[name]:
                try:
                    q.put(o, False)
                except queue.Full:
                    pass
            self.last_object[name] = o

    def add_publisher(self, name):
        logging.debug(f"Adding {name} publisher to WebsocketServer")
        self.publishers[name] = mp.Queue(2)
        self.subscribers[name] = []
        Thread(target=self._put_to_subscribers, args=(name,)).start()
        return self.publishers[name]
