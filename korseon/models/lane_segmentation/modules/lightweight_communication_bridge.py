from typing import Callable
import json
import threading
from websockets.sync.client import connect
from websockets import ConnectionClosedOK

WebsocketMessageHandler = Callable[[dict[str, object]], None]

class LCB:
    def __init__(self, handler: WebsocketMessageHandler, host:str, port:str) -> None:
        self.handler = handler
        self.websocket = connect(f"ws://{host}:{port}", max_size= 10 * 1024 * 1024)
        self.listeningThread = threading.Thread(target=self.listeningLoop)
        self.listeningThread.daemon = True
        self.listeningThread.start()

    def listeningLoop(self) -> None:
        while True:
            try:
                msg = self.websocket.recv()
                self.handler(json.loads(msg))
            except ConnectionClosedOK:
                break

    def send(self, destination:str, payload: dict[str, object]):
        self.websocket.send(f"preflight:{destination}")
        self.websocket.send(json.dumps(payload))

    def close(self):
        self.send({'tag': 'close'})