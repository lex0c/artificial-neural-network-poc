import asyncio
import websockets
from datetime import datetime
import json
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def echo(websocket, path):
    try:
        logger.info("%s - Client Connected: %s", datetime.now(), websocket.remote_address)

        async for message in websocket:
            logger.info("%s - Received: %s", datetime.now(), message)

            data = None

            try:
                data = json.loads(message)
                if not isinstance(data, dict):  # Validating received data
                    raise ValueError("Invalid data format")
            except json.JSONDecodeError:
                logger.error("%s - Invalid JSON received", datetime.now())
                continue
            except ValueError as ve:
                logger.error("%s - %s", datetime.now(), ve)
                continue

            await websocket.send(json.dumps(data))

            logger.info("%s - Sent: %s", datetime.now(), data)
    except websockets.ConnectionClosed as e:
        logger.warning(e)
    except Exception as e:
        logger.error(e)
    finally:
        logger.info("%s - Client Disconnected: %s", datetime.now(), websocket.remote_address)


async def main():
    port = 5000
    server = await websockets.serve(echo, "localhost", port)
    logger.info("Network on. Waiting for interactions on port %s", port)
    await server.wait_closed()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Server has been stopped.")


