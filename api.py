import asyncio
import websockets
from datetime import datetime
import json
import logging

from feedforward import FeedForward
from etc import normalize_minmax, save_model, load_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


nn_agent = FeedForward(verbose=False)
nn_agent.add_layer(num_inputs=4, num_neurons=4, act_fn="relu")
nn_agent.add_layer(num_inputs=4, num_neurons=4, act_fn="relu")
nn_agent.add_layer(num_inputs=4, num_neurons=1, act_fn="linear")


async def echo(websocket, path):
    try:
        logger.info("%s - Client Connected: %s", datetime.now(), websocket.remote_address)

        state = await websocket.recv() # Receive the initial state from the client.
        state = json.loads(state)

        action = nn_agent.select_action(normalize_minmax(state)) # The model selects an action based on the current state.
        await websocket.send(json.dumps(action))  # Send the action to the client.

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

            next_state = normalize_minmax(data['next_state'])

            nn_agent.learn(state, action, data['reward'], next_state, data['done'])  # The model learns from the experience.
            save_model('api_agent', nn_agent.layers)

            if not data['done']:
                action = nn_agent.select_action(next_state) # The model selects an action based on the current state.
                await websocket.send(json.dumps(action))  # Send the action to the client.
                logger.info("%s - Sent: %s", datetime.now(), action)

            state = next_state
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


