"""Interface transport only: forward official WebSocket frames to a WSS endpoint."""
import argparse
import asyncio
import signal
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve


async def forward(local, url):
    # No observation decoding or policy logic occurs in this bridge.
    try:
        async with connect(url, compression=None, max_size=None, open_timeout=30,
                           ping_interval=20, ping_timeout=20) as remote:
            async def pump(source, destination):
                async for message in source:
                    await destination.send(message)
            tasks = [asyncio.create_task(pump(local, remote)), asyncio.create_task(pump(remote, local))]
            try:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    except Exception:
        await local.close(code=1011, reason='Remote endpoint unavailable')


async def run(args):
    stopped = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(sig, stopped.set)
    async with serve(lambda connection: forward(connection, args.url), args.host, args.port,
                     compression=None, max_size=None):
        print(f'Interface bridge ready on {args.host}:{args.port}', flush=True)
        await stopped.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--port', type=int, default=19021)
    ap.add_argument('--host', default='127.0.0.1')
    args = ap.parse_args()
    if not args.url.startswith(('wss://', 'ws://')):
        ap.error('url must start with ws:// or wss://')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
