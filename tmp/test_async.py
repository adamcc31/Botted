import asyncio
import httpx
import sys
from pathlib import Path
sys.path.append('.')
from scripts.fetch_oracle_prices import fetch_pyth_price

async def main():
    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(1)
        r = await fetch_pyth_price(client, 1767916800000, sem)
        print(f"Price: {r}")

if __name__ == "__main__":
    asyncio.run(main())
