import asyncio

import pytest

from app.memory.thread_locks import ThreadLocks


@pytest.mark.asyncio
async def test_same_thread_runs_are_serialized():
    locks = ThreadLocks()
    running = 0
    peak = 0

    async def run():
        nonlocal running, peak
        async with locks.lock("thread1"):
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.001)
            running -= 1

    await asyncio.gather(*(run() for _ in range(10)))
    assert peak == 1


@pytest.mark.asyncio
async def test_different_threads_are_not_blocked_by_each_other():
    locks = ThreadLocks()
    entered = asyncio.Event()

    async def first():
        async with locks.lock("one"):
            entered.set()
            await asyncio.sleep(0.01)

    async def second():
        await entered.wait()
        async with locks.lock("two"):
            return True

    a, b = await asyncio.gather(first(), second())
    assert b is True
