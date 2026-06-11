import asyncio
import pytest
from sluice.bank import TokenBank

@pytest.mark.asyncio
async def test_bank_basic_acquire_release():
    bank = TokenBank(100, 20, 50)
    sid = await bank.acquire(30)
    assert sid == 1
    assert bank.used == 30
    await bank.release(sid)
    assert bank.used == 0

@pytest.mark.asyncio
async def test_bank_barrier():
    bank = TokenBank(100, 40, 50)
    await bank.acquire(30)
    await bank.acquire(25)
    with pytest.raises(TimeoutError):
        await bank.acquire(10, timeout=0.1)

@pytest.mark.asyncio
async def test_bank_starvation():
    bank = TokenBank(100, 10, 50)
    await bank.acquire(50)
    large_task = asyncio.create_task(bank.acquire(50))
    await asyncio.sleep(0.1)
    with pytest.raises(TimeoutError):
        await bank.acquire(5, timeout=0.1)
    await bank.release(1)
    await large_task
