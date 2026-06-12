import asyncio
import pytest
from sluice.bank import TokenBank, BankSaturated

@pytest.mark.asyncio
async def test_bank_basic():
    bank = TokenBank(100, 20, 50)
    sid = await bank.acquire(30)
    assert sid == 0
    assert bank.used == 30
    await bank.release(sid)
    assert bank.used == 0

@pytest.mark.asyncio
async def test_bank_starvation():
    bank = TokenBank(100, 40, 50)
    await bank.acquire(60) # sid 0
    # Large task waiting
    lt = asyncio.create_task(bank.acquire(40))
    await asyncio.sleep(0.1)
    # Small task blocked
    with pytest.raises(BankSaturated):
        await bank.acquire(5, timeout=0.1)
    await bank.release(0)
    await lt