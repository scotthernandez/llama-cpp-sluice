import asyncio
import pytest
from sluice.bank import TokenBank, BankSaturated

@pytest.mark.asyncio
async def test_bank_basic():
    bank = TokenBank(["p"], {"p": 100}, 20, 50)
    sid = await bank.acquire("p", 30)
    assert sid == 1
    assert bank.used["p"] == 30
    await bank.release(sid)
    assert bank.used["p"] == 0

@pytest.mark.asyncio
async def test_bank_starvation():
    bank = TokenBank(["p"], {"p": 100}, 40, 50)
    await bank.acquire("p", 60) # sid 1
    # Large task waiting
    lt = asyncio.create_task(bank.acquire("p", 40))
    await asyncio.sleep(0.1)
    # Small task blocked
    with pytest.raises(BankSaturated):
        await bank.acquire("p", 5, timeout=0.1)
    await bank.release(1)
    await lt
