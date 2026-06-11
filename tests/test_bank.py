import asyncio
import pytest
from sluice.bank import TokenBank

@pytest.mark.asyncio
async def test_bank_basic_acquire_release():
    bank = TokenBank(total_tokens=100, reserved_for_large=20, large_threshold=50)
    sid = await bank.acquire(30)
    assert sid == 1
    assert bank.used == 30
    await bank.release(sid)
    assert bank.used == 0

@pytest.mark.asyncio
async def test_bank_barrier_logic():
    bank = TokenBank(total_tokens=100, reserved_for_large=40, large_threshold=50)
    await bank.acquire(30)
    await bank.acquire(25)
    assert bank.used == 55
    with pytest.raises(TimeoutError):
        await bank.acquire(10, timeout=0.1)

@pytest.mark.asyncio
async def test_bank_priority_large_request():
    bank = TokenBank(total_tokens=100, reserved_for_large=40, large_threshold=50)
    await bank.acquire(50)
    sid = await bank.acquire(50, timeout=0.1)
    assert sid is not None
    assert bank.used == 100

@pytest.mark.asyncio
async def test_bank_anti_starvation():
    # Threshold is 50. 
    bank = TokenBank(total_tokens=100, reserved_for_large=40, large_threshold=50)
    await bank.acquire(60) # sid 1
    
    # Start large request (>= 50) in background
    # It will be blocked (Total 100, Used 60, Need 50)
    large_task = asyncio.create_task(bank.acquire(50))
    
    # Wait for large_task to increment waiting_large
    for _ in range(20):
        if bank.waiting_large == 1: break
        await asyncio.sleep(0.05)
    assert bank.waiting_large == 1
    
    # Small request (2) should be blocked even if it fits (100-60-2 = 38 > reserved? No wait, 38 < 40)
    # Available is 40. Request 2 would leave 38. 38 < 40 (Reserved). 
    # So it would be blocked by barrier anyway.
    
    # Let's test anti-starvation specifically:
    # If we have 50 tokens used, 50 available. Reserved is 40.
    # A small request of 5 tokens would leave 45 (OK by barrier).
    # But if a LARGE is waiting, it should block.
    
    bank2 = TokenBank(total_tokens=100, reserved_for_large=10, large_threshold=50)
    await bank2.acquire(50) # sid 1
    
    # Large waiting
    large_task2 = asyncio.create_task(bank2.acquire(50))
    for _ in range(10):
        if bank2.waiting_large == 1: break
        await asyncio.sleep(0.05)
    
    # Small request (5) would fit (100-50-5=45 > 10), but blocked by Large waiting
    with pytest.raises(TimeoutError):
        await bank2.acquire(5, timeout=0.1)

@pytest.mark.asyncio
async def test_bank_drain():
    bank = TokenBank(total_tokens=100, reserved_for_large=20)
    await bank.acquire(10)
    asyncio.create_task(bank.drain())
    await asyncio.sleep(0.1)
    assert bank.is_draining is True
    with pytest.raises(RuntimeError, match="draining"):
        await bank.acquire(5)
    await bank.release(1)
    await asyncio.sleep(0.1)
