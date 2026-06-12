import pytest
import asyncio
from sluice.server import low_level_generate, ChatCompletionRequest, ARGS, BANK
from sluice.pools import PoolConfig

@pytest.mark.asyncio
async def test_low_level_generate_crash():
    # Attempt to actually run the generator loop 
    # Requires engine to be real, but we can mock it?
    pass
