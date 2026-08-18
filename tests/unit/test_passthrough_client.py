from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from coeval.clients.passthrough import PassthroughClient


@pytest.mark.asyncio
async def test_generate_raises_when_response_content_is_none() -> None:
    llm = SimpleNamespace(
        achat=AsyncMock(
            return_value=SimpleNamespace(message=SimpleNamespace(content=None))
        )
    )

    with pytest.raises(RuntimeError, match="Inference response contained no content"):
        await PassthroughClient(llm=llm).generate([])
