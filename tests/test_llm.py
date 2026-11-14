"""
Integration test for streaming LLM client with local Ollama.
"""

import pytest
from sonus.config import LLMConfig
from sonus.llm.client import StreamingLLMClient


@pytest.mark.anyio
async def test_ollama_streaming():
    config = LLMConfig(model="llama3.2:1b", temperature=0.1)
    client = StreamingLLMClient(config)

    sentences = []
    async for sentence in client.generate_sentences("Say hello and nothing else."):
        sentences.append(sentence)

    assert len(sentences) >= 1
    full_text = " ".join(sentences)
    assert len(full_text.strip()) > 0
