"""
Streaming LLM client communicating with Ollama with sentence-boundary chunking.
"""

import json
import re
from typing import AsyncGenerator, List, Optional
import httpx

from sonus.config import LLMConfig


class StreamingLLMClient:
    """Async streaming client for local LLM inference with sentence-level pipelining."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._sentence_regex = re.compile(r"([^.?!]+[.?!]+)")

    async def generate_sentences(
        self,
        user_message: str,
        conversation_history: Optional[List[dict]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream LLM response chunked at natural sentence boundaries.
        
        Yields:
            Individual sentences as soon as they are fully completed.
        """
        messages = [{"role": "system", "content": self.config.system_prompt}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.config.temperature,
            }
        }

        url = f"{self.config.base_url.rstrip('/')}/api/chat"
        buffer = ""

        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        buffer += content

                        # Extract completed sentences
                        parts = self._sentence_regex.split(buffer)
                        if len(parts) > 1:
                            # parts alternates: [sentence1, delimiter, sentence2, delimiter, remainder]
                            # Or with capture groups: [sentence1, remainder]
                            # Let's cleanly yield all complete sentences and leave remainder in buffer
                            completed = parts[:-1]
                            for sent in completed:
                                clean_sent = sent.strip()
                                if clean_sent:
                                    yield clean_sent
                            buffer = parts[-1]

                    if chunk.get("done", False):
                        break

        # Flush any trailing words that didn't end in punctuation
        trailing = buffer.strip()
        if trailing:
            yield trailing
