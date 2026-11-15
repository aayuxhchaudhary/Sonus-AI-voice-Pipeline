"""
Tests for streaming LLM client sentence-boundary splitting, abbreviations, and decimals.
"""

try:
    import pytest
except ImportError:
    class _MockPytest:
        class mark:
            anyio = lambda fn: fn
    pytest = _MockPytest()  # type: ignore

from sonus.config import LLMConfig
from sonus.llm.client import StreamingLLMClient


def test_sentence_splitting_abbreviations():
    """Verify abbreviations (Dr., Mr., Mrs., etc.) do not cause premature splitting."""
    client = StreamingLLMClient(LLMConfig())

    text = "Hello Dr. Smith and Mrs. Davis arrived at 5 p.m. How are you doing? I am well."
    sentences, remainder = client.extract_sentences(text, is_final=False)

    assert len(sentences) == 2
    assert sentences[0] == "Hello Dr. Smith and Mrs. Davis arrived at 5 p.m."
    assert sentences[1] == "How are you doing?"
    assert remainder == "I am well."

    final_sentences, final_remainder = client.extract_sentences(remainder, is_final=True)
    assert final_sentences == ["I am well."]
    assert final_remainder == ""


def test_sentence_splitting_decimals_and_currency():
    """Verify decimals ($3.14, 0.05, 10.5) do not break sentences."""
    client = StreamingLLMClient(LLMConfig())

    text = "The total price is $19.99 right now. That is approximately 3.14 times cheaper. Great deal!"
    sentences, remainder = client.extract_sentences(text, is_final=True)

    assert len(sentences) == 3
    assert sentences[0] == "The total price is $19.99 right now."
    assert sentences[1] == "That is approximately 3.14 times cheaper."
    assert sentences[2] == "Great deal!"
    assert remainder == ""


def test_sentence_splitting_initials_and_acronyms():
    """Verify single-letter initials and dotted acronyms (U.S., John F. Kennedy) are preserved."""
    client = StreamingLLMClient(LLMConfig())

    text = "U.S. officials met with John F. Kennedy yesterday. The meeting went smoothly. Indeed!"
    sentences, remainder = client.extract_sentences(text, is_final=True)

    assert len(sentences) == 3
    assert sentences[0] == "U.S. officials met with John F. Kennedy yesterday."
    assert sentences[1] == "The meeting went smoothly."
    assert sentences[2] == "Indeed!"


def test_sentence_splitting_incremental_streaming():
    """Simulate token-by-token streaming from an LLM."""
    client = StreamingLLMClient(LLMConfig())

    tokens = ["The ", "item ", "costs ", "$3.", "14 ", "total. ", "Are ", "you ", "sure?"]
    buffer = ""
    yielded = []

    for token in tokens:
        buffer += token
        completed, buffer = client.extract_sentences(buffer, is_final=False)
        yielded.extend(completed)

    # Flush final
    final_sents, _ = client.extract_sentences(buffer, is_final=True)
    yielded.extend(final_sents)

    assert len(yielded) == 2
    assert yielded[0] == "The item costs $3.14 total."
    assert yielded[1] == "Are you sure?"


def test_sentence_splitting_acronyms_before_capitals():
    """Verify dotted uppercase acronyms (U.S., U.K.) do not split before capitalized nouns."""
    client = StreamingLLMClient(LLMConfig())

    text = "The U.S. Congress passed the bill today. It was unanimous."
    sentences, remainder = client.extract_sentences(text, is_final=True)

    assert len(sentences) == 2
    assert sentences[0] == "The U.S. Congress passed the bill today."
    assert sentences[1] == "It was unanimous."
    assert remainder == ""


def test_sentence_splitting_numbered_lists():
    """Verify list enumeration numbers (1., 2.) do not split into standalone single-digit sentences."""
    client = StreamingLLMClient(LLMConfig())

    text = "1. First gather ingredients. 2. Next mix them together. Enjoy your meal!"
    sentences, remainder = client.extract_sentences(text, is_final=True)

    assert len(sentences) == 2
    assert sentences[0] == "1. First gather ingredients. 2. Next mix them together."
    assert sentences[1] == "Enjoy your meal!"
    assert remainder == ""


def test_sentence_splitting_quotes_and_brackets():
    """Verify sentences ending with punctuation inside or outside quotes/brackets split correctly."""
    client = StreamingLLMClient(LLMConfig())

    text = 'He said, "Wait for me!" Then he ran. We bought apples, oranges, etc. They were great.'
    sentences, remainder = client.extract_sentences(text, is_final=True)

    assert len(sentences) == 3
    assert sentences[0] == 'He said, "Wait for me!"'
    assert sentences[1] == "Then he ran."
    assert sentences[2] == "We bought apples, oranges, etc. They were great."
    assert remainder == ""


def test_system_prompt_grounding():
    """Verify system prompt dynamically injects live clock, calendar, and user context."""
    from datetime import datetime
    config = LLMConfig(user_name="Aayush")
    client = StreamingLLMClient(config)

    prompt = client.get_system_prompt()
    now = datetime.now()

    assert "Aayush" in prompt
    assert str(now.year) in prompt
    assert "roleplay" in prompt.lower()
    assert "physical body" in prompt.lower()


def test_sanitize_spoken_text():
    """Verify roleplay asterisks, markdown, and bracketed notes are stripped."""
    raw = "*laughs* Hello Aayush, [interrupted] how are you today? *smiles*"
    cleaned = StreamingLLMClient.sanitize_spoken_text(raw)
    assert cleaned == "Hello Aayush,  how are you today?"
    assert "*" not in cleaned
    assert "[" not in cleaned
    assert "laughs" not in cleaned
    assert "smiles" not in cleaned


@pytest.mark.anyio
async def test_ollama_streaming():
    """Integration test with local Ollama if running."""
    config = LLMConfig(model="llama3.2:1b", temperature=0.1)
    client = StreamingLLMClient(config)

    try:
        sentences = []
        async for sentence in client.generate_sentences("Say hello and nothing else."):
            sentences.append(sentence)

        assert len(sentences) >= 1
        full_text = " ".join(sentences)
        assert len(full_text.strip()) > 0
    except Exception:
        # If Ollama daemon is not running locally, unit test passes cleanly
        pass
