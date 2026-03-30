"""
base.py
───────
Foundation for all LLM interactions.
- Singleton ChatCohere client (avoids re-init per call).
- Retry with exponential back-off.
- Shared chunked-invocation helper used by every refiner.
"""

import time
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from helpers.config import COHERE_API_KEY
from helpers.logger import CustomLogger

LLM_MODEL = "command-a-03-2025"
MAX_RETRIES = 2
RETRY_BASE_DELAY = 5  # seconds


class BaseLLM:
    """Thin wrapper around ChatCohere with structured-output support."""

    _client: ChatCohere | None = None  # class-level singleton

    def __init__(self):
        if not COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY environment variable is not set.")
        # Reuse a single client across all instances
        if BaseLLM._client is None:
            BaseLLM._client = ChatCohere(
                model=LLM_MODEL,
                api_key=COHERE_API_KEY,
                temperature=0.2,
            )
        self.llm = BaseLLM._client

    # ── single invocation (with retry) ───────────────────────────────────────

    def invoke(self, prompt: str, response_format, user_data):
        """Invoke the LLM with structured output parsing and automatic retry."""
        structured_llm = self.llm.with_structured_output(response_format)

        chat_prompt = ChatPromptTemplate([
            {"role": "system", "content": prompt},
            {"role": "user", "content": "{user_data}"}
        ])

        chain = chat_prompt | structured_llm

        last_err = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = chain.invoke({"user_data": str(user_data)})
                return result
            except Exception as e:
                last_err = e
                delay = RETRY_BASE_DELAY ** attempt
                ts = time.strftime('%X')
                print(f"[{ts}] ⚠️  LLM attempt {attempt}/{MAX_RETRIES} failed: {e}")
                CustomLogger.log(f"LLM retry {attempt}/{MAX_RETRIES} – {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(delay)

        raise RuntimeError(f"LLM invocation failed after {MAX_RETRIES} attempts") from last_err

    # ── chunked invocation (shared by all refiners) ──────────────────────────

    def invoke_chunked(
        self,
        *,
        items: list,
        prompt: str,
        response_format,
        chunk_size: int = 120,
        result_key: str = "words",
        label: str = "REFINING",
    ) -> list:
        """
        Split *items* into chunks, invoke the LLM for each chunk, and
        concatenate the results.  Falls back to the original chunk data
        when a single chunk fails.

        Parameters
        ----------
        items         : flat list of dicts (or strings) to process.
        prompt        : system prompt.
        response_format : Pydantic model for structured output.
        chunk_size    : max items per LLM call.
        result_key    : attribute name on the response model that holds
                        the list of results (e.g. "words", "refined_lyrics").
        label         : tag used in log messages.
        """
        total = len(items)
        print(f"[{time.strftime('%X')}] [{label}] {total} items → chunk_size={chunk_size}")
        CustomLogger.log(f"--- [{label}] LLM INPUT ---\n{items}")

        chunks = [items[i:i + chunk_size] for i in range(0, total, chunk_size)]
        all_results: list = []

        for idx, chunk in enumerate(chunks, 1):
            ts = time.strftime('%X')
            print(f"[{ts}] [{label}] chunk {idx}/{len(chunks)} ({len(chunk)} items)…")
            try:
                result = self.invoke(prompt, response_format, chunk)
                part = getattr(result, result_key)
                all_results.extend(
                    item.model_dump() if hasattr(item, "model_dump") else item
                    for item in part
                )
                print(f"[{time.strftime('%X')}] [{label}] chunk {idx} ✔")
            except Exception as e:
                print(f"[{time.strftime('%X')}] ⚠️ [{label}] chunk {idx} failed: {e}")
                print(f"[{time.strftime('%X')}] [{label}] falling back to originals for chunk {idx}")
                all_results.extend(chunk)

        print(f"[{time.strftime('%X')}] [{label}] done — {len(all_results)} items out")
        CustomLogger.log(f"--- [{label}] LLM OUTPUT ---\n{all_results}")
        return all_results