import os
import re
import time

from dotenv import load_dotenv
from google import genai

# The google-genai SDK raises typed errors (ClientError / ServerError)
# from google.genai.errors for HTTP-level failures such as 429 rate
# limits. Import defensively -- if the SDK's error module ever moves,
# fall back to catching plain Exception so the app never hard-crashes
# just because a response couldn't be generated.
try:
    from google.genai import errors as genai_errors
except ImportError:  # pragma: no cover
    genai_errors = None

load_dotenv()


class LLM:

    # Friendly message shown when the Gemini free-tier quota is hit.
    RATE_LIMIT_MESSAGE = (
        "⚠️ **The AI is temporarily rate-limited.**\n\n"
        "Gemini's free-tier quota for this model has been used up "
        "for the current window. This isn't a bug in the app -- "
        "it's a limit on the API key's plan. Please wait a minute "
        "and try again, or ask a shorter / more specific question "
        "next time to use fewer tokens.\n\n"
        "See: https://ai.google.dev/gemini-api/docs/rate-limits"
    )

    def __init__(self):

        print("=" * 70)
        print("Initializing Gemini LLM")
        print("=" * 70)

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in .env file"
            )

        self.client = genai.Client(
            api_key=api_key
        )

        # Current stable lightweight Gemini model
        self.model_name = "gemini-3.1-flash-lite"

        print(f"Model : {self.model_name}")

    def generate(self, prompt, max_retries=1):
        """
        Calls Gemini and returns the generated text.

        On a 429 / RESOURCE_EXHAUSTED rate-limit error, this retries
        once after a short backoff (using the server-suggested
        retryDelay when available), and if it still fails, returns a
        friendly, user-facing message instead of letting the raw
        ClientError bubble up and crash the Streamlit app.

        Any other API/network error is also caught and turned into a
        readable in-chat message rather than an unhandled traceback.
        """

        attempt = 0

        while True:

            print("\nSending Prompt to Gemini...")
            print("-" * 70)

            try:

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )

                print("Gemini Response Generated Successfully")

                return response.text

            except Exception as e:

                is_rate_limit = self._is_rate_limit_error(e)

                if is_rate_limit and attempt < max_retries:

                    wait_seconds = self._extract_retry_delay(e)

                    print(
                        f"Rate limited by Gemini. Retrying in "
                        f"{wait_seconds:.0f}s "
                        f"(attempt {attempt + 1}/{max_retries})..."
                    )

                    time.sleep(wait_seconds)

                    attempt += 1

                    continue

                if is_rate_limit:

                    print(f"Gemini rate limit error: {e}")

                    return self.RATE_LIMIT_MESSAGE

                print(f"Gemini request failed: {e}")

                return (
                    "⚠️ I couldn't generate an answer because the AI "
                    "service returned an error. Please try again in "
                    "a moment.\n\n"
                    f"Details: {e}"
                )

    # =========================================================
    # ERROR HELPERS
    # =========================================================

    def _is_rate_limit_error(self, error):

        if genai_errors is not None and isinstance(
            error,
            getattr(genai_errors, "ClientError", ())
        ):

            status_code = getattr(error, "code", None) or getattr(
                error, "status_code", None
            )

            if status_code == 429:
                return True

        text = str(error)

        return (
            "RESOURCE_EXHAUSTED" in text
            or "429" in text
            or "rate limit" in text.lower()
            or "quota" in text.lower()
        )

    def _extract_retry_delay(self, error, default_seconds=15):

        match = re.search(
            r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s",
            str(error)
        )

        if match:

            try:
                return float(match.group(1)) + 1

            except ValueError:
                pass

        return default_seconds
