import os
import time

from google import genai
from dotenv import load_dotenv


load_dotenv()


class GeminiServiceError(Exception):
    """Raised when Gemini is temporarily unavailable."""

    pass


class GeminiGenerator:

    def __init__(
        self,
        model_name: str = "gemini-3.5-flash-lite",
        max_retries: int = 3
    ):

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured."
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.model_name = model_name
        self.max_retries = max_retries

    def generate(
        self,
        prompt: str
    ) -> str:

        for attempt in range(self.max_retries + 1):

            try:

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )

                if not response.text:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                return response.text

            except Exception as error:

                error_message = str(error)

                is_temporary_error = (
                    "503" in error_message
                    or "429" in error_message
                    or "UNAVAILABLE" in error_message
                    or "RESOURCE_EXHAUSTED" in error_message
                )

                if not is_temporary_error:

                    raise

                if attempt == self.max_retries:

                    raise GeminiServiceError(
                        "Gemini is temporarily unavailable "
                        "after multiple retries."
                    ) from error

                wait_seconds = 2 ** attempt

                print(
                    f"Gemini temporary error. "
                    f"Retrying in {wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)