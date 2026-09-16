import os

from google import genai
from dotenv import load_dotenv


load_dotenv()


class GeminiGenerator:

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash"
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

    def generate(
        self,
        prompt: str
    ) -> str:

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

        if not response.text:
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        return response.text