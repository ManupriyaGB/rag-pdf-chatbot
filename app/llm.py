import os

from dotenv import load_dotenv
from google import genai

load_dotenv()


class LLM:

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

    def generate(self, prompt):

        print("\nSending Prompt to Gemini...")
        print("-" * 70)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

        print("Gemini Response Generated Successfully")

        return response.text