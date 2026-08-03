import ollama


class LLM:

    def __init__(self, model_name="llama3.2"):

        self.model_name = model_name

        print("=" * 70)
        print("Initializing LLM...")
        print("=" * 70)

        print(f"Model : {self.model_name}")

    # ----------------------------------------------------
    # Generate Response
    # ----------------------------------------------------

    def generate(self, prompt):

        print("\nSending Prompt to LLM...")

        response = ollama.chat(

            model=self.model_name,

            messages=[

                {
                    "role": "user",
                    "content": prompt
                }

            ]

        )

        answer = response["message"]["content"]

        print("\nLLM Response Generated Successfully")

        print("\n" + "=" * 70)
        print("ANSWER")
        print("=" * 70)

        print(answer)

        return answer