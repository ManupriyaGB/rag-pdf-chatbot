class PromptBuilder:

    def __init__(self):

        print("=" * 70)
        print("Prompt Builder Initialized")
        print("=" * 70)

    def build_prompt(self, retrieved_chunks, query):

        print("\nBuilding Prompt...")

        context = "\n\n".join(retrieved_chunks)

        prompt = f"""
You are an AI assistant.

Use ONLY the information provided in the context below.

If the answer cannot be found in the context,
reply with:

"I don't know based on the provided document."

------------------------------------------------------------
CONTEXT
------------------------------------------------------------

{context}

------------------------------------------------------------
QUESTION
------------------------------------------------------------

{query}

------------------------------------------------------------
ANSWER
------------------------------------------------------------
"""

        print("\nPrompt Created Successfully")

        print("\n" + "=" * 70)
        print("FINAL PROMPT")
        print("=" * 70)

        print(prompt)

        return prompt