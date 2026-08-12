class PromptBuilder:

    def __init__(self):

        print("=" * 70)
        print("Prompt Builder Initialized")
        print("=" * 70)

    def build_prompt(
        self,
        retrieved_chunks,
        query,
        chat_history=None
    ):

        context = "\n\n".join(
            retrieved_chunks
        )

        history_text = ""

        if chat_history:

            history_text = "\nPrevious Conversation:\n"

            for message in chat_history[-6:]:

                role = message["role"]
                content = message["content"]

                history_text += (
                    f"{role}: {content}\n"
                )

        prompt = f"""
You are an expert document question-answering assistant.

Your job is to answer the user's question using the
provided document context.

IMPORTANT RULES:

1. Use the document context as the primary source.
2. Give a clear and complete answer.
3. Do not simply repeat the user's question.
4. Combine information from multiple retrieved chunks when useful.
5. If the answer is not available in the documents, say:
   "I could not find the answer in the provided documents."
6. For follow-up questions, use the previous conversation
   to understand what the user is referring to.

{history_text}

Document Context:
-----------------
{context}
-----------------

Current Question:
{query}

Answer:
"""

        print("\nPrompt Created Successfully")

        return prompt