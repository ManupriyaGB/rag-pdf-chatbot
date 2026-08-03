from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingModel:

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name
        )

    def create_embeddings(self, chunks):

        print("=" * 60)
        print("Embedding Model Started")
        print("=" * 60)

        print(f"Total Chunks Received : {len(chunks)}")

        print("\nFirst Chunk:")
        print(chunks[0])

        print("\nGenerating Embeddings...\n")

        vectors = self.embeddings.embed_documents(chunks)

        print("Embedding Generation Completed\n")

        print(f"Total Embeddings : {len(vectors)}")

        print(f"Embedding Dimension : {len(vectors[0])}")

        print("\nFirst 10 Values of First Embedding:\n")

        print(vectors[0][:10])

        return vectors

    def create_query_embedding(self, query):
        """
        Convert user query into embedding.
        """

        vector = self.embeddings.embed_query(query)

        return vector

# from transformers import AutoTokenizer, AutoModel
# import torch


# class EmbeddingModel:

#     def __init__(self,
#                  model_name="sentence-transformers/all-MiniLM-L6-v2"):

#         print("=" * 70)
#         print("Loading Embedding Model...")
#         print("=" * 70)

#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(model_name)

#         print("Embedding Model Loaded Successfully")
#         print()

#     def create_embeddings(self, chunks):

#         all_embeddings = []

#         print("=" * 70)
#         print("Embedding Generation Started")
#         print("=" * 70)

#         print(f"Total Chunks Received : {len(chunks)}")

#         for index, chunk in enumerate(chunks):

#             print("\n" + "=" * 70)
#             print(f"Processing Chunk : {index + 1}")
#             print("=" * 70)

#             print("\nChunk Text:\n")
#             print(chunk)

#             # --------------------------------------------------
#             # STEP 1 : TOKENIZATION
#             # --------------------------------------------------
#             tokens = self.tokenizer.tokenize(chunk)

#             print("\nSTEP 1 : TOKENIZATION")
#             print("-" * 50)

#             print(f"Total Tokens : {len(tokens)}")
#             print("\nFirst 20 Tokens :")
#             print(tokens[:20])

#             # --------------------------------------------------
#             # STEP 2 : TOKEN IDS
#             # --------------------------------------------------
#             token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

#             print("\nSTEP 2 : TOKEN IDS")
#             print("-" * 50)

#             print(f"Total Token IDs : {len(token_ids)}")
#             print("\nFirst 20 Token IDs :")
#             print(token_ids[:20])

#             # --------------------------------------------------
#             # STEP 3 : MODEL INPUT
#             # --------------------------------------------------
#             inputs = self.tokenizer(
#                 chunk,
#                 return_tensors="pt",
#                 truncation=True,
#                 max_length=512
#             )

#             print("\nSTEP 3 : MODEL INPUT")
#             print("-" * 50)

#             print("Input IDs Shape :", inputs["input_ids"].shape)
#             print("Attention Mask Shape :", inputs["attention_mask"].shape)

#             # --------------------------------------------------
#             # STEP 4 : TRANSFORMER
#             # --------------------------------------------------
#             with torch.no_grad():
#                 outputs = self.model(**inputs)

#             hidden_states = outputs.last_hidden_state

#             print("\nSTEP 4 : TRANSFORMER OUTPUT")
#             print("-" * 50)

#             print("Hidden State Shape :", hidden_states.shape)

#             print("""
# Meaning:

# Batch Size      -> Number of sentences processed together.

# Number of Tokens -> Tokens in this chunk.

# Embedding Size   -> Vector length for each token.
# """)

#             # --------------------------------------------------
#             # STEP 5 : MEAN POOLING
#             # --------------------------------------------------
#             embedding = hidden_states.mean(dim=1)

#             print("\nSTEP 5 : MEAN POOLING")
#             print("-" * 50)

#             print("Sentence Embedding Shape :", embedding.shape)

#             print("\nEmbedding Dimension :", embedding.shape[1])

#             print("\nFirst 10 Values of Embedding:\n")

#             print(embedding[0][:10])

#             all_embeddings.append(embedding.squeeze().numpy())

#         print("\n" + "=" * 70)
#         print("Embedding Generation Completed")
#         print("=" * 70)

#         print(f"Total Embeddings Created : {len(all_embeddings)}")

#         print(
#             f"Dimension of Each Embedding : {len(all_embeddings[0])}"
#         )

#         return all_embeddings

#     def create_query_embedding(self, query):

#         print("=" * 70)
#         print("Generating Query Embedding")
#         print("=" * 70)

#         print("Query :")
#         print(query)

#         inputs = self.tokenizer(
#             query,
#             return_tensors="pt",
#             truncation=True,
#             max_length=512
#         )

#         with torch.no_grad():
#             outputs = self.model(**inputs)

#         embedding = outputs.last_hidden_state.mean(dim=1)

#         print("\nQuery Embedding Shape :", embedding.shape)

#         print("\nEmbedding Dimension :", embedding.shape[1])

#         return embedding.squeeze().numpy()