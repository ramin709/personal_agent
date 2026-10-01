from llama_index.core import Document, StorageContext, load_index_from_storage, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from sentence_transformers import CrossEncoder
from pathlib import Path


class LongTermMemory:
    def __init__(self, storage_path="/memory/storage"):
        self.storage_dir = Path(storage_path)

        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        if self.storage_dir.exists():
            storage_context = StorageContext.from_defaults(persist_dir=str(self.storage_dir))
            self.index = load_index_from_storage(storage_context, embed_model=self.embed_model)
            print("directory found, index found")
        else:
            self.index = VectorStoreIndex.from_documents([], embed_model= self.embed_model)
            self.index.storage_context.persist(persist_dir= str(self.storage_dir))
            print("Index has been initialized")
        self.reranker = CrossEncoder("BAAI/bge-reranker-base")


    def store(self, memory: dict):

        if memory["memory_type"] == "experience":
            text = memory["metadata"]["goal"]

            metadata = {
                "memory_type": "experience",
                "source": memory["source"],
                "lesson": memory["content"],
                **memory["metadata"]
            }

        else:
            text = memory["content"]

            metadata = {
                "memory_type": memory["memory_type"],
                "source": memory["source"],
                **memory["metadata"]
            }

        doc = Document(
            text=text,
            metadata=metadata
        )

        self.index.insert(doc)

        self.index.storage_context.persist(
            persist_dir=str(self.storage_dir)
        )


    def retrieve(self, query: str, memoryKey: str, retrievalTopK: int = 5, finalTopK: int = 2):

        filters = MetadataFilters(
            filters=[
                ExactMatchFilter(
                    key="memory_type",
                    value=memoryKey
                )
            ]
        )

        retriever = self.index.as_retriever(
            similarity_top_k=retrievalTopK,
            filters=filters
        )

        nodes = retriever.retrieve(query)

        if memoryKey == "experience":

            pairs = [
                (
                    query,
                    node.metadata["goal"]
                )
                for node in nodes
            ]

        else:

            pairs = [
                (
                    query,
                    node.text
                )
                for node in nodes
            ]

        scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(nodes, scores),
            key=lambda x: x[1],
            reverse=True
        )

        memories = []

        for node, score in ranked:

            if memoryKey == "experience":
                content = node.metadata["lesson"]
            else:
                content = node.text

            memories.append({
                "content": content,
                "score": float(score),
                "metadata": node.metadata
            })

        return memories[:finalTopK]