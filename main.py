from openai import OpenAI
from pathlib import Path
import pickle
import json
import uuid
import math

class Memory:
    def __init__(self, openAI_client: OpenAI, model_name:str,save_path=None):
        self.client = openAI_client
        self.model_name = model_name

        if save_path is not None and save_path.exists():
            try:
                with save_path.open("rb") as f:
                    self.nodes = pickle.load(f)
            except Exception as e:
                print(f"Error loading memory from {save_path}: {e}")
                self.nodes = []
        else:
            self.nodes = []

    def search_in_memory(self, message: str):
        message = f"User says: {message}"
        memory_output_for_message = []
        for sentence in self._separate_sentences(message):
            best_match = self._get_best_match_for_sentence(sentence)
            if best_match[1] >= .5:
                memory_output_for_message.append(best_match[0])

        return memory_output_for_message

    def add_to_memory(self, message):
        for sentence in self._separate_sentences(message):
            embedding = self.get_embedding(sentence)
            similarity_score = self._get_best_match_for_sentence(embedding)[1]
            if similarity_score <= 0.8:
                repsonse = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": f"""
                        Summarize every fact of this sentence in a single 5-10 words sentence about the user.
                        Example: message: "Damn my sister is annoying", summary: "User has an annoying sister".
                        Do Not talk about the user's sister no matter what.
                        If there is not relevant information (e.g. no information; greetings; instructions; questions; etc), just say "None".
                        Short-lived facts you MUST IGNORE (output None):
                        - Current location or transport: "I'm in a train right now", "I'm at the gym".
                        - Current activity: "I'm writing an email", "I'm cooking dinner".
                        - Current mood or energy: "I'm tired today", "I'm stressed right now".
                        - Time-bound things: "today", "right now", "this week", "tonight", "tomorrow", "yesterday".
                        - One-off events or micro-context: "my train is late", "my meeting just ended".


                        message: {sentence}
                        """}]
                )
                to_store = repsonse.choices[0].message.content
                if to_store == "None":
                    continue

                emb = self.get_embedding(to_store)
                node, similarity_score = self._get_best_match_for_sentence(emb)
                if similarity_score >= 0.85:
                    # hard copies
                    continue
                elif similarity_score >= 0.75:
                    #soft copies, merge
                    continue

                elif similarity_score >= 0.55:
                    #related, add as neighbors
                    new_node = Node(to_store, emb, [node])
                    self.nodes.append(new_node)
                    node.add_neighbor(new_node)
                else:
                    self.nodes.append(Node(to_store, emb))

    def load(self, save_path: Path):
        with save_path.open("rb") as f:
            self.nodes = pickle.load(f)

    def save(self, save_path: Path):
        with save_path.open("wb") as f:
            pickle.dump(self.nodes, f)

    def _separate_sentences(self, message: str):
        raw = message.split(".")
        sentences = [s.strip() for s in raw if len(s.strip()) > 1]
        return sentences

    def _get_best_match_for_sentence(self, sentence: str):
        if type(sentence) == str:
            message_embedding = self.get_embedding(sentence)
        else:
            message_embedding = sentence
        best_node = None
        best_score = -1.0

        for node in self.nodes:
            node_embedding = node.get_embedding()
            score = self.cosine_similarity(message_embedding, node_embedding)
            if score > best_score:
                best_score = score
                best_node = node

        return best_node, best_score

    def get_embedding(self, text: str) -> list[float]:
        resp = self.client.embeddings.create(
            model="EmbeddingGemma",
            input=text,
        )
        return resp.data[0].embedding

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        denom = norm_a * norm_b
        if denom == 0:
            return 0.0
        return dot / denom

    def to_json(self):
        json_nodes = {"nodes": [node.to_json() for node in self.nodes]}
        Path(f"memory_export.json").write_text(json.dumps(json_nodes, indent=2))


class Node:
    def __init__(self, title, embedding, neighbors=None):
        if neighbors is None:
            self.neighbors = []
        else:
            self.neighbors = neighbors
        self.title = title
        self.id = str(uuid.uuid4())
        self.embedding = embedding

    def to_json(self):
        return {
            "title": self.title,
            "id": self.id,
            "neighbors": [n.id for n in self.neighbors]
        }

    def get_embedding(self):
        return self.embedding

    def get_id(self):
        return self.id

    def add_neighbor(self, neighbor_node):
        self.neighbors.append(neighbor_node)

client = OpenAI(
    api_key="sk-local-admin",
    base_url=""
)
MODEL = "qwen3-vl-30b"
PATH = Path("./memory.pkl")
mem = Memory(client, MODEL, save_path=PATH)

# message = "I don't know what budget I have for my servers right now"
# message = "I'm thinking of buying a new server for my rack."
# message = "My current budget for my homelab is around 500€"
message = "I'm in the mountains, on a hike right now !"
mem.add_to_memory(message)

mem.save(PATH)
mem.to_json()

# for m in mem.search_in_memory("I don't have a lot of money right now"):
    # print(m.title)
