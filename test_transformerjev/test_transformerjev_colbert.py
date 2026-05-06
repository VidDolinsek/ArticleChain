"""
Evaluate multiple transformer models on the testing dataset.

Two checks are reported per model:
1) Topic neighbor purity: for 20 random articles, what percentage of the top-k
   nearest neighbors share at least one topic with the chosen article.
2) Entity coherence: pick 5 random entities and compare intra-entity similarity
   against inter-entity similarity with random negatives.
"""

from pathlib import Path
import random
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from read_yaml_data import read_yaml_exclude_urls



# Dataset and evaluation settings
DATASET_PATH = Path("testing_dataset.yaml")
RANDOM_SEED = 42
TOPIC_SAMPLE_SIZE = 20
TOP_K_NEIGHBORS = 50
ENTITY_SAMPLE_SIZE = 10
NEGATIVE_SAMPLE_TRIES = 50

# Field weights for combining embeddings
FIELD_WEIGHTS = {
    "lead": 0.5,
    "keywords": 0.5,
    "mention": 0.3,
    "authors": 0.0,
    "paragraphs": 0.1,
}

# Slovenian labels for explicit semantic boundaries
SLO_LABELS = {
    "lead": "Uvod",
    "keywords": "Ključne besede",
    "mention": "Omembe",
    "authors": "Avtorji",
    "paragraphs": "Odstavki",
}

# Model identifiers to evaluate
MODEL_NAMES = [
    # "sentence-transformers/all-MiniLM-L6-v2",
    # "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    # "sentence-transformers/all-mpnet-base-v2",
    # "embaas/sentence-transformers-e5-large-v2",
    # "sentence-transformers/distiluse-base-multilingual-cased-v2",
    # "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    # "sentence-transformers/stsb-xlm-r-multilingual",
    # "intfloat/multilingual-e5-large",
    # "BAAI/bge-large-zh-v1.5",
    "BAAI/bge-m3",
    "BAAI/bge-multilingual-gemma2",
]

NAMED_ENTITIES = [
	"Tadej Pogačar",
	"Donald Trump",
	"Tour de France",
	"Vojna v Ukrajini",
	"Poplave 2023",
	"Olimpijske igre",
	"Janja Garnbret",
	"Borut Pahor",
	"Ljubljanski maraton",
	"Ivana Kobilca",
]

def normalize_topics(topic_value):
    """Normalize the topics field into a list of clean strings."""
    if isinstance(topic_value, str) and topic_value.strip():
        return [topic_value.strip()]
    if isinstance(topic_value, list):
        return [str(item).strip() for item in topic_value if str(item).strip()]
    return []


def extract_field_texts(record):
    """Extract individual fields as separate strings."""
    field_texts = {}

    for field in ("lead", "keywords", "authors", "paragraphs", "mention"):
        value = record.get(field)
        if isinstance(value, list):
            field_texts[field] = " ".join([str(v) for v in value if v is not None]).strip()
        elif isinstance(value, str):
            field_texts[field] = value.strip()
        else:
            field_texts[field] = ""
            
    return field_texts


def build_entity_blob(record):
    """
    Create a lowercased text blob used only for entity matching.

    This combines fields where entity names are likely to appear.
    """
    parts = []

    for field in ("lead",):
        value = record.get(field)
        if isinstance(value, str):
            parts.append(value)

    for field in ("mention", "paragraphs", "keywords", "authors"):
        value = record.get(field)
        if isinstance(value, list):
            parts.extend(str(item) for item in value if item is not None)
        elif isinstance(value, str):
            parts.append(value)

    return " ".join(parts).lower()


def normalize_embeddings(embeddings):
    """L2-normalize embeddings so dot product equals cosine similarity."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms
def compute_colbert_score(emb1, emb2):
    """
    Compute symmetric MaxSim ColBERT score between two multi-vector document representations.
    emb1: (L1, D), emb2: (L2, D)
    """
    if len(emb1) == 0 or len(emb2) == 0:
        return 0.0
        
    sim_matrix = emb1 @ emb2.T
    # Symmetric MaxSim (mean of maxes in both directions)
    score1 = np.mean(np.max(sim_matrix, axis=1))
    score2 = np.mean(np.max(sim_matrix, axis=0))
    return float((score1 + score2) / 2.0)


def encode_colbert_multivector(model, texts, batch_size=4):
    """
    Encode texts into ColBERT-style multi-vectors.
    Returns a list of 2D numpy arrays (token embeddings for each document).
    """
    device = model.device
    all_token_embeddings = []
    tokenizer = model.tokenizer
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding ColBERT Multi-Vectors"):
        batch_texts = texts[i:i+batch_size]
        
        encoded = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=model.max_seq_length,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            # Pass through the SentenceTransformer modules to get token embeddings
            output = model(encoded)
            token_embeddings = output['token_embeddings']
            
        attention_mask = encoded['attention_mask'].bool()
        
        for j in range(len(batch_texts)):
            valid_tokens = token_embeddings[j][attention_mask[j]]
            # L2 normalize each token's vector (Standard for ColBERT)
            norms = torch.norm(valid_tokens, p=2, dim=-1, keepdim=True)
            valid_tokens = valid_tokens / norms.clamp(min=1e-8)
            # Convert to float32 numpy arrays for fast CPU dot products later
            all_token_embeddings.append(valid_tokens.cpu().to(torch.float32).numpy())
            
    return all_token_embeddings


def load_records(dataset_path: Path):
    """Load dataset arrays and build a record list for easy indexing."""
    extraction = read_yaml_exclude_urls(str(dataset_path))
    record_count = len(extraction["id"])

    records = []
    for index in range(record_count):
        records.append(
            {
                "lead": extraction["lead"][index],
                "paragraphs": extraction["paragraphs"][index],
                "topics": extraction["topics"][index],
                "mention": extraction["mention"][index],
                "keywords": extraction["keywords"][index],
                "authors": extraction["authors"][index],
            }
        )

    return records


def share_topic(topic_set_a, topic_set_b):
    """Return True if two topic sets intersect."""
    return len(topic_set_a.intersection(topic_set_b)) > 0


def evaluate_topic_neighbor_purity(embeddings, topic_sets, rng):
    """
    For random articles with topics, compute the percentage of top-k neighbors
    that share at least one topic.
    """
    candidate_indices = [i for i, topics in enumerate(topic_sets) if topics]
    if not candidate_indices:
        return 0.0, 0

    sample_size = min(TOPIC_SAMPLE_SIZE, len(candidate_indices))
    sample_indices = rng.sample(candidate_indices, sample_size) # sample 20 random articles

    total_percentage = 0.0
    for index in sample_indices:
        # compute cosine similarities from embedding at index to all embeddings
        similarities = embeddings @ embeddings[index] # gives us 3000 calculated similarities
        # set similarity to self to a low value (will not be selected)
        similarities[index] = -1.0
        # compute colbert similarities from embedding at index to all embeddings
        similarities = np.zeros(len(embeddings))
        emb_q = embeddings[index]
        for i, emb_d in enumerate(embeddings):
            if i == index:
                similarities[i] = -1.0
                continue
            similarities[i] = compute_colbert_score(emb_q, emb_d)

        top_k = min(TOP_K_NEIGHBORS, len(similarities) - 1)
        # get tok k embeddings in any order
        top_indices = np.argpartition(-similarities, top_k)[:top_k]

        shared = 0
        for neighbor in top_indices:
            if share_topic(topic_sets[index], topic_sets[neighbor]):
                shared += 1

        # calculate the percantage of top_k articles close to the chosen article that share the same topic
        total_percentage += shared / top_k

    # average percantage of articles sharing topics over 20 articles
    average_percentage = (total_percentage / sample_size) * 100.0
    return average_percentage, sample_size


def average_intra_similarity(embeddings, indices):
    """Average cosine similarity within a group of indices (excluding self)."""
    """Average symmetric MaxSim similarity within a group of indices (excluding self)."""
    if len(indices) < 2:
        return None

    group = embeddings[indices]
    similarity_matrix = group @ group.T
    # the upper triangle of the similarity matrix above the diagonal
    # we do not use self similarities and the duplicated similarities (down triangle)
    upper = np.triu_indices(len(indices), k=1)
    # return the average similarity for these upper matrix similarities
    return float(similarity_matrix[upper].mean())
    scores = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            scores.append(compute_colbert_score(embeddings[indices[i]], embeddings[indices[j]]))
            
    return float(np.mean(scores))


def average_inter_similarity(embeddings, indices, topic_sets, entity_blob, entity_name, rng):
    """
    For each index in E, compare it to a random negative article that:
    - does not mention the same entity,
    - does not share the same topic (if a topic exists).
    """
    if not indices:
        return None

    negative_similarities = []
    entity_lower = entity_name.lower()

    # indices contains article that share the same entity
    for index in indices:
        tries = 0
        chosen_negative = None

        # if we do not get the negative article in NEGATIVE_SAMPLE_TRIE we move on to the next index
        while tries < NEGATIVE_SAMPLE_TRIES:
            candidate = rng.randrange(len(embeddings))
            tries += 1

            if candidate in indices:
                continue

            if entity_lower in entity_blob[candidate]:
                continue

            if topic_sets[index] and share_topic(topic_sets[index], topic_sets[candidate]):
                continue
            
            # if the article does not contain the entities or the topics or the article at index "index"
            # we choose it since it should have a low similarity to the chosen article
            chosen_negative = candidate
            break

        if chosen_negative is not None:
            similarity = float(embeddings[index] @ embeddings[chosen_negative])
            similarity = compute_colbert_score(embeddings[index], embeddings[chosen_negative])
            negative_similarities.append(similarity)

    if not negative_similarities:
        return None

    # return the average of the negative similarities
    return float(np.mean(negative_similarities))


def evaluate_entity_coherence(embeddings, entity_blob, topic_sets, rng):
    """Evaluate coherence for a random subset of entities."""
    entities = rng.sample(NAMED_ENTITIES, min(ENTITY_SAMPLE_SIZE, len(NAMED_ENTITIES)))
    results = []

    for entity in entities:
        indices = [i for i, blob in enumerate(entity_blob) if entity.lower() in blob]

        intra = average_intra_similarity(embeddings, indices)
        inter = average_inter_similarity(embeddings, indices, topic_sets, entity_blob, entity, rng)

        if intra is None or inter is None:
            coherence = None
        else:
            coherence = intra - inter

        results.append(
            {
                "entity": entity,
                "count": len(indices),
                "intra": intra,
                "inter": inter,
                "coherence": coherence,
            }
        )

    return results


def main():
    """Load data once, then evaluate each model in sequence."""
    rng = random.Random(RANDOM_SEED)

    records = load_records(DATASET_PATH)
    field_texts_list = [extract_field_texts(record) for record in records]
    topic_sets = [set(normalize_topics(record["topics"])) for record in records]
    entity_blob = [build_entity_blob(record) for record in records]

    for model_name in MODEL_NAMES:
        print("\n" + "=" * 80)
        print(f"Model: {model_name}")

        # Testing a new model
        # Load large models (like Gemma) in float16 to save VRAM
        model_kwargs = {"torch_dtype": torch.float16} if "gemma" in model_name.lower() else {}
        model = SentenceTransformer(model_name, model_kwargs=model_kwargs)
        
        # # Cap the max sequence length to prevent quadratic memory explosion with 8k contexts
        if hasattr(model, "max_seq_length") and model.max_seq_length > 2048:
            model.max_seq_length = 2048
        
        # Concatenate all fields into a single string per record
        texts_to_embed = []
        for ft in field_texts_list:
            parts = []
            for field, weight in FIELD_WEIGHTS.items():
                if weight > 0.0 and ft.get(field):
                    slo_label = SLO_LABELS.get(field, field.capitalize())
                    # Simulate weights by repeating the text (e.g., 0.8 -> 8 repeats)
                    repeats = int(weight * 10)
                    if repeats > 0:
                        parts.extend([f"{slo_label}: {ft[field]}"] * repeats)
            texts_to_embed.append(" | ".join(parts).strip())

        # Embed the concatenated texts
        # print(texts_to_embed[0])
        # Drop batch size to 1 for Gemma, keep standard for smaller models, and add a progress bar
        b_size = 1 if "gemma" in model_name.lower() else 32
        embeddings = model.encode(texts_to_embed, batch_size=b_size, show_progress_bar=True)
        embeddings = normalize_embeddings(np.array(embeddings))
        embeddings = encode_colbert_multivector(model, texts_to_embed, batch_size=b_size)

        purity, sample_size = evaluate_topic_neighbor_purity(embeddings, topic_sets, rng)
        print(
            f"Topic neighbor purity: {purity:.2f}% "
            f"(sample size {sample_size}, k={TOP_K_NEIGHBORS})"
        )

        entity_results = evaluate_entity_coherence(embeddings, entity_blob, topic_sets, rng)
        print("Entity coherence:")
        entity_results = sorted(entity_results, key=lambda result: result["entity"].lower())
        for result in entity_results:
            entity = result["entity"]
            count = result["count"]
            intra = result["intra"]
            inter = result["inter"]
            coherence = result["coherence"]

            if intra is None or inter is None or coherence is None:
                print(f"  {entity}: not enough data for a stable score (count={count})")
                continue

            print(
                f"  {entity}: count={count}, intra={intra:.4f}, "
                f"inter={inter:.4f}, coherence={coherence:.4f}"
            )


if __name__ == "__main__":
    main()
