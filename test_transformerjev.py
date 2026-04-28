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
from sentence_transformers import SentenceTransformer

from read_yaml_data import read_yaml_exclude_urls
from make_test_dataset import NAMED_ENTITIES


# Dataset and evaluation settings
DATASET_PATH = Path("testing_dataset.yaml")
RANDOM_SEED = 42
TOPIC_SAMPLE_SIZE = 20
TOP_K_NEIGHBORS = 10
ENTITY_SAMPLE_SIZE = 5
NEGATIVE_SAMPLE_TRIES = 50

# Model identifiers to evaluate
MODEL_NAMES = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "embaas/sentence-transformers-e5-large-v2",
]


def normalize_topics(topic_value):
    """Normalize the topics field into a list of clean strings."""
    if isinstance(topic_value, str) and topic_value.strip():
        return [topic_value.strip()]
    if isinstance(topic_value, list):
        return [str(item).strip() for item in topic_value if str(item).strip()]
    return []


def build_article_text(record):
    """Build embedding text from lead, titles and keywords (if available)."""
    parts = []

    lead = record.get("lead")
    if isinstance(lead, str):
        parts.append(lead)
    
    titles = record.get("title") 
    if isinstance(titles, list):
        parts.extend([str(t) for t in titles if t is not None])

    keywords = record.get("keywords")
    if isinstance(keywords, list):
        parts.extend([str(k) for k in keywords if k is not None])

    # paragraphs = record.get("paragraphs")
    # if isinstance(paragraphs, list):
    #     parts.extend([str(p) for p in paragraphs if p is not None])
 

    return " ".join(parts).strip()


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
    if len(indices) < 2:
        return None

    group = embeddings[indices]
    similarity_matrix = group @ group.T
    # the upper triangle of the similarity matrix above the diagonal
    # we do not use self similarities and the duplicated similarities (down triangle)
    upper = np.triu_indices(len(indices), k=1)
    # return the average similarity for these upper matrix similarities
    return float(similarity_matrix[upper].mean())


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
    texts = [build_article_text(record) for record in records]
    topic_sets = [set(normalize_topics(record["topics"])) for record in records]
    entity_blob = [build_entity_blob(record) for record in records]

    for model_name in MODEL_NAMES:
        print("\n" + "=" * 80)
        print(f"Model: {model_name}")

        # Testing a new model
        model = SentenceTransformer(model_name)
        # vectorize the article
        embeddings = model.encode(texts)
        embeddings = normalize_embeddings(np.array(embeddings))

        purity, sample_size = evaluate_topic_neighbor_purity(embeddings, topic_sets, rng)
        print(
            f"Topic neighbor purity: {purity:.2f}% "
            f"(sample size {sample_size}, k={TOP_K_NEIGHBORS})"
        )

        entity_results = evaluate_entity_coherence(embeddings, entity_blob, topic_sets, rng)
        print("Entity coherence:")
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
