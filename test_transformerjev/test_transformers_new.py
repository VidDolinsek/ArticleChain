"""
Evaluate BSC-LT/MrBERT and bert-base-multilingual variants on the testing dataset.

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

from test_transformerjev.read_yaml_data import read_yaml


# Dataset and evaluation settings
DATASET_PATH = Path("testing_dataset.yaml")
RANDOM_SEED = 42
TOPIC_SAMPLE_SIZE = 20
TOP_K_NEIGHBORS = 50
ENTITY_SAMPLE_SIZE = 10
NEGATIVE_SAMPLE_TRIES = 50

# Field weights for combining embeddings
FIELD_WEIGHTS = {
    "lead": 1.0,
    "paragraphs": 0.8,
    "keywords": 0.5,
    "mention": 0.3,
    "authors": 0.0,
}

MODEL_NAMES = [
    "BSC-LT/MrBERT",
    "google-bert/bert-base-multilingual-uncased",
    "google-bert/bert-base-multilingual-cased",
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
    if isinstance(topic_value, str) and topic_value.strip():
        return [topic_value.strip()]
    if isinstance(topic_value, list):
        return [str(item).strip() for item in topic_value if str(item).strip()]
    return []


def extract_field_texts(record):
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
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms


def load_records(dataset_path: Path):
    extraction = read_yaml(str(dataset_path))
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
    return len(topic_set_a.intersection(topic_set_b)) > 0


def evaluate_topic_neighbor_purity(embeddings, topic_sets, rng):
    candidate_indices = [i for i, topics in enumerate(topic_sets) if topics]
    if not candidate_indices:
        return 0.0, 0

    sample_size = min(TOPIC_SAMPLE_SIZE, len(candidate_indices))
    sample_indices = rng.sample(candidate_indices, sample_size)

    total_percentage = 0.0
    for index in sample_indices:
        similarities = embeddings @ embeddings[index]
        similarities[index] = -1.0

        top_k = min(TOP_K_NEIGHBORS, len(similarities) - 1)
        top_indices = np.argpartition(-similarities, top_k)[:top_k]

        shared = sum(1 for neighbor in top_indices if share_topic(topic_sets[index], topic_sets[neighbor]))
        total_percentage += shared / top_k

    return (total_percentage / sample_size) * 100.0, sample_size


def average_intra_similarity(embeddings, indices):
    if len(indices) < 2:
        return None
    group = embeddings[indices]
    similarity_matrix = group @ group.T
    upper = np.triu_indices(len(indices), k=1)
    return float(similarity_matrix[upper].mean())


def average_inter_similarity(embeddings, indices, topic_sets, entity_blob, entity_name, rng):
    if not indices:
        return None

    negative_similarities = []
    entity_lower = entity_name.lower()

    for index in indices:
        tries = 0
        chosen_negative = None

        while tries < NEGATIVE_SAMPLE_TRIES:
            candidate = rng.randrange(len(embeddings))
            tries += 1

            if candidate in indices:
                continue
            if entity_lower in entity_blob[candidate]:
                continue
            if topic_sets[index] and share_topic(topic_sets[index], topic_sets[candidate]):
                continue

            chosen_negative = candidate
            break

        if chosen_negative is not None:
            negative_similarities.append(float(embeddings[index] @ embeddings[chosen_negative]))

    if not negative_similarities:
        return None
    return float(np.mean(negative_similarities))


def evaluate_entity_coherence(embeddings, entity_blob, topic_sets, rng):
    entities = rng.sample(NAMED_ENTITIES, min(ENTITY_SAMPLE_SIZE, len(NAMED_ENTITIES)))
    results = []

    for entity in entities:
        indices = [i for i, blob in enumerate(entity_blob) if entity.lower() in blob]
        intra = average_intra_similarity(embeddings, indices)
        inter = average_inter_similarity(embeddings, indices, topic_sets, entity_blob, entity, rng)

        coherence = (intra - inter) if (intra is not None and inter is not None) else None
        results.append({"entity": entity, "count": len(indices), "intra": intra, "inter": inter, "coherence": coherence})

    return results


def main():
    rng = random.Random(RANDOM_SEED)

    records = load_records(DATASET_PATH)
    field_texts_list = [extract_field_texts(record) for record in records]
    topic_sets = [set(normalize_topics(record["topics"])) for record in records]
    entity_blob = [build_entity_blob(record) for record in records]

    num_records = len(records)

    for model_name in MODEL_NAMES:
        print("\n" + "=" * 80)
        print(f"Model: {model_name}")

        model = SentenceTransformer(model_name)

        embeddings_list = []
        for field in FIELD_WEIGHTS:
            texts = [ft[field] for ft in field_texts_list]
            field_embeddings = model.encode(texts)
            embeddings_list.append((field, field_embeddings))

        final_embeddings = np.zeros_like(embeddings_list[0][1])
        weight_sums = np.zeros((num_records, 1))

        for field, field_embeddings in embeddings_list:
            weight = FIELD_WEIGHTS[field]
            if weight == 0.0:
                continue
            has_text_mask = np.array([1.0 if ft[field] else 0.0 for ft in field_texts_list])[:, None]
            final_embeddings += field_embeddings * weight * has_text_mask
            weight_sums += weight * has_text_mask

        weight_sums[weight_sums == 0] = 1.0
        embeddings = normalize_embeddings(final_embeddings / weight_sums)

        purity, sample_size = evaluate_topic_neighbor_purity(embeddings, topic_sets, rng)
        print(f"Topic neighbor purity: {purity:.2f}% (sample size {sample_size}, k={TOP_K_NEIGHBORS})")

        entity_results = sorted(
            evaluate_entity_coherence(embeddings, entity_blob, topic_sets, rng),
            key=lambda r: r["entity"].lower(),
        )
        print("Entity coherence:")

        mean_coherence = 0
        divide = len(entity_results)
        for result in entity_results:
            entity, count, intra, inter, coherence = (
                result["entity"], result["count"], result["intra"], result["inter"], result["coherence"],
            )
            if coherence is None:
                print(f"  {entity}: not enough data for a stable score (count={count})")
                divide -= 1
                continue
            mean_coherence += coherence
            print(f"  {entity}: count={count}, intra={intra:.4f}, inter={inter:.4f}, coherence={coherence:.4f}")

        if divide > 0:
            print(f"mean_coherence = {mean_coherence / divide}")


if __name__ == "__main__":
    main()
