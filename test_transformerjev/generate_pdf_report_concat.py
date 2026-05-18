"""
Generate a PDF report of transformer model evaluations using
concatenated title + lead + paragraphs as the text to encode.

Requires:
    pip install reportlab
"""

import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch, gc
from sentence_transformers import SentenceTransformer
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

from read_yaml_data import read_yaml

# Named entities you want to repeatedly include in the subset
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

# Dataset and evaluation settings
DATASET_PATH = Path("testing_dataset.yaml")
RANDOM_SEED = 42
TOPIC_SAMPLE_SIZE = 20
TOP_K_NEIGHBORS = 50
ENTITY_SAMPLE_SIZE = 10
NEGATIVE_SAMPLE_TRIES = 50

# Model identifiers to evaluate (same set used in other reports)
MODEL_NAMES = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/stsb-xlm-r-multilingual",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "embaas/sentence-transformers-e5-large-v2",
    #"intfloat/multilingual-e5-large",
    # "BAAI/bge-large-zh-v1.5",
    #"BAAI/bge-m3",
    # "BAAI/bge-multilingual-gemma2",
]


def safe_text(text: str) -> str:
    """
    Replace characters unsupported by the default PDF font (Helvetica).
    This keeps the report generation robust across environments.
    """
    replacements = {
        "č": "c",
        "š": "s",
        "ž": "z",
        "Č": "C",
        "Š": "S",
        "Ž": "Z",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def normalize_topics(topic_value):
    """Normalize the topics field into a list of clean strings."""
    if isinstance(topic_value, str) and topic_value.strip():
        return [topic_value.strip()]
    if isinstance(topic_value, list):
        return [str(item).strip() for item in topic_value if str(item).strip()]
    return []


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings so dot product equals cosine similarity."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms


def load_records(dataset_path: Path):
    """Load dataset arrays and build a record list for easy indexing."""
    extraction = read_yaml(str(dataset_path))
    record_count = len(extraction["id"])

    records = []
    for index in range(record_count):
        records.append(
            {
                "title": extraction["title"][index],
                "lead": extraction["lead"][index],
                "paragraphs": extraction["paragraphs"][index],
                "topics": extraction["topics"][index],
                "mention": extraction["mention"][index],
                "keywords": extraction["keywords"][index],
                "authors": extraction["authors"][index],
            }
        )

    return records


def _as_text(value) -> str:
    """Convert a field value to a single string."""
    if isinstance(value, list):
        return " ".join(str(item) for item in value if item is not None).strip()
    if isinstance(value, str):
        return value.strip()
    return ""


def build_concat_text(record) -> str:
    """
    Build the text to embed by concatenating title, lead, and paragraphs.
    Paragraphs are joined to preserve article flow.
    """
    title = _as_text(record.get("title"))
    lead = _as_text(record.get("lead"))

    paragraphs = record.get("paragraphs")
    if isinstance(paragraphs, list):
        paragraph_text = "\n\n".join(str(p) for p in paragraphs if p is not None).strip()
    else:
        paragraph_text = _as_text(paragraphs)

    parts = [part for part in (title, lead, paragraph_text) if part]
    return "\n\n".join(parts).strip()


def build_entity_blob(record) -> str:
    """
    Create a lowercased text blob used only for entity matching.
    This combines fields where entity names are likely to appear.
    """
    parts = []

    lead = record.get("lead")
    if isinstance(lead, str):
        parts.append(lead)

    for field in ("mention", "paragraphs", "keywords", "authors"):
        value = record.get(field)
        if isinstance(value, list):
            parts.extend(str(item) for item in value if item is not None)
        elif isinstance(value, str):
            parts.append(value)

    return " ".join(parts).lower()


def share_topic(topic_set_a, topic_set_b) -> bool:
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

    # sample as much topics as we want, make sure that the number does not exceed the number of topics we have
    sample_size = min(TOPIC_SAMPLE_SIZE, len(candidate_indices)) 
    sample_indices = rng.sample(candidate_indices, sample_size)

    total_percentage = 0.0
    for index in sample_indices:
        similarities = embeddings @ embeddings[index]
        similarities[index] = -1.0

        top_k = min(TOP_K_NEIGHBORS, len(similarities) - 1)
        top_indices = np.argpartition(-similarities, top_k)[:top_k]

        shared = 0
        for neighbor in top_indices:
            if share_topic(topic_sets[index], topic_sets[neighbor]):
                shared += 1

        total_percentage += shared / top_k

    average_percentage = (total_percentage / sample_size) * 100.0
    return average_percentage, sample_size


def average_intra_similarity(embeddings, indices):
    """Average cosine similarity within a group of indices (excluding self)."""
    if len(indices) < 2:
        return None

    group = embeddings[indices]
    similarity_matrix = group @ group.T
    upper = np.triu_indices(len(indices), k=1)
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
            similarity = float(embeddings[index] @ embeddings[chosen_negative])
            negative_similarities.append(similarity)

    if not negative_similarities:
        return None

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


def generate_pdf(output_filename=None):
    """Generate the PDF report for concatenated title + lead + paragraphs."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if output_filename is None:
        output_filename = f"eval_report_concat_{timestamp}.pdf"

    doc = SimpleDocTemplate(output_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    title_style = styles["Heading1"]
    elements.append(Paragraph(f"eval_report_concat_{timestamp}", title_style))
    elements.append(Spacer(1, 12))

    h2_style = styles["Heading2"]
    normal_style = styles["Normal"]

    # Describe the embedding text used in this report.
    elements.append(Paragraph("Encoded Text", h2_style))
    elements.append(Paragraph("title + lead + paragraphs", normal_style))
    elements.append(Spacer(1, 12))

    print("Loading data for PDF generation...")
    rng = random.Random(RANDOM_SEED)
    records = load_records(DATASET_PATH)

    texts = [build_concat_text(record) for record in records]
    topic_sets = [set(normalize_topics(record["topics"])) for record in records]
    entity_blob = [build_entity_blob(record) for record in records]

    for model_name in MODEL_NAMES:
        print(f"Evaluating {model_name}...")
        elements.append(Paragraph(safe_text(f"Model: {model_name}"), h2_style))

        model = SentenceTransformer(model_name)
        # Let the model truncate when the input exceeds max length.
        embeddings = model.encode(texts)
        embeddings = normalize_embeddings(np.array(embeddings))

        purity, sample_size = evaluate_topic_neighbor_purity(embeddings, topic_sets, rng)
        elements.append(
            Paragraph(
                f"<b>Topic neighbor purity (k={TOP_K_NEIGHBORS}):</b> {purity:.2f}% "
                f"(sample size {sample_size}, k={TOP_K_NEIGHBORS})",
                normal_style,
            )
        )
        elements.append(Spacer(1, 6))

        elements.append(Paragraph("<b>Entity Coherence:</b>", normal_style))
        entity_results = evaluate_entity_coherence(embeddings, entity_blob, topic_sets, rng)
        entity_results = sorted(entity_results, key=lambda result: result["entity"].lower())

        coherence_values = [
            result["coherence"]
            for result in entity_results
            if result["coherence"] is not None
        ]
        mean_coherence = float(np.mean(coherence_values)) if coherence_values else None

        table_data = [["Entity", "Count", "Intra", "Inter", "Coherence"]]
        for result in entity_results:
            entity = safe_text(result["entity"])
            count = result["count"]
            intra = result["intra"]
            inter = result["inter"]
            coherence = result["coherence"]

            if intra is None or inter is None or coherence is None:
                table_data.append([entity, str(count), "N/A", "N/A", "N/A"])
            else:
                table_data.append(
                    [entity, str(count), f"{intra:.4f}", f"{inter:.4f}", f"{coherence:.4f}"]
                )

        table = Table(table_data)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        elements.append(table)
        if mean_coherence is None:
            elements.append(Paragraph("<b>Mean coherence:</b> N/A", normal_style))
        else:
            elements.append(Paragraph(f"<b>Mean coherence:</b> {mean_coherence:.4f}", normal_style))
        elements.append(Spacer(1, 15))
        
        # cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()

    doc.build(elements)
    print(f"PDF successfully generated: {output_filename}")


if __name__ == "__main__":
    generate_pdf()

    
