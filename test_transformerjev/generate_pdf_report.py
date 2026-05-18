"""
Generate a PDF report of transformer model evaluations.
Requires: pip install reportlab
"""

import random
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

from test_transformerjev.test_transformerjev_chunk_and_average import (
    DATASET_PATH,
    RANDOM_SEED,
    TOP_K_NEIGHBORS,
    FIELD_WEIGHTS,
    MODEL_NAMES,
    load_records,
    extract_field_texts,
    normalize_topics,
    build_entity_blob,
    normalize_embeddings,
    evaluate_topic_neighbor_purity,
    evaluate_entity_coherence,
    encode_texts_with_chunking,
)

def safe_text(text):
    """
    Replace specific characters that the standard PDF font (Helvetica) doesn't support.
    (e.g., Slovenian characters like č, š, ž).
    """
    replacements = {'č': 'c', 'š': 's', 'ž': 'z', 'Č': 'C', 'Š': 'S', 'Ž': 'Z'}
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def generate_pdf(output_filename=None):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if output_filename is None:
        output_filename = f"eval_report_{timestamp}.pdf"

    # Setup PDF document
    doc = SimpleDocTemplate(output_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Title
    title_style = styles['Heading1']
    elements.append(Paragraph(f"eval_report_{timestamp}", title_style))
    elements.append(Spacer(1, 12))
    
    # Section 1: Embedded Fields & Weights
    h2_style = styles['Heading2']
    elements.append(Paragraph("Embedded Fields and Weights", h2_style))
    
    normal_style = styles['Normal']
    for field, weight in FIELD_WEIGHTS.items():
        elements.append(Paragraph(safe_text(f"<b>{field.capitalize()}</b>: {weight}"), normal_style))
    
    elements.append(Spacer(1, 12))
    
    # Load data
    print("Loading data for PDF generation...")
    rng = random.Random(RANDOM_SEED)
    records = load_records(DATASET_PATH)
    field_texts_list = [extract_field_texts(record) for record in records]
    topic_sets = [set(normalize_topics(record["topics"])) for record in records]
    entity_blob = [build_entity_blob(record) for record in records]
    
    # Evaluate each model
    for model_name in MODEL_NAMES:
        print(f"Evaluating {model_name}...")
        elements.append(Paragraph(safe_text(f"Model: {model_name}"), h2_style))
        
        # Encode
        model = SentenceTransformer(model_name)
        num_records = len(records)
        embeddings_list = []
        
        for field in FIELD_WEIGHTS.keys():
            texts = [ft[field] for ft in field_texts_list]
            field_embeddings = encode_texts_with_chunking(model, texts)
            embeddings_list.append((field, field_embeddings))

        final_embeddings = np.zeros_like(embeddings_list[0][1])
        weight_sums = np.zeros((num_records, 1))

        for field, field_embeddings in embeddings_list:
            weight = FIELD_WEIGHTS.get(field, 0.0)
            if weight == 0.0:
                continue
            has_text_mask = np.array([1.0 if ft[field] else 0.0 for ft in field_texts_list])[:, None]
            final_embeddings += field_embeddings * weight * has_text_mask
            weight_sums += weight * has_text_mask
            
        weight_sums[weight_sums == 0] = 1.0
        embeddings = final_embeddings / weight_sums
        embeddings = normalize_embeddings(embeddings)

        # Topic purity
        purity, sample_size = evaluate_topic_neighbor_purity(embeddings, topic_sets, rng)
        elements.append(Paragraph(f"<b>Topic neighbor purity:</b> {purity:.2f}% (sample size {sample_size}, k={TOP_K_NEIGHBORS})", normal_style))
        elements.append(Spacer(1, 6))
        
        # Entity coherence
        elements.append(Paragraph("<b>Entity Coherence:</b>", normal_style))
        entity_results = evaluate_entity_coherence(embeddings, entity_blob, topic_sets, rng)
        entity_results = sorted(entity_results, key=lambda result: result["entity"].lower())
        
        # Create table data
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
                table_data.append([entity, str(count), f"{intra:.4f}", f"{inter:.4f}", f"{coherence:.4f}"])
        
        # Add table
        t = Table(table_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 10),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 15))

    # Build PDF
    doc.build(elements)
    print(f"PDF successfully generated: {output_filename}")

if __name__ == "__main__":
    generate_pdf()