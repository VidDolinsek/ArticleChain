"""
Create a balanced testing subset from mmc.yaml.

The goal is a random subset of about 3000 articles that still keeps useful structure:
1. multiple articles for each named entity,
2. multiple articles from each year,
3. multiple articles from selected topic(s).

The output is written to testing_dataset.yaml.
"""

from pathlib import Path
import random
import yaml


# Input/output files
INPUT_YAML = Path("mmc.yaml")
OUTPUT_YAML = Path("testing_dataset.yaml")

# Main sampling targets
TARGET_ARTICLE_COUNT = 3000
RANDOM_SEED = 42

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

# For every named entity we try to include a random amount in this interval
MIN_PER_ENTITY = 30
MAX_PER_ENTITY = 50

# Guarantee at least this many articles per year (if available)
MIN_PER_YEAR = 120

# Guarantee repeated topic coverage for these topic names (if available)
FOCUS_TOPICS = ["zabava-in-slog", "gospodarstvo", "okolje", "kultura", "znanost-in-tehnologija", "sport", "slovenija", "svet", "crna-kronika"]
MIN_PER_FOCUS_TOPIC = 120


def load_articles(path: Path):
	"""Load and validate YAML content as a list of article dictionaries."""
	with path.open("r", encoding="utf-8") as file:
		data = yaml.safe_load(file)

	if not isinstance(data, list):
		raise ValueError("Expected top-level YAML structure to be a list of articles.")

	return data


def build_search_blob(article):
	"""
	Build one lowercase text blob per article for simple entity matching.

	We merge fields where names usually appear: title, lead, mention, paragraphs,
	keywords, and authors.
	"""
	parts = []

	for field in ("title", "lead"):
		value = article.get(field)
		if isinstance(value, str):
			parts.append(value)

	for field in ("mention", "paragraphs", "keywords", "authors"):
		value = article.get(field)
		if isinstance(value, list):
			parts.extend(str(item) for item in value if item is not None)
		elif isinstance(value, str):
			parts.append(value)

	return " ".join(parts).lower()


def extract_year(article):
	"""Extract YYYY from date field like '2024-11-19T15:30:02'."""
	date_value = article.get("date")
	if isinstance(date_value, str) and len(date_value) >= 4 and date_value[:4].isdigit():
		return date_value[:4]
	return None


def extract_topics(article):
	"""Return article topics as a normalized list of strings."""
	topic_value = article.get("topics")

	if isinstance(topic_value, str) and topic_value.strip():
		return [topic_value.strip()]

	if isinstance(topic_value, list):
		return [str(item).strip() for item in topic_value if str(item).strip()]

	return []


def add_random_indices(candidates, target_count, selected_indices, rng):
	"""
	Add random, non-duplicate candidates to selected_indices.

	Returns the number of indices actually added.
	"""
	available = [index for index in candidates if index not in selected_indices]
	amount_to_add = min(target_count, len(available))

	if amount_to_add > 0:
		sampled = rng.sample(available, amount_to_add)
		selected_indices.update(sampled)

	return amount_to_add


def build_subset(articles):
	"""Build a balanced subset according to entity/year/topic constraints."""
	rng = random.Random(RANDOM_SEED)

	search_blobs = [build_search_blob(article) for article in articles]
	years = [extract_year(article) for article in articles]
	topics_per_article = [extract_topics(article) for article in articles]

	selected_indices = set()

	# 1) Entity coverage: include 30-50 random articles for each entity (if available).
	for entity in NAMED_ENTITIES:
		matching_indices = [
			index
			for index, blob in enumerate(search_blobs)
			if entity.lower() in blob
		]

		requested = rng.randint(MIN_PER_ENTITY, MAX_PER_ENTITY)
		added = add_random_indices(matching_indices, requested, selected_indices, rng)
		print(f"Entity '{entity}': requested {requested}, added {added}.")

	# 2) Year coverage: ensure each available year has repeated representation.
	indices_by_year = {}
	for index, year in enumerate(years):
		if year is None:
			continue
		indices_by_year.setdefault(year, []).append(index)

	for year in sorted(indices_by_year.keys()):
		added = add_random_indices(indices_by_year[year], MIN_PER_YEAR, selected_indices, rng)
		print(f"Year '{year}': requested {MIN_PER_YEAR}, added {added}.")

	# 3) Topic coverage: ensure repeated representation for selected focus topics.
	indices_by_topic = {}
	for index, article_topics in enumerate(topics_per_article):
		for topic in article_topics:
			indices_by_topic.setdefault(topic, []).append(index)

	for topic in FOCUS_TOPICS:
		matching_indices = indices_by_topic.get(topic, [])
		added = add_random_indices(matching_indices, MIN_PER_FOCUS_TOPIC, selected_indices, rng)
		print(f"Topic '{topic}': requested {MIN_PER_FOCUS_TOPIC}, added {added}.")

	# 4) Fill the remaining space randomly until target size is reached.
	max_possible = len(articles)
	final_target = min(TARGET_ARTICLE_COUNT, max_possible)

	if len(selected_indices) < final_target:
		remaining_indices = [
			index for index in range(max_possible) if index not in selected_indices
		]
		missing_count = final_target - len(selected_indices)
		add_random_indices(remaining_indices, missing_count, selected_indices, rng)

	final_indices = list(selected_indices)
	rng.shuffle(final_indices)

	subset = [articles[index] for index in final_indices]
	return subset


def save_subset(subset, path: Path):
	"""Save subset articles to YAML while preserving Unicode characters."""
	with path.open("w", encoding="utf-8") as file:
		yaml.safe_dump(
			subset,
			file,
			allow_unicode=True,
			sort_keys=False,
			default_flow_style=False,
		)


def main():
	"""Load full dataset, build subset, and save testing_dataset.yaml."""
	articles = load_articles(INPUT_YAML)
	subset = build_subset(articles)
	save_subset(subset, OUTPUT_YAML)

	print(f"Created subset with {len(subset)} articles.")
	print(f"Saved to: {OUTPUT_YAML}")


if __name__ == "__main__":
	main()