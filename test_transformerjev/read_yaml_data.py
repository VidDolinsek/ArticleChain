import yaml
import numpy as np
import pandas as pd


def read_yaml(yaml_file):
    """
    Read YAML file and extract all article fields into numpy arrays and a DataFrame.
    The first img URL from figures is extracted as the article thumbnail.

    Args:
        yaml_file (str): Path to the YAML file

    Returns:
        dict: numpy arrays keyed by field name, plus a 'dataframe' key
    """

    with open(yaml_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    storage = {
        '_id':        [],
        'id':         [],
        'title':      [],
        'topics':     [],
        'authors':    [],
        'date':       [],
        'keywords':   [],
        'lead':       [],
        'mention':    [],
        'paragraphs': [],
        'captions':   [],
        'sources':    [],
        'url':        [],
        'thumbnail':  [],  # first img URL from figures
    }

    for entry in data:
        figures   = entry.get('figures') or []
        thumbnail = figures[0].get('img') if figures else None

        storage['_id'].append(entry.get('_id'))
        storage['id'].append(entry.get('id'))
        storage['title'].append(entry.get('title'))
        storage['topics'].append(entry.get('topics'))
        storage['authors'].append(entry.get('authors'))
        storage['date'].append(entry.get('date'))
        storage['keywords'].append(entry.get('keywords'))
        storage['lead'].append(entry.get('lead'))
        storage['mention'].append(entry.get('mention'))
        storage['paragraphs'].append(entry.get('paragraphs'))
        storage['url'].append(entry.get('url'))
        storage['thumbnail'].append(thumbnail)
        storage['captions'].append([f.get('caption') for f in figures])
        storage['sources'].append([f.get('source')  for f in figures])

    result = {key: np.array(val, dtype=object) for key, val in storage.items()}

    result['dataframe'] = pd.DataFrame({
        '_id':       result['_id'],
        'id':        result['id'],
        'title':     result['title'],
        'topics':    result['topics'],
        'authors':   result['authors'],
        'date':      result['date'],
        'keywords':  result['keywords'],
        'lead':      result['lead'],
        'mention':   result['mention'],
        'sources':   result['sources'],
        'url':       result['url'],
        'thumbnail': result['thumbnail'],
    })

    return result


if __name__ == "__main__":
    yaml_file = "mmc-10.yaml"

    arrays = read_yaml(yaml_file)
    total  = len(arrays['id'])

    print(f"Number of entries: {total}")
    print(f"\nArray shapes:")
    for key, value in arrays.items():
        if key != 'dataframe':
            print(f"  {key}: {value.shape}")

    print(f"\nDataFrame shape: {arrays['dataframe'].shape}")
    print(f"\nDataFrame head:\n{arrays['dataframe'].head()}")

    print(f"\nFirst entry ID:        {arrays['id'][0]}")
    print(f"First entry title:     {arrays['title'][0]}")
    print(f"First entry thumbnail: {arrays['thumbnail'][0]}")
    print(f"First entry lead:      {str(arrays['lead'][0])[:100]}...")

    # ── Empty/null check per column ───────────────────────────────────────────
    print(f"\n{'─'*40}")
    print(f"Empty / null values per column (out of {total} rows):")
    print(f"{'─'*40}")
    for key, arr in arrays.items():
        if key == 'dataframe':
            continue
        empty = sum(
            1 for v in arr
            if v is None or v == [] or v == '' or (isinstance(v, float) and v != v)
        )
        pct = (empty / total) * 100
        flag = "  ⚠️" if empty > 0 else ""
        print(f"  {key:<12} {empty:>5} empty  ({pct:.1f}%){flag}")
    print(f"{'─'*40}")