import yaml
import numpy as np
import pandas as pd
from pathlib import Path

def read_yaml_exclude_urls(yaml_file):
    """
    Read YAML file and extract all data except URLs into separate arrays/dataframes.
    
    Args:
        yaml_file (str): Path to the YAML file
    
    Returns:
        dict: Dictionary containing numpy arrays and pandas DataFrames for each field
    """
    
    # Load YAML file
    with open(yaml_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    # Initialize storage containers
    storage = {}
    
    # Define fields to exclude (URLs)
    exclude_fields = {'url', 'img'}
    
    # Collect all data except URLs
    for field_name, field_values in {
        'id': [],
        'topics': [],
        'authors': [],
        'date': [],
        'keywords': [],
        'lead': [],
        'mention': [],
        'paragraphs': [],
        'captions': [],
        'sources': [],
    }.items():
        storage[f'{field_name}'] = field_values
    
    # Extract data from each entry
    for entry in data:
        storage['id'].append(entry.get('id'))
        storage['topics'].append(entry.get('topics'))
        storage['authors'].append(entry.get('authors'))
        storage['date'].append(entry.get('date'))
        storage['keywords'].append(entry.get('keywords'))
        storage['lead'].append(entry.get('lead'))
        storage['mention'].append(entry.get('mention'))
        storage['paragraphs'].append(entry.get('paragraphs'))
        
        # Extract captions and sources from figures (excluding img URLs)
        figures = entry.get('figures', [])
        captions = [fig.get('caption') for fig in figures]
        sources = [fig.get('source') for fig in figures]
        storage['captions'].append(captions)
        storage['sources'].append(sources)
    
    # Convert to numpy arrays and pandas DataFrames
    result = {}
    
    # Simple fields as numpy arrays
    result['id'] = np.array(storage['id'], dtype=object)
    result['topics'] = np.array(storage['topics'], dtype=object)
    result['date'] = np.array(storage['date'], dtype=object)
    
    # Complex fields as pandas DataFrames or numpy object arrays
    result['authors'] = np.array(storage['authors'], dtype=object)
    result['keywords'] = np.array(storage['keywords'], dtype=object)
    result['lead'] = np.array(storage['lead'], dtype=object)
    result['mention'] = np.array(storage['mention'], dtype=object)
    result['paragraphs'] = np.array(storage['paragraphs'], dtype=object)
    result['captions'] = np.array(storage['captions'], dtype=object)
    result['sources'] = np.array(storage['sources'], dtype=object)
    
    # Alternative: Create a pandas DataFrame if you prefer
    result['dataframe'] = pd.DataFrame({
        'id': result['id'],
        'topics': result['topics'],
        'authors': result['authors'],
        'date': result['date'],
        'keywords': result['keywords'],
        'lead': result['lead'],
        'mention': result['mention'],
        'sources': result['sources']
    })
    
    return result

# Example usage
if __name__ == "__main__":
    yaml_file = "mmc-10.yaml"  # Change to mmc-100.yaml or mmc.yaml as needed
    
    # Read and extract data
    arrays = read_yaml_exclude_urls(yaml_file)
    
    # Display information
    print(f"Number of entries: {len(arrays['id'])}")
    print(f"\nArray shapes:")
    for key, value in arrays.items():
        if key != 'dataframe':
            print(f"  {key}: {value.shape}")
    
    print(f"\nDataFrame shape: {arrays['dataframe'].shape}")
    print(f"\nDataFrame head:\n{arrays['dataframe'].head()}")
    
    # Access individual arrays
    print(f"\nFirst entry ID: {arrays['id'][0]}")
    print(f"First entry lead: {arrays['lead'][0][:100]}...")
    print(f"First entry keywords: {arrays['keywords'][0]}")
