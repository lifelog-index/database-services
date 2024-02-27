import requests
import os
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
import pandas as pd 
import random
dotenv_path = Path(os.path.realpath(__file__)).parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

ELASTIC_PORT = os.environ.get("ELASTIC_PORT", None)
ELASTIC_USERNAME = os.environ.get("ELASTIC_USERNAME", None)
ELASTIC_PASSWORD = os.environ.get("ELASTIC_PASSWORD", None)

assert ELASTIC_PORT is not None, "ELASTIC_PORT is not set"
assert ELASTIC_USERNAME is not None, "ELASTIC_USERNAME is not set"
assert ELASTIC_PASSWORD is not None, "ELASTIC_PASSWORD is not set"

from pysearch.elastic import ElasticProcessor

config = {
    # Global config
    "HOST": "0.0.0.0",
    "PORT": ELASTIC_PORT,
    "USERNAME": ELASTIC_USERNAME,
    "PASSWORD": ELASTIC_PASSWORD,
    "INDEX": "test_index_lsc",
    "RETURN_SIZE": 10,
    "CACHE_DIR": ".cache/",
    "DIMENSION": 2,
}

def test_index_document():
    
    def gendata(n: int = 10):
        features = np.ones((n, config['DIMENSION']), dtype=float)
        features = features * np.arange(n).reshape(-1, 1)
        months = ['March', 'May']
        years = ['2019', '2020', '2021']
        dayOfWeek = ['Monday', 'Tuesday']
        semantic_locs = [
            'Starbucks', 'Beshoffs Of Howth', "Woodie's DIY", 'Dunnes Stores',
        ]
        # df from dict 
        # datetime must be yyyyMMdd
        # https://www.elastic.co/guide/en/elasticsearch/reference/2.0/mapping-date-format.html#built-in-date-formats
        df = pd.DataFrame.from_dict({
            'index':  [f'image{x}' for x in range(n)], \
            'field1': [f'string{x}' for x in range(n)], \
            'field2': [f'string{x+1}' for x in range(n)], \
            'field3': [x for x in range(n)], \
            'dayOfWeek': [random.choice(dayOfWeek) for x in range(n)], \
            'month': [random.choice(months) for x in range(n)], \
            'year': [random.choice(years) for x in range(n)], \
            'semantic_name': [random.choice(semantic_locs) for x in range(n)], \
            'timestamp': [f'202001{x:02d}' for x in range(1,n+1)], \
            # 'feature': features.tolist() \
        })
        return df
    proc = ElasticProcessor(config)
    df = gendata()
    print(df)
    df_structure = {  
        "mappings": {
            "properties": {
                "index": {"type": "integer"},
                "field1": {"type": "text"},
                "field2": {"type": "text"},
                "field3": {"type": "integer"},
                'dayOfWeek': {"type": "text"},
                'month': {"type": "text"},
                'year': {"type": "text"},
                'semantic_name': {"type": "text"},
                "timestamp": {"type": "date", "format": "basic_date"},
                # "feature": {"type": "dense_vector", "dims": config['DIMENSION'], "index": True, "similarity": "l2_norm"}
            }
        }
    }

    proc.index_dataframe(df, df_structure)

def test_search_text_semantic_time_pipeline():
    proc = ElasticProcessor(config)
    print(proc.info())
    # results = proc.search_by_tags_pipeline('string', ['field1', 'field2'], tags={'year': ['2019']}, filter=None)
    results = proc.search_by_tags_pipeline('string', ['field1', 'field2'], tags={'dayOfWeek': ['monday'], 'year': [2019,2020]}, filter=None)
    for result in results:
        print(result['_id'])

def test_search_text_semantic_loc_pipeline():
    proc = ElasticProcessor(config)
    print(proc.info())
    # results = proc.search_by_tags_pipeline('string', ['field1', 'field2'], tags={'year': ['2019']}, filter=None)
    results = proc.search_by_tags_pipeline('string', ['field1', 'field2'], tags={'semantic_name': ['diy']}, filter=None)
    for result in results:
        print(result['_id'])
