import requests
import os
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
import pandas as pd 


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
    "INDEX": "test_index",
    "RETURN_SIZE": 10,
    "CACHE_DIR": ".cache/",
    "DIMENSION": 2,
}
import pytest 
@pytest.mark.order(1)
def test_index_document():
    
    def gendata(n: int = 10):
        features = np.ones((n, config['DIMENSION']), dtype=float)
        features = features * np.arange(n).reshape(-1, 1)

        # df from dict 
        # datetime must be yyyyMMdd
        # https://www.elastic.co/guide/en/elasticsearch/reference/2.0/mapping-date-format.html#built-in-date-formats
        df = pd.DataFrame.from_dict({
            'index':  [f'image{x}' for x in range(n)], \
            'field1': [f'string{x}' for x in range(n)], \
            'field2': [f'string{x+1}' for x in range(n)], \
            'field3': [x for x in range(n)], \
            'timestamp': [f'202001{x:02d}' for x in range(1,n+1)], \
            'feature': features.tolist() \
        })
        return df
    proc = ElasticProcessor(config)
    df = gendata()
    df_structure = {  
        "mappings": {
            "properties": {
                "index": {"type": "integer"},
                "field1": {"type": "text"},
                "field2": {"type": "text"},
                "field3": {"type": "integer"},
                "timestamp": {"type": "date", "format": "basic_date"},
                "feature": {"type": "dense_vector", "dims": config['DIMENSION'], "index": True, "similarity": "l2_norm"}
            }
        }
    }

    proc.index_dataframe(df, df_structure)

def test_search_text_inrange_pipeline():
    proc = ElasticProcessor(config)
    print(proc.info())
    results1 = proc.search_text_inrange_pipeline('string1', ['field1', 'field2'], timefield='timestamp', start=None, end=None, filter=None)
    results2 = proc.search_text_inrange_pipeline('string1', ['field1', 'field2'], timefield='timestamp', start=None, end=None, filter=['image0','image1','image2'])
    results3 = proc.search_text_inrange_pipeline('string1', ['field1', 'field2', 'field3'], timefield='timestamp', start=datetime(2020, 1, 1), end=datetime(2020, 1, 2), filter=['image0','image1','image2'])
    from pprint import pprint
    pprint(results1)
    print('-'*80)
    pprint(results2)
    print('-'*80)
    pprint(results3)

def test_search_text_closestday_pipeline():
    proc = ElasticProcessor(config)
    print(proc.info())
    results1 = proc.search_text_closestday_pipeline('string1', ['field1', 'field2'], timefield='timestamp', timestamp=None, filter=None)
    results2 = proc.search_text_closestday_pipeline('string1', ['field1', 'field2'], timefield='timestamp', timestamp=datetime(2020, 1, 2), filter=None)
    from pprint import pprint
    pprint(results1)
    print('-'*80)
    pprint(results2)

def test_search_timestamp_pipeline():
    proc = ElasticProcessor(config)
    print(proc.info())
    results1 = proc.search_timestamp_pipeline(timefield='timestamp', timestamp=datetime(2020, 1, 2), filter=None)
    results2 = proc.search_timestamp_pipeline(timefield='timestamp', timestamp=datetime(2020, 1, 2), filter=['image1','image2','image3'])
    from pprint import pprint
    pprint(results1)
    print('-'*80)
    pprint(results2)

def test_search_timerange_pipeline():
    proc = ElasticProcessor(config)
    print(proc.info())
    results1 = proc.search_time_range_pipeline(timefield='timestamp', start=datetime(2020, 1, 3),  end=datetime(2020, 1, 4), filter=None)
    results2 = proc.search_time_range_pipeline(timefield='timestamp', start=datetime(2020, 1, 3),  end=datetime(2020, 1, 4), filter=['image1', 'image2', 'image3'])
    from pprint import pprint
    pprint(results1)
    print('-'*80)
    pprint(results2)

