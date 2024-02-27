# write a code to index vector data into elasticsearch
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, RequestError, RequestsHttpConnection
from elasticsearch.helpers import bulk
from tqdm import tqdm

dotenv_path = Path(os.path.realpath(__file__)).parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

ELASTIC_PORT = os.environ.get("ELASTIC_PORT", None)
ELASTIC_USERNAME = os.environ.get("ELASTIC_USERNAME", None)
ELASTIC_PASSWORD = os.environ.get("ELASTIC_PASSWORD", None)

assert ELASTIC_PORT is not None, "ELASTIC_PORT is not set"
assert ELASTIC_USERNAME is not None, "ELASTIC_USERNAME is not set"
assert ELASTIC_PASSWORD is not None, "ELASTIC_PASSWORD is not set"
import pytest 

document_structure = {  
    "mappings": {
        "properties": {
            "index": {"type": "integer"},
            "field1": {"type": "text"},
            "field2": {"type": "integer"},
            "field3": {"type": "integer"},
            "feature": {"type": "dense_vector", "dims": 768, "index": True, "similarity": "l2_norm"}
        }
    }
}
@pytest.mark.order("first")
def test_es_connection():
    es = Elasticsearch([f'http://0.0.0.0:{ELASTIC_PORT}'],
                        timeout=100, \
                        connection_class=RequestsHttpConnection, 
                        http_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD), 
                        use_ssl=False, 
                        verify_certs=False)
    
    assert es.ping() == True, "Elasticsearch is not running"
    es.close()

def test_index_created(index_name: str = 'test'):

    es = Elasticsearch([f'http://0.0.0.0:{ELASTIC_PORT}'],
                        timeout=100, \
                        connection_class=RequestsHttpConnection, 
                        http_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD), 
                        use_ssl=False, 
                        verify_certs=False)
    body = {
        "query": {"match_all": {}},
    }
    responses = es.search(index=index_name, body=body)['hits']['hits']
    assert len(responses) > 0, "index is not created"

@pytest.mark.order("first")
def test_index_document(index_name: str = 'test'):
    def gendata(n: int = 10):
        features = np.ones((n, 768), dtype=float)
        features = features * np.arange(n).reshape(-1, 1)

        # df from dict 
        df = pd.DataFrame.from_dict({
            'index':  [x for x in range(n)], \
            'field1': [f'string{x}' for x in range(n)], \
            'field2': [x for x in range(n)], \
            'field3': [x for x in range(n)], \
            'feature': features.tolist() \
        })
        return df 
    
    def index_iterator(df):
        for i, row in tqdm(df.iterrows(), total=len(df)):
            data = row.to_dict()
            index = data.pop('index', None)
            yield {
                "_index": index_name,
                "_id": index,
                "_source": data
            }
    # if exist index test then es.indices.delete(index='test')
    

    es = Elasticsearch([f'http://0.0.0.0:{ELASTIC_PORT}'],
                        timeout=100, \
                        connection_class=RequestsHttpConnection, 
                        http_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD), 
                        use_ssl=False, 
                        verify_certs=False)
    
    es.indices.delete(index=index_name, ignore=[400, 404])
    es.indices.create(index=index_name, body=document_structure, ignore=400)
    es.indices.get_alias("*")

    df = gendata(10)
    print(df.head())

    bulk(es, index_iterator(df))
    # refresh index
    es.indices.refresh(index=index_name)
    test_index_created(index_name)
    es.close()

    
def test_dsl_search(index_name: str = 'test'):
    from elasticsearch_dsl import Search
    es = Elasticsearch([f'http://0.0.0.0:{ELASTIC_PORT}'],
                        timeout=100, \
                        connection_class=RequestsHttpConnection, 
                        http_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD), 
                        use_ssl=False, 
                        verify_certs=False)
    def search(query):
        s = Search(using=es, index=index_name) \
            .query("multi_match", query=query, fields=['field1']) \
            .extra(size=100, explain=True)
        response = s.execute()
        return response
        
    responses = search("string3")
    assert responses[0].meta.id == '3', "dsl search is not working, expected id 3 but got {}".format(responses[0].meta.id)

def test_vector_search(index_name: str = 'test'):
    query = np.ones((768), dtype=float) * 7
    # expected result id = 7
    es = Elasticsearch([f'http://0.0.0.0:{ELASTIC_PORT}'],
                        timeout=100, \
                        connection_class=RequestsHttpConnection, 
                        http_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD), 
                        use_ssl=False, 
                        verify_certs=False)
    def search(query_vector, index, top_k=10):
        script_query = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "1 / (1 + l2norm(params.query_vector, 'feature'))",
                    "params": {"query_vector": query_vector.tolist()}
                }
            }
        }
        body = {
            "size": top_k,
            "query": script_query
        }
        return es.search(index=index, body=body)['hits']['hits']

    print(query[:10])

    try:
        responses = search(query, index=index_name, top_k=10)
        for hit in responses[:10]:
            print(hit['_id'],hit['_score'])
            print(hit['_source']['feature'][:10])
        assert responses[0]['_id'] == '7', "vector search is not working, expected id 7 but got {}".format(responses[0]['_id'])
    except RequestError as e:
        print(e.info['error'])
@pytest.mark.order("last")
def test_delete_index(index_name: str = 'test'):
    es = Elasticsearch([f'http://0.0.0.0:{ELASTIC_PORT}'],
                        timeout=100, \
                        connection_class=RequestsHttpConnection, 
                        http_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD), 
                        use_ssl=False, 
                        verify_certs=False)
    es.indices.delete(index=index_name, ignore=[400, 404])