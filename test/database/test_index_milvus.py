import requests
import os
from dotenv import load_dotenv
from pathlib import Path
import numpy as np

dotenv_path = Path(os.path.realpath(__file__)).parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

CLIP_PORT = os.environ.get("CLIP_PORT", None)
MILVUS_PORT = os.environ.get("MILVUS_PORT", None)

assert CLIP_PORT is not None, "CLIP_PORT is not set"
assert MILVUS_PORT is not None, "MILVUS_PORT is not set"

from pymilvus import CollectionSchema, FieldSchema, DataType
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from tqdm import tqdm
import numpy as np
import pytest
config = {
    # Global config
    "HOST": "localhost",
    "PORT": MILVUS_PORT,
    "INDEX": "test_index",
    "RETURN_SIZE": 10,
    "CACHE_DIR": ".cache/",
    # Milvus config
    "DIMENSION": 768,
}

@pytest.mark.order("first")
def test_index_document(index_name: str = 'test'):
    print(MILVUS_PORT)
    connections.connect(
        alias="default", 
        host=config['HOST'], 
        port=config['PORT'],
    )

    if utility.list_collections() == [index_name]:
        utility.drop_collection(index_name)

    def create_milvus_collection(collection_name, dim):
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
        
        fields = [
            FieldSchema(name='name', dtype=DataType.VARCHAR, descrition='image name', max_length=500, 
                        is_primary=True, auto_id=False),
            FieldSchema(name='id', dtype=DataType.INT64, descrition='image id'),
            FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, descrition='image embedding vectors', dim=dim)
        ]
        schema = CollectionSchema(fields=fields, description='reverse image search')
        collection = Collection(name=collection_name, schema=schema)

        index_params = {
            'metric_type':'L2',
            'index_type':"IVF_FLAT",
            'params':{"nlist":2048}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        return collection

    collection = create_milvus_collection(index_name, config['DIMENSION'])

    # n = (2508110 // 256) * 256 
    n = 256 * 4
    bs = n // 256 if n > 256 else n

    pbar=tqdm(range(0, n, bs))
    for i in pbar:

        features = np.ones((bs, 768), dtype=np.int32)
        features = features * np.arange(i,i+bs).reshape(-1, 1)
        ids = [x for x in range(i, i+bs)]
        image_names = [f"image{i}" for i in range(i, i+bs)]

        data = [
            image_names, ids, features
        ]
        info = collection.insert(data)
        pbar.set_description(str(info))
    
    print('Total number of inserted data is {}.'.format(collection.num_entities))
    assert collection.num_entities == n, "Number of inserted data is not equal to the number of data"

def test_search(index_name: str = 'test'):
    connections.connect(
        alias="default", 
        host=config['HOST'], 
        port=config['PORT'],
    )

    n = 4 * 256
    # n = (2508110 // 256) * 256

    collection = Collection(name=index_name)
    collection.load()
    assert collection.num_entities == n, "Number of inserted data is not equal to the number of data"

    query = np.ones((1, config['DIMENSION']), dtype=np.int32)
    query = query * np.arange(0,1).reshape(-1, 1)

    res = collection.search(
        data=query, 
        anns_field="embedding", 
        param={"nprobe": 10}, 
        limit=100, 
        consistency_level="Strong"
    )
    print(res)

def test_search_with_int_filter(index_name: str = 'test'):
    connections.connect(
        alias="default", 
        host=config['HOST'], 
        port=config['PORT'],
    )

    n = 4 * 256
    # n = (2508110 // 256) * 256
    query = np.ones((1, config['DIMENSION'])) * 7
    random_ids = np.random.randint(0, n, min(100000,n-1)).tolist() + [7]
    # expected_top1 = closest id to 7 in random_ids (7)
    expected_top1 = 'image7'
    collection = Collection(name=index_name)
    collection.load()

    assert collection.num_entities == n, "Number of inserted data is not equal to the number of data"
    expr = 'id in [' + ','.join([str(x) for x in random_ids]) + ']'

    search_params = {"metric_type": "L2", "params": {"nprobe": 128}}

    results = collection.search(
        data=query, 
        anns_field="embedding", 
        param=search_params, 
        limit=100, 
        expr= expr,
        consistency_level="Strong"
    )
    result = results[0].ids
    assert result[0] == expected_top1, "Expected top1 id is not equal to the result"

def test_search_with_string_filter(index_name: str = 'test'):
    connections.connect(
        alias="default", 
        host=config['HOST'], 
        port=config['PORT'],
    )
    # n = (2508110 // 256) * 256
    n = 4 * 256
    query = np.ones((1, config['DIMENSION'])) * 7
    random_ids = np.random.randint(0, n, min(100000,n-1)).tolist() + [7]
    # expected_top1 = closest id to 7 in random_ids (7)
    collection = Collection(name=index_name)
    collection.load()
    assert collection.num_entities == n, "Number of inserted data is not equal to the number of data"
    expr = 'name in [' + ','.join([f'\"image{x}\"' for x in random_ids]) + ']'
    expected_top1 = 'image7'

    search_params = {"metric_type": "L2", "params": {"nprobe": 128}}

    results = collection.search(
        data=query, 
        anns_field="embedding", 
        param=search_params, 
        limit=100, 
        expr= expr,
        consistency_level="Strong"
    )
    result = results[0].ids
    assert result[0] == expected_top1, "Expected top1 id is not equal to the result"

@pytest.mark.order("last")
def test_delete(index_name: str = 'test'):
    connections.connect(
        alias="default", 
        host=config['HOST'], 
        port=config['PORT'],
    )
    collection = Collection(name=index_name)
    collection.drop_index()