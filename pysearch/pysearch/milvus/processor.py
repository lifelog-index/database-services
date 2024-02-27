import json
from pysearch.utils.time import time_this
from typing import Dict, List, Any, Optional
from pysearch.base.processor import Processor
import logging
from pymilvus import CollectionSchema, FieldSchema, DataType
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

from tqdm import tqdm
import numpy as np 

logger = logging.getLogger(__name__)

class Milvus2Processor(Processor):
    def __init__(self, config: Dict[str, Any], autoload_collection: bool = True):
        super().__init__(config)
        assert 'DIMENSION' in config, "DIMENSION is not defined in config"
        self.dimension = config['DIMENSION']
        self.topk: int = config['RETURN_SIZE']
        self.client = self._connect()
        self.collection = self.create_milvus_collection(self.index)
        self._available_indexes = None 
        if autoload_collection:
            self.collection.load()

    def available_indexes(self) -> List[str]:
        if self._available_indexes is None:
            self._available_indexes = self.collection.query(expr="id != \"\"")
            self._available_indexes = [x['id'] for x in self._available_indexes]
        return self._available_indexes
    
    def get_document_by_id(self, ids: List[str]) -> list[list[float]]:
        expr = 'id in [' + ','.join([f'\"{id}\"' for id in ids]) + ']'
        res = self.collection.query(expr=expr, output_fields=['embedding'])
        return [x['embedding'] for x in res]
    
    def create_milvus_collection(self, collection_name: str):
        if utility.has_collection(collection_name):
            return Collection(name=collection_name)
        
        fields = [
            FieldSchema(name='id', dtype=DataType.VARCHAR, descrition='image name', max_length=500, 
                        is_primary=True, auto_id=False),
            FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, descrition='image embedding vectors', dim=self.dimension)
        ]

        schema = CollectionSchema(fields=fields, description='Pysearch collection')
        collection = Collection(name=collection_name, schema=schema)

        index_params = {
            'metric_type':'L2',
            'index_type':"IVF_FLAT",
            'params':{"nlist":2048}
        }

        collection.create_index(field_name="embedding", index_params=index_params)
        return collection


    def drop(self):
        if utility.has_collection(self.index):
            utility.drop_collection(self.index)

    def index_document(self, batch_document):
        super().index_document(batch_document)
        info = self.collection.insert(batch_document)
        return info
    
    @time_this
    def update_list_document(self, document_list, raw_ids=None):
        n = len(document_list)
        bs = n // 256 if n > 256 else n
        raw_ids = raw_ids if raw_ids is not None else [i for i in range(n)]
        pbar=tqdm(range(0, n, bs))
        for i in pbar:
            data = [
                raw_ids[i:i+bs], document_list[i:i+bs]
            ]
            info = self.index_document(data)
            pbar.set_description(str(info))
    
    def index_list_document(self, document_list, raw_ids=None):
        self.drop()
        self.collection = self.create_milvus_collection(self.index)
        self.update_list_document(document_list, raw_ids)
        self.collection.load()

    
    @time_this
    def search(self, 
               query_embedding: np.ndarray, 
               top_k: Optional[int] = None, 
               return_distance: bool = True, 
               filter: Optional[List[str]] = None) -> tuple[list[str], list[float]] | list[str]:
        
        if len(query_embedding.shape) != 2:
            raise ValueError("Invalid shape for feature vector!")
        
        expr = None
        if filter is not None:
            expr = 'id in [' + ','.join([f'\"{id}\"' for id in filter]) + ']'
        
        if top_k is None:
            top_k = self.topk
        
        assert top_k > 0, "top_k must be greater than 0"

        search_params = {"metric_type": "L2", "params": {"nprobe": min(1024, top_k)}}

        results = self.collection.search(
            data=query_embedding, 
            anns_field="embedding", 
            param=search_params, 
            limit=top_k, 
            expr= expr,
            consistency_level="Strong"
        )
        hits = results[0]
        if return_distance:
            return (hits.ids, hits.distances)
        return hits.ids
    
    def ping(self):
        return True 
    
    def _connect(self):
        connections.connect(
            alias="default", 
            host=self.host,
            port=self.port,
        )
        
    def kill(self, collection_name):
        utility.drop_collection(collection_name)

    def info(self):
        super().info()

        # collection.name                  # Return the name of the collection.
        # collection.description           # Return the description of the collection.
        # collection.num_entities          # Return the number of entities in the collection.

        return {
            "name": self.collection.name,
            "description": self.collection.description,
            "num_entities": self.collection.num_entities,
            "collections": utility.list_collections(),
            "details": [index.params for index in self.collection.indexes]
        }