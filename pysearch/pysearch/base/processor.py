import json
from pysearch.utils.time import time_this
from typing import Dict, List, Any
from pathlib import Path 
from pysearch.utils.network import ping

class Processor:
    def __init__(self, config: Dict[str, Any]):

        assert 'HOST' in config, "HOST is not defined in config"
        assert 'PORT' in config, "PORT is not defined in config"
        assert 'INDEX' in config, "INDEX is not defined in config"
        assert 'RETURN_SIZE' in config, "RETURN_SIZE is not defined in config" 

        self.host = config['HOST']
        self.port = config['PORT']
        self.index = config['INDEX']
        self.return_size = config['RETURN_SIZE']
        self.cache_dir = Path(config.get('CACHE_DIR', 'cache'))
        
        self.generator = None
        self.analyser = None

    def ping(self):
        return self.client.ping()
    
    def info(self):
        assert self.ping(), "Cannot connect to cluster"  
    
    def _connect(self):
        raise NotImplementedError
    
    def index_document(self, document):
        assert self.ping(), "Cannot connect to cluster"
    
    def update_data_field(self, doc_id, field, value):
        assert self.ping(), "Cannot connect to cluster"

    @time_this
    def search(self, text_query, incremental_query=False, document_set=None):
        assert self.ping(), "Cannot connect to cluster"
    
    @time_this
    def get_document_by_id(self, ids):
        assert self.ping(), "Cannot connect to cluster"
