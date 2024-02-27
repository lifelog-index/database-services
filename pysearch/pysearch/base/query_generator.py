from pysearch.utils.time import time_this

class QueryGenerator:
    def __init__(self, config): 
        self.config = config
        self.reset_query()
    
    def reset_query(self):
        raise NotImplementedError
    
    @time_this
    def run(self):
        raise NotImplementedError