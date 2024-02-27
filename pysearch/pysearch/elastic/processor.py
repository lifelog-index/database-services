import json
from pysearch.utils import time_this
from elasticsearch import Elasticsearch, RequestsHttpConnection
from .query_generator import QueryGenerator
from elasticsearch.helpers import bulk
import argparse
from pysearch.base.processor import Processor
from tqdm import tqdm 
from typing import List, Dict, Any, Union, Optional
from datetime import datetime

class ElasticProcessor(Processor):
    def __init__(self, config, max_result_window=1000000):
        # convert dict to namespace
        config = argparse.Namespace(**config)

        self.host = config.HOST
        self.port = config.PORT
        self.index = config.INDEX
        self.username = config.USERNAME
        self.password = config.PASSWORD
        self.return_size = config.RETURN_SIZE
        self.client = self._connect()
        self.max_result_window = max_result_window
        # change elastic _settings, index.max_result_window to 1000000
        # self.analyser = QueryAnalyser(self.generator)
    
    def index_dataframe(self, df, df_structure):

        self.client.indices.delete(index=self.index, ignore=[400, 404])
        self.client.indices.create(index=self.index, body=df_structure, ignore=400)
        self.client.indices.put_settings(index=self.index, body={"index": {"max_result_window": self.max_result_window}})

        def index_iterator(df):
            for i, row in tqdm(df.iterrows(), total=len(df)):
                data = row.to_dict()
                index = data.pop('index', None)
                yield {
                    "_index": self.index,
                    "_id": index,
                    "_source": data
                }
        bulk(self.client, index_iterator(df))
        self.client.indices.refresh(index=self.index)
        
    def _connect(self):
        es = Elasticsearch([f'http://0.0.0.0:{self.port}'],
                        timeout=100, \
                        connection_class=RequestsHttpConnection, 
                        http_auth=(self.username, self.password), 
                        use_ssl=False, 
                        verify_certs=False)
        if es.ping():
            print("Connected to Elasticsearch node")
        else:
            print("Error: Cannot connect to Elasticsearch cluster")
        return es
    
    """
    Main function for searching in elasticsearch
    """
    # @time_this
    def search(self, text_query, topk=None):
        """
        Default search function is to search by text all 
        """
        default_params = {   
            'filter': None, 'time': None, \
            'text': {'fields': None, 'should': text_query, 'must': None}
        }
        return self.compose_pipeline(query=default_params, topk=topk)
    
    def available_fields(self):
        return list(self.client.indices.get_mapping(index=self.index)[self.index]['mappings']['properties'].keys())
    
    def _field_properties(self, field):
        return self.client.indices.get_mapping(index=self.index)[self.index]['mappings']['properties'][field]
    
    def _field_type(self, field):
        return self._field_properties(field)['type']
    
    def search_filter_only(self, filter: List[str]):
        """
            Example: search_filter_only(['image1', 'image2'])
        """
        return self.compose_pipeline({ 'filter': filter})

    def search_text_closestday_pipeline(self, text_query: str, fields: List[str], filter: List[str], timefield: str, timestamp: datetime):
        """
            Example: search_text_inrange_pipeline('text', ['field1', 'field2'], \
                ['image1', 'image2'], 'timestamp', '20200101', '20200110')
            The steps are:
            1. Filter by document set
            2. Search by scoring (text in fields and time)
            3. Return the result
        """
        return  self.compose_pipeline({'filter': filter, 'text': {'fields': fields, 'should': text_query, 'must': None}, \
                                       'time': {'field': timefield, 'timestamp': timestamp}})
    
    def search_text_inrange_pipeline(self, text_query: str, fields: List[str], filter: List[str], \
                                     timefield: str, start: datetime, end: datetime):

        """
            Example: search_text_inrange_pipeline('text', ['field1', 'field2'], \
                ['image1', 'image2'], 'timestamp', '20200101', '20200110')
            The steps are:
            1. Filter by document set
            2. Filter by time range
            3. Search text in fields
            4. Return the result
        """
        return self.compose_pipeline({  'filter': filter, 'time': {'field': timefield, 'start': start, 'end': end}, \
                                        'text': {'fields': fields, 'should': text_query, 'must': None}})


    def search_by_tags_pipeline(self, text_query: str, fields: Optional[List[Any]], tags: Dict[str, List[str]], filter: List[str]):
        return self.compose_pipeline({'filter': filter, 'tags': tags, 
                                      'text': {'fields': fields, 'should': text_query, 'must': None}})
    
    def search_timestamp_pipeline(self, timefield: str, timestamp: datetime, filter: List[str]):
        """
            Example: search_time_range_pipeline('timestamp', '20200101', '20200110', ['image1', 'image2'])
            The steps are:
            1. Filter by document set
            2. Find top nearest timestamp
            3. Return the result
        """
        return self.compose_pipeline({'filter': filter, 'time': {'field': timefield, 'timestamp': timestamp}})
    
    def search_time_range_pipeline(self, timefield: str, start: datetime, end: datetime, filter: List[str]):
        """
            Example: search_time_range_pipeline('timestamp', '20200101', '20200110', ['image1', 'image2'])
            The steps are:
            1. Filter by document set
            2. Filter by time range
            3. Return the result
        """
        return self.compose_pipeline({'filter': filter, 'time': {'field': timefield, 'start': start, 'end': end}})

    def compose_pipeline(self, query: dict, topk: Optional[int] = None):
        """
            Example: compose_pipeline({'filter': ['image1', 'image2'], \
                'time': {'field': 'timestamp', 'start': '20200101', 'end': '20200110'}, \
                'text': {'fields': ['field1', 'field2'], 'must': 'text1', 'should': 'text2'}})
            The steps are:
            1. Filter by document set
            2. Filter by time range
            3. Search text in fields
            4. Return the result
        """
        topk = self.return_size if topk is None else topk
        self.generator = QueryGenerator(self.client, self.index)
        assert len(query) > 0, "Query cannot be empty"
        if query.get('filter') is not None:
            self._filter(query['filter'])
        if query.get('tags') is not None:
            self._filterByTags(query['tags'])
        if query.get('time') is not None:
            if 'timestamp' in query['time'].keys():
                self._search_time_fields(query['time']['field'], query['time']['timestamp'])
            else:
                self._search_time_fields(query['time']['field'], (query['time']['start'], query['time']['end']))
    
        if query.get('text') is not None:
            if query['text']['fields'] is None or len(query['text']['fields']) == 0:
                query['text']['fields'] = self.available_fields()
            self._search_normal_fields(query['text']['fields'], query['text']['must'], query['text']['should'])
        query = self.generator.run(profiler=False)
        # from pprint import pprint
        # pprint(query)
        result = self.client.search(index=self.index, body=json.dumps(query), size=topk)
        return result['hits']['hits']

    def _filterByTags(self, filterDict):
        """
        For each tag, generate a query that must match the tag
        Expectation:
        tags = ['tag1', 'tag2', 'tag3']
        filter_dict = {
            'tag1': [value1, value2, value3],
            'tag2': [value1, value2, value3],
        }
        Example:
        tags = ['Month', 'Year']
        filter_dict = {
            'Month': ['January', 'February', 'March'],
            'Year': ['2018', '2019', '2020'],
        }
        """
        tags = filterDict.keys()
        for tag in tags:
            values = filterDict[tag]
            index_values = []
            for value in values:
                try:
                    if not(isinstance(value, str)):
                        value = str(value)
                except Exception as e:
                    raise Exception("Value of tag {} must be a string".format(tag))
                index_values.append(value.lower())
            self.generator.gen_multi_term_query(tag, index_values)

    def _filter(self, filter: Optional[List[str]] = None):
        if filter is not None and len(filter) > 0 :
            self.generator.add_document_set(filter)

    def _search_normal_fields(self, fields, must_part, should_part):
        # https://stackoverflow.com/questions/28768277/elasticsearch-difference-between-must-and-should-bool-query
        # only support string field
        
        _fields = fields.copy()
        fields = []
        for i, field in enumerate(_fields):
            if self._field_type(field) == 'text':
                fields.append(field)

        assert isinstance(fields, list), "fields must be a list"

        # todo: check if fields are valid
        if must_part is not None:
            assert isinstance(must_part, str), "must_part must be a string, only one string is allowed"
            self.generator.gen_query_string_query(fields, must_part, False)
        if should_part is not None:
            assert isinstance(should_part, str), "should_part must be a string, only one string is allowed"
            self.generator.gen_query_string_query(fields, should_part, True)
    
    def _search_time_fields(self, timefield, timestamp):
        # only support date field, basic_date format
        if self._field_type(timefield) != 'date':
            raise Exception("timefield must be a date field")

        def _search_closest_time(timefield, timestamp):
            assert isinstance(timestamp, datetime), "timestamp must be a datetime object"
            timestamp = timestamp.strftime("%Y%m%d")
            self.generator.gen_closest_time_query(timefield, timestamp)

        def _search_time_range(timefield, timestamp):
            from_, to_ = timestamp
            from_ = from_.strftime("%Y%m%d")
            to_ = to_.strftime("%Y%m%d")
            # https://www.elastic.co/guide/en/elasticsearch/reference/2.0/mapping-date-format.html#built-in-date-formats
            self.generator.gen_range_query(timefield, from_, to_, 'basic_date')
        
        if timestamp is None or (isinstance(timestamp, tuple) and (timestamp[0] is None or timestamp[1] is None)):
            # do nothing
            return
        
        if isinstance(timestamp, datetime):    
            _search_closest_time(timefield, timestamp)
        elif isinstance(timestamp, tuple) and list(map(type, timestamp)) == [datetime, datetime]:
            _search_time_range(timefield, timestamp)
        else:
            raise ValueError("timestamp must be a datetime object or a tuple of datetime objects")
    
    def kill(self, index_name: str):
        self.client.indices.delete(index=index_name, ignore=[400, 404])
    
    def available_indices(self):        
        return self.client.indices.get_alias("*").keys()
    
    def info(self):
        super().info()
        return self.client.indices.get_mapping(index=self.index)[self.index]['mappings']

    @time_this
    def get_document_by_id(self, ids):
        index = self.index
        query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "ids": {
                                "values": ids
                            }
                        },
                        {
                            "match": {
                                "_index": index
                            }
                        }

                    ]
                }
            }
        }


        result = self.client.search(query, size=self.return_size)
        return result['hits']['hits']

