# Indexing and Querying with Vector Databases

This repository provides a collection of services and utilities for indexing, querying, and composing queries over vector databases such as Elasticsearch and Milvus. The tools here are designed to streamline the process of managing metadata and vector embeddings.

## Prerequisites

#### Install dependencies:

```sh
    uv sync
    source venv/bin/activate  # Activate virtual environment
```

#### Prepare a `.env` file with the following variables:

```sh
ELASTIC_PORT=9200
ELASTIC_USERNAME=your_username
ELASTIC_PASSWORD=your_password
MILVUS_PORT=19530
CLIP_PORT=20541
CLIP_EMB_DIM=768
```

#### Start databases (Elasticsearch and Milvus) on your local machine or server.

```
    docker compose up -d
```

#### Ensure everything is running correctly by running the tests:

```sh
    uv run pytest
```

## Example Notebooks

It includes ready-to-use scripts for database operations, as well as a unified interface for building and executing complex search queries. For practical examples, refer to the followingnotebooks:

-   [`examples/lsc2024/elastic_search_index_and_query.ipynb`](examples/lsc2024/elastic_search_index_and_query.ipynb):  
    Indexes metadata into Elasticsearch and demonstrates various search queries.
-   [`examples/vbs2024/milvus_index_and_query.ipynb`](examples/vbs2024/milvus_index_and_query.ipynb):  
    Indexes CLIP embeddings into Milvus and demonstrates vector search.

## Pysearch

The [`pysearch`](pysearch/pysearch) package provides a unified interface for indexing and querying both Elasticsearch and Milvus vector databases.

### Main Classes

-   [`pysearch.elastic.ElasticProcessor`](pysearch/pysearch/elastic/processor.py):  
    Handles indexing and querying metadata in Elasticsearch, including text, time, and tag-based search.

-   [`pysearch.milvus.Milvus2Processor`](pysearch/pysearch/milvus/processor.py):  
    Handles indexing and vector search in Milvus, supporting CLIP embeddings and filtering.

### Usage Example

#### Elasticsearch

```python
from pysearch.elastic import ElasticProcessor

config = {
    "HOST": "0.0.0.0",
    "PORT": 9200,
    "USERNAME": "your_username",
    "PASSWORD": "your_password",
    "INDEX": "my_index",
    "RETURN_SIZE": 10,
    "CACHE_DIR": ".cache/",
}

proc = ElasticProcessor(config)
# Index a pandas DataFrame
proc.index_dataframe(df, df_structure)
# Search by text
results = proc.search("search text")
```

#### Milvus

```python
from pysearch.milvus import Milvus2Processor

config = {
    "HOST": "0.0.0.0",
    "PORT": 19530,
    "INDEX": "my_milvus_index",
    "RETURN_SIZE": 10,
    "DIMENSION": 768,
}

proc = Milvus2Processor(config)
# Index a list of embeddings
proc.index_list_document(embeddings, ids)
# Vector search
results = proc.search(query_embedding, top_k=5)
```

### Running Tests

To run all tests:

```sh
uv run pytest
```

Test files are located in the [`tests`](tests) directory and cover both Elasticsearch and Milvus pipelines.
