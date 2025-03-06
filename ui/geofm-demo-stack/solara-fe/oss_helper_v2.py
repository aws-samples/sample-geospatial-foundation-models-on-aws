import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from typing import List, Dict

class OpenSearchHelper:
    def __init__(self, config):
        """
        Initialize OpenSearch helper with configuration
        
        Args:
            config: Configuration dictionary containing vectordb settings
        """
        self.boto3_session = boto3.session.Session()
        self.region_name = config["config"]["aws_region"]
        
        # S3 clients
        self.s3 = boto3.client('s3')
        self.s3_res = boto3.resource('s3')
        
        # OpenSearch connection params
        self.host = config["config"]["vectordb_collection_host"]
        self.index_name = config["config"]["vectordb_collection_name"]
        
        # Setup AWS auth
        self.service = 'aoss'
        self.credentials = boto3.Session().get_credentials()
        self.awsauth = AWSV4SignerAuth(self.credentials, self.region_name, self.service)
        
        # Initialize OpenSearch client
        self.aoss_client = boto3.client('opensearchserverless', region_name=self.region_name)
        self.oss_client = self._initialize_opensearch()

    def _initialize_opensearch(self) -> OpenSearch:
        """Initialize and return OpenSearch client"""
        return OpenSearch(
            hosts=[{'host': self.host, 'port': 443}],
            http_auth=self.awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=300
        )

    def find_similar_items_from_query(
        self,
        query_emb: List[float],
        k: int,
        num_results: int,
        chip_ids: List[str] = None,
        date_filter: str = None
    ) -> List[Dict]:
        """Find similar items using vector search"""
        knn_query = {
            "knn": {
                "cls_emb": {
                    "vector": query_emb,
                    "k": k,
                }
            }
        }
        
        must_clauses = [knn_query]
        
        if date_filter:
            # TODO date filter is not working
            pass
            # must_clauses.append({
            #     "terms": {
            #         "date": [date_filter],
            #     }
            # })
        
        body = {
            "size": num_results,
            "_source": {
                "exclude": ["cls_emb"],
            },
            "query": {
                "bool": {
                    "must": must_clauses
                }
            }
        }
        
        if chip_ids:
            body["query"]["bool"]["must_not"] = [
                {
                    "terms": {
                        "chip_id": chip_ids
                    }
                }
            ]
        
        res = self.oss_client.search(index=self.index_name, body=body)
        
        return [{
            "score": hit["_score"],
            "chip_id": hit["_source"]["chip_id"],
            "s3_location_netcdf": hit["_source"]["s3_location_netcdf"],
            "s3_location_chip_png": hit["_source"]["s3_location_chip_png"],
            "date": hit["_source"]["date"],
            "s2_tile_id": hit["_source"]["s2_tile_id"]
        } for hit in res["hits"]["hits"]]
