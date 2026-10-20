import argparse

from opensearchpy import OpenSearch, helpers


BASELINE_INDEX = "anthropic-bm25-index"
THAI_INDEX = "anthropic-bm25-thai-index"


def create_thai_index(client, index_name):
    body = {
        "settings": {
            "analysis": {"analyzer": {"default": {"type": "thai"}}},
            "similarity": {"default": {"type": "BM25"}},
            "index.queries.cache.enabled": False,
        },
        "mappings": {
            "properties": {
                "content": {
                    "type": "text",
                    "analyzer": "thai",
                    "search_analyzer": "thai",
                },
                "contextualized_content": {
                    "type": "text",
                    "analyzer": "thai",
                    "search_analyzer": "thai",
                },
                "doc_id": {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
                "original_index": {"type": "integer", "index": False},
            }
        },
    }
    client.indices.create(index=index_name, body=body)


def copy_documents(client, source_index, target_index):
    documents = helpers.scan(
        client,
        index=source_index,
        query={"query": {"match_all": {}}},
        preserve_order=False,
    )
    actions = (
        {
            "_op_type": "index",
            "_index": target_index,
            "_id": hit["_id"],
            "_source": hit["_source"],
        }
        for hit in documents
    )
    success, errors = helpers.bulk(client, actions, raise_on_error=False)
    client.indices.refresh(index=target_index)
    return success, errors


def main():
    parser = argparse.ArgumentParser(
        description="Create a Thai-analyzer BM25 index from the assignment baseline index."
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=9200)
    parser.add_argument("--source", default=BASELINE_INDEX)
    parser.add_argument("--target", default=THAI_INDEX)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and rebuild the target index if it already exists.",
    )
    args = parser.parse_args()

    client = OpenSearch(
        hosts=[{"host": args.host, "port": args.port}], use_ssl=False
    )
    if not client.ping():
        raise RuntimeError(f"OpenSearch is unavailable at http://{args.host}:{args.port}")
    if not client.indices.exists(index=args.source):
        raise RuntimeError(f"Baseline index does not exist: {args.source}")

    if client.indices.exists(index=args.target):
        if not args.recreate:
            print(f"Target index already exists: {args.target}")
            print("Use --recreate to rebuild it.")
            return
        client.indices.delete(index=args.target)

    create_thai_index(client, args.target)
    success, errors = copy_documents(client, args.source, args.target)
    count = client.count(index=args.target)["count"]
    tokens = client.indices.analyze(
        index=args.target, body={"analyzer": "thai", "text": "อาการของโรคหัดเยอรมัน"}
    )["tokens"]

    print(f"Created {args.target}")
    print(f"Indexed documents: {success}; document count: {count}; errors: {len(errors)}")
    print("Thai tokens:", [item["token"] for item in tokens])


if __name__ == "__main__":
    main()
