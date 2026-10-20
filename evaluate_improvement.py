import argparse
import json
import statistics
import time
from pathlib import Path

from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

from evaluate_performance import (
    dense_search,
    elapsed_ms,
    latency_summary,
    ranking_metrics,
    rrf_fusion,
)


VECTOR_INDEX = "anthropic-vector-index"
BASELINE_BM25_INDEX = "anthropic-bm25-index"
THAI_BM25_INDEX = "anthropic-bm25-thai-index"


def sparse_search(client, index, query, k):
    response = client.search(
        index=index,
        body={
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content", "contextualized_content"],
                    "type": "best_fields",
                }
            },
            "size": k,
        },
    )
    return [(hit["_id"], hit["_score"]) for hit in response["hits"]["hits"]]


def analyzer_tokens(client, index, text):
    mapping = client.indices.get_mapping(index=index)[index]["mappings"]["properties"]
    analyzer = mapping["content"].get("analyzer", "standard")
    response = client.indices.analyze(
        index=index, body={"analyzer": analyzer, "text": text}
    )
    return analyzer, [item["token"] for item in response["tokens"]]


def weighted_rrf(rankings, weights, rrf_k=60):
    scores = {}
    for ranking, weight in zip(rankings, weights):
        for rank, (doc_id, _) in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def evaluate(client, model, qrels, top_k, candidates, repeats):
    methods = (
        "baseline_bm25",
        "thai_bm25",
        "vector",
        "baseline_rrf",
        "improved_rrf",
        "weighted_rrf",
    )
    metrics = {name: [] for name in methods}
    latency = {
        name: []
        for name in (
            "baseline_bm25",
            "thai_bm25",
            "baseline_embedding",
            "baseline_vector",
            "improved_embedding",
            "improved_vector",
            "baseline_rrf_total",
            "improved_rrf_total",
        )
    }
    per_query = []

    model.encode(qrels[0]["question"]).tolist()
    sparse_search(client, BASELINE_BM25_INDEX, qrels[0]["question"], candidates)
    sparse_search(client, THAI_BM25_INDEX, qrels[0]["question"], candidates)

    for item in qrels:
        final = None
        for _ in range(repeats):
            baseline_start = time.perf_counter()
            start = time.perf_counter()
            baseline_bm25 = sparse_search(
                client, BASELINE_BM25_INDEX, item["question"], candidates
            )
            latency["baseline_bm25"].append(elapsed_ms(start))

            start = time.perf_counter()
            embedding = model.encode(item["question"]).tolist()
            latency["baseline_embedding"].append(elapsed_ms(start))

            start = time.perf_counter()
            vector = dense_search(client, embedding, candidates)
            latency["baseline_vector"].append(elapsed_ms(start))
            baseline_rrf = rrf_fusion([baseline_bm25, vector])
            latency["baseline_rrf_total"].append(elapsed_ms(baseline_start))

            improved_start = time.perf_counter()
            start = time.perf_counter()
            thai_bm25 = sparse_search(
                client, THAI_BM25_INDEX, item["question"], candidates
            )
            latency["thai_bm25"].append(elapsed_ms(start))

            start = time.perf_counter()
            improved_embedding = model.encode(item["question"]).tolist()
            latency["improved_embedding"].append(elapsed_ms(start))

            start = time.perf_counter()
            improved_vector = dense_search(client, improved_embedding, candidates)
            latency["improved_vector"].append(elapsed_ms(start))

            improved_rrf = rrf_fusion([thai_bm25, improved_vector])
            improved_weighted_rrf = weighted_rrf(
                [thai_bm25, improved_vector], weights=[0.25, 1.0]
            )
            latency["improved_rrf_total"].append(elapsed_ms(improved_start))

            final = {
                "baseline_bm25": baseline_bm25,
                "thai_bm25": thai_bm25,
                "vector": vector,
                "baseline_rrf": baseline_rrf,
                "improved_rrf": improved_rrf,
                "weighted_rrf": improved_weighted_rrf,
            }

        query_row = {
            "id": item["id"],
            "question": item["question"],
            "relevant_doc_ids": item["relevant_doc_ids"],
            "methods": {},
        }
        for method, ranking in final.items():
            ids = [doc_id for doc_id, _ in ranking[:top_k]]
            row = ranking_metrics(ids, item["relevant_doc_ids"], top_k)
            metrics[method].append(row)
            query_row["methods"][method] = {"top_doc_ids": ids, "metrics": row}
        per_query.append(query_row)

    aggregate = {
        method: {
            metric: round(statistics.mean(row[metric] for row in rows), 4)
            for metric in rows[0]
        }
        for method, rows in metrics.items()
    }
    return aggregate, {name: latency_summary(rows) for name, rows in latency.items()}, per_query


def main():
    parser = argparse.ArgumentParser(description="Compare assignment baseline with Thai BM25 improvement.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=9200)
    parser.add_argument("--qrels", default="evaluation_qrels.json")
    parser.add_argument("--output", default="improvement_evaluation_report.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    client = OpenSearch(hosts=[{"host": args.host, "port": args.port}], use_ssl=False)
    for index in (VECTOR_INDEX, BASELINE_BM25_INDEX, THAI_BM25_INDEX):
        if not client.indices.exists(index=index):
            raise RuntimeError(f"Required index does not exist: {index}")

    qrels = json.loads(Path(args.qrels).read_text(encoding="utf-8"))
    model = SentenceTransformer("BAAI/bge-m3")
    aggregate, latency, per_query = evaluate(
        client, model, qrels, args.top_k, args.candidates, args.repeats
    )

    sample = "อาการของโรคหัดเยอรมัน"
    baseline_analyzer, baseline_tokens = analyzer_tokens(
        client, BASELINE_BM25_INDEX, sample
    )
    thai_analyzer, thai_tokens = analyzer_tokens(client, THAI_BM25_INDEX, sample)
    report = {
        "experiment": {
            "baseline": "Standard-analyzer BM25 + vector search + RRF",
            "improved": "Thai-analyzer BM25 + the same vector search + the same RRF",
            "weighted_improved": "Thai-analyzer BM25 (weight 0.25) + vector search (weight 1.0) + weighted RRF",
            "controlled_variables": "Corpus, chunks, embeddings, candidate pool, RRF k, qrels and final-answer LLM are unchanged.",
        },
        "analyzer_example": {
            "text": sample,
            "baseline": {"analyzer": baseline_analyzer, "tokens": baseline_tokens},
            "improved": {"analyzer": thai_analyzer, "tokens": thai_tokens},
        },
        "configuration": {
            "questions": len(qrels),
            "top_k": args.top_k,
            "candidates": args.candidates,
            "repeats": args.repeats,
            "rrf_k": 60,
        },
        "aggregate_retrieval_metrics": aggregate,
        "latency": latency,
        "per_query": per_query,
    }
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("method          Hit@5  Precision@5  Recall@5  MRR     nDCG@5")
    for method, row in aggregate.items():
        print(
            f"{method:<15} {row['hit@5']:<6.3f} {row['precision@5']:<12.3f} "
            f"{row['recall@5']:<9.3f} {row['reciprocal_rank']:<7.3f} {row['ndcg@5']:.3f}"
        )
    print(f"\nBaseline tokens ({baseline_analyzer}): {baseline_tokens}")
    print(f"Thai tokens ({thai_analyzer}): {thai_tokens}")
    print(f"Saved full report to {args.output}")


if __name__ == "__main__":
    main()
