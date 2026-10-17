import argparse
import json
import math
import statistics
import time
from pathlib import Path

from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer


VECTOR_INDEX = "anthropic-vector-index"
BM25_INDEX = "anthropic-bm25-index"


def elapsed_ms(start):
    return (time.perf_counter() - start) * 1000


def percentile_95(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def latency_summary(values):
    return {
        "mean_ms": round(statistics.mean(values), 3),
        "p95_ms": round(percentile_95(values), 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
    }


def ranking_metrics(ranked_ids, relevant_ids, k):
    ranked = ranked_ids[:k]
    relevant = set(relevant_ids)
    hits = [1 if doc_id in relevant else 0 for doc_id in ranked]
    first_relevant_rank = next((i + 1 for i, hit in enumerate(hits) if hit), None)
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1 / math.log2(rank + 2) for rank in range(ideal_hits))
    return {
        f"hit@{k}": int(any(hits)),
        f"precision@{k}": round(sum(hits) / k, 4),
        f"recall@{k}": round(sum(hits) / len(relevant), 4) if relevant else 0.0,
        "reciprocal_rank": round(1 / first_relevant_rank, 4) if first_relevant_rank else 0.0,
        f"ndcg@{k}": round(dcg / idcg, 4) if idcg else 0.0,
    }


def sparse_search(client, query, k):
    response = client.search(
        index=BM25_INDEX,
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


def dense_search(client, embedding, k):
    response = client.search(
        index=VECTOR_INDEX,
        body={
            "query": {"knn": {"embedding": {"vector": embedding, "k": k}}},
            "size": k,
        },
    )
    return [(hit["_id"], hit["_score"]) for hit in response["hits"]["hits"]]


def rrf_fusion(rankings, rrf_k=60):
    scores = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def index_documents(client, index):
    response = client.search(
        index=index,
        body={
            "size": 10000,
            "_source": ["doc_id", "chunk_id", "content", "contextualized_content"],
            "query": {"match_all": {}},
        },
    )
    return response["hits"]["hits"]


def evaluate_lab1(client):
    vector_docs = index_documents(client, VECTOR_INDEX)
    bm25_docs = index_documents(client, BM25_INDEX)
    vector_ids = {hit["_id"] for hit in vector_docs}
    bm25_ids = {hit["_id"] for hit in bm25_docs}
    stats = client.indices.stats(index=f"{VECTOR_INDEX},{BM25_INDEX}")
    mapping = client.indices.get_mapping(index=VECTOR_INDEX)
    dimension = mapping[VECTOR_INDEX]["mappings"]["properties"]["embedding"]["dimension"]

    def coverage(docs, field):
        present = sum(bool(hit["_source"].get(field, "").strip()) for hit in docs)
        return round(present / len(docs), 4) if docs else 0.0

    def average_chars(docs, field):
        values = [len(hit["_source"].get(field, "")) for hit in docs]
        return round(statistics.mean(values), 2) if values else 0.0

    indices = {}
    for index_name, docs in ((VECTOR_INDEX, vector_docs), (BM25_INDEX, bm25_docs)):
        indices[index_name] = {
            "document_count": len(docs),
            "store_size_bytes": stats["indices"][index_name]["total"]["store"]["size_in_bytes"],
            "content_coverage": coverage(docs, "content"),
            "context_coverage": coverage(docs, "contextualized_content"),
            "average_content_chars": average_chars(docs, "content"),
            "average_context_chars": average_chars(docs, "contextualized_content"),
        }

    checks = {
        "both_indices_non_empty": bool(vector_docs and bm25_docs),
        "document_counts_match": len(vector_docs) == len(bm25_docs),
        "document_ids_match": vector_ids == bm25_ids,
        "all_documents_have_content": all(
            item["content_coverage"] == 1.0 for item in indices.values()
        ),
        "all_documents_have_context": all(
            item["context_coverage"] == 1.0 for item in indices.values()
        ),
        "embedding_dimension_is_1024": dimension == 1024,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "embedding_dimension": dimension,
        "indices": indices,
    }


def evaluate_lab2(client, model, qrels, k, candidates, repeats):
    method_metrics = {name: [] for name in ("bm25", "vector", "rrf")}
    latency = {
        name: []
        for name in ("bm25_search", "embedding", "vector_search", "rrf_fusion", "hybrid_total")
    }
    query_results = []

    model.encode(qrels[0]["question"]).tolist()
    sparse_search(client, qrels[0]["question"], candidates)

    for item in qrels:
        final_rankings = None
        for _ in range(repeats):
            total_start = time.perf_counter()

            start = time.perf_counter()
            sparse = sparse_search(client, item["question"], candidates)
            latency["bm25_search"].append(elapsed_ms(start))

            start = time.perf_counter()
            embedding = model.encode(item["question"]).tolist()
            latency["embedding"].append(elapsed_ms(start))

            start = time.perf_counter()
            dense = dense_search(client, embedding, candidates)
            latency["vector_search"].append(elapsed_ms(start))

            start = time.perf_counter()
            fused = rrf_fusion([sparse, dense])
            latency["rrf_fusion"].append(elapsed_ms(start))
            latency["hybrid_total"].append(elapsed_ms(total_start))
            final_rankings = {"bm25": sparse, "vector": dense, "rrf": fused}

        per_query = {
            "id": item["id"],
            "question": item["question"],
            "relevant_doc_ids": item["relevant_doc_ids"],
            "methods": {},
        }
        for method, ranking in final_rankings.items():
            ids = [doc_id for doc_id, _ in ranking[:k]]
            metrics = ranking_metrics(ids, item["relevant_doc_ids"], k)
            method_metrics[method].append(metrics)
            per_query["methods"][method] = {"top_doc_ids": ids, "metrics": metrics}
        query_results.append(per_query)

    aggregate = {}
    for method, rows in method_metrics.items():
        aggregate[method] = {
            metric: round(statistics.mean(row[metric] for row in rows), 4)
            for metric in rows[0]
        }

    existing_result = Path("authentic_rag_search_results.json")
    answer_checks = {"file_found": existing_result.exists()}
    if existing_result.exists():
        answers = json.loads(existing_result.read_text(encoding="utf-8"))
        answer_checks.update(
            {
                "question_count": len(answers),
                "all_answers_non_empty": all(bool(row.get("answer", "").strip()) for row in answers),
                "all_questions_have_top_k_results": all(len(row.get("results", [])) == k for row in answers),
                "contains_api_error": any(
                    "error" in row.get("answer", "").lower()
                    or "ข้อผิดพลาด" in row.get("answer", "")
                    for row in answers
                ),
            }
        )

    return {
        "configuration": {
            "questions": len(qrels),
            "top_k": k,
            "candidate_pool_per_retriever": candidates,
            "latency_repeats_per_question": repeats,
            "rrf_k": 60,
            "qrels_note": "Human-defined relevant chunk IDs; review when corpus or chunking changes.",
        },
        "aggregate_retrieval_metrics": aggregate,
        "latency": {name: latency_summary(values) for name, values in latency.items()},
        "answer_output_checks": answer_checks,
        "per_query": query_results,
    }


def print_summary(report):
    print("\nLAB 1 — INDEX QUALITY")
    print(f"Status: {report['lab1']['status'].upper()}")
    for name, value in report["lab1"]["checks"].items():
        print(f"  {'PASS' if value else 'FAIL'}  {name}")

    print("\nLAB 2 — RETRIEVAL QUALITY (average across questions)")
    print("method   Hit@5  Precision@5  Recall@5  MRR     nDCG@5")
    for method, metrics in report["lab2"]["aggregate_retrieval_metrics"].items():
        print(
            f"{method:<8} {metrics['hit@5']:<6.3f} {metrics['precision@5']:<12.3f} "
            f"{metrics['recall@5']:<9.3f} {metrics['reciprocal_rank']:<7.3f} {metrics['ndcg@5']:.3f}"
        )

    print("\nLATENCY")
    for name, values in report["lab2"]["latency"].items():
        print(f"  {name:<14} mean={values['mean_ms']:.3f} ms, p95={values['p95_ms']:.3f} ms")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Authentic RAG Lab 1 and Lab 2")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=9200)
    parser.add_argument("--qrels", default="evaluation_qrels.json")
    parser.add_argument("--output", default="evaluation_report.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    client = OpenSearch(hosts=[{"host": args.host, "port": args.port}], use_ssl=False)
    if not client.ping():
        raise RuntimeError(f"OpenSearch is unavailable at http://{args.host}:{args.port}")

    qrels = json.loads(Path(args.qrels).read_text(encoding="utf-8"))
    model_start = time.perf_counter()
    model = SentenceTransformer("BAAI/bge-m3")
    model_load_ms = elapsed_ms(model_start)

    report = {
        "methodology": {
            "lab1": "Index completeness, field coverage, ID consistency, dimension, and storage.",
            "lab2": "Human-labelled qrels with Hit@5, Precision@5, Recall@5, MRR, nDCG@5, and latency.",
        },
        "model_load_ms": round(model_load_ms, 3),
        "lab1": evaluate_lab1(client),
        "lab2": evaluate_lab2(
            client,
            model,
            qrels,
            args.top_k,
            args.candidates,
            args.repeats,
        ),
    }
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print_summary(report)
    print(f"\nSaved full report to {args.output}")


if __name__ == "__main__":
    main()
