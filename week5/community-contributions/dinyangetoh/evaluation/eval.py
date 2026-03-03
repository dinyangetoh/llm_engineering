import json
import math
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

from evaluation.test import TestQuestion, load_tests

load_dotenv(override=True)

_default_judge_client = None
_default_judge_model = "gpt-4.1-nano"


def set_judge_client(client: OpenAI | None, model: str | None = None):
    global _default_judge_client, _default_judge_model
    _default_judge_client = client
    if model is not None:
        _default_judge_model = model


class RetrievalEval(BaseModel):
    mrr: float = Field(description="Mean Reciprocal Rank - average across all keywords")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain (binary relevance)")
    keywords_found: int = Field(description="Number of keywords found in top-k results")
    total_keywords: int = Field(description="Total number of keywords to find")
    keyword_coverage: float = Field(description="Percentage of keywords found")


class AnswerEval(BaseModel):
    feedback: str = Field(description="Concise feedback on answer quality vs reference")
    accuracy: float = Field(description="Factual correctness 1-5")
    completeness: float = Field(description="Completeness 1-5")
    relevance: float = Field(description="Relevance to question 1-5")


def _calculate_mrr(keyword: str, retrieved_docs: list) -> float:
    keyword_lower = keyword.lower()
    for rank, doc in enumerate(retrieved_docs, start=1):
        content = getattr(doc, "page_content", doc) if isinstance(doc, object) else str(doc)
        if hasattr(content, "lower"):
            text = content.lower()
        else:
            text = str(content).lower()
        if keyword_lower in text:
            return 1.0 / rank
    return 0.0


def _calculate_dcg(relevances: list[int], k: int) -> float:
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)
    return dcg


def _calculate_ndcg(keyword: str, retrieved_docs: list, k: int = 10) -> float:
    keyword_lower = keyword.lower()
    relevances = []
    for doc in retrieved_docs[:k]:
        content = getattr(doc, "page_content", doc) if isinstance(doc, object) else str(doc)
        text = content.lower() if hasattr(content, "lower") else str(content).lower()
        relevances.append(1 if keyword_lower in text else 0)
    dcg = _calculate_dcg(relevances, k)
    idcg = _calculate_dcg(sorted(relevances, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(
    test: TestQuestion,
    fetch_context_fn,
    k: int = 10,
) -> RetrievalEval:
    retrieved_docs = fetch_context_fn(test.question, k=k)
    mrr_scores = [_calculate_mrr(kw, retrieved_docs) for kw in test.keywords]
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
    ndcg_scores = [_calculate_ndcg(kw, retrieved_docs, k) for kw in test.keywords]
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    keywords_found = sum(1 for s in mrr_scores if s > 0)
    total_keywords = len(test.keywords)
    keyword_coverage = (keywords_found / total_keywords * 100) if total_keywords else 0.0
    return RetrievalEval(
        mrr=avg_mrr,
        ndcg=avg_ndcg,
        keywords_found=keywords_found,
        total_keywords=total_keywords,
        keyword_coverage=keyword_coverage,
    )


def evaluate_answer(
    test: TestQuestion,
    get_answer_fn,
    model: str | None = None,
    client: OpenAI | None = None,
) -> tuple[AnswerEval, str]:
    generated_answer = get_answer_fn(test.question)
    judge_client = client if client is not None else _default_judge_client
    judge_model = model if model is not None else _default_judge_model
    if judge_client is None:
        judge_client = OpenAI()
    judge_messages = [
        {
            "role": "system",
            "content": "You are an expert evaluator. Compare the generated answer to the reference. Only give 5/5 for perfect answers. Output valid JSON with keys: feedback, accuracy, completeness, relevance (all numbers 1-5).",
        },
        {
            "role": "user",
            "content": f"""Question: {test.question}

Generated Answer: {generated_answer}

Reference Answer: {test.reference_answer}

Evaluate on: 1) Accuracy (factually correct vs reference) 2) Completeness (covers all aspects) 3) Relevance (directly answers question). If wrong, accuracy must be 1. Reply with JSON only: {{"feedback": "...", "accuracy": N, "completeness": N, "relevance": N}}""",
        },
    ]
    response = judge_client.chat.completions.create(
        model=judge_model,
        messages=judge_messages,
        temperature=0,
    )
    raw = response.choices[0].message.content
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"feedback": raw, "accuracy": 3.0, "completeness": 3.0, "relevance": 3.0}
    answer_eval = AnswerEval(
        feedback=data.get("feedback", ""),
        accuracy=float(data.get("accuracy", 3)),
        completeness=float(data.get("completeness", 3)),
        relevance=float(data.get("relevance", 3)),
    )
    return answer_eval, generated_answer


def run_eval_summary(
    fetch_context_fn,
    get_answer_fn,
    k_retrieval: int = 10,
    judge_client: OpenAI | None = None,
    judge_model: str | None = None,
):
    tests = load_tests()
    ret_mrrs, ret_coverages = [], []
    ans_accuracy, ans_completeness, ans_relevance = [], [], []
    per_test = []
    for test in tests:
        ret = evaluate_retrieval(test, fetch_context_fn, k=k_retrieval)
        ret_mrrs.append(ret.mrr)
        ret_coverages.append(ret.keyword_coverage)
        ans_eval, _ = evaluate_answer(
            test, get_answer_fn, model=judge_model, client=judge_client
        )
        ans_accuracy.append(ans_eval.accuracy)
        ans_completeness.append(ans_eval.completeness)
        ans_relevance.append(ans_eval.relevance)
        per_test.append(
            {
                "category": test.category,
                "mrr": ret.mrr,
                "keyword_coverage": ret.keyword_coverage,
                "accuracy": ans_eval.accuracy,
                "completeness": ans_eval.completeness,
                "relevance": ans_eval.relevance,
            }
        )
    n = len(tests)
    return {
        "retrieval": {
            "avg_mrr": sum(ret_mrrs) / n if n else 0,
            "avg_keyword_coverage_pct": sum(ret_coverages) / n if n else 0,
        },
        "answer": {
            "avg_accuracy": sum(ans_accuracy) / n if n else 0,
            "avg_completeness": sum(ans_completeness) / n if n else 0,
            "avg_relevance": sum(ans_relevance) / n if n else 0,
        },
        "n_tests": n,
        "per_test": per_test,
    }
