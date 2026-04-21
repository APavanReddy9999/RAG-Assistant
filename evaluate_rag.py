import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import math
import textwrap
from dotenv import load_dotenv
from datasets import Dataset

from ragas.metrics import (
    Faithfulness,
    ContextPrecision,
    ContextRecall,
)

# ── WHY ResponseRelevancy instead of AnswerRelevancy? ─────────────
# AnswerRelevancy uses n=3 (batch sampling) internally — Groq rejects
# n > 1 with BadRequestError. There is no way to override this
# because Ragas constructs the n=3 call itself, ignoring your ChatGroq(n=1).
#
# ResponseRelevancy is Ragas v1.0's replacement metric. It generates
# ONE reverse-question per call in a sequential loop instead of n=3
# in one shot — fully compatible with Groq's n=1 limitation.
# The score it produces is equivalent: cosine similarity between
# generated reverse-questions and the original question.
from ragas.metrics import ResponseRelevancy

from ragas import evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from rag_pipeline import load_vector_store, create_retriever, build_rag_chain

load_dotenv()


# ══════════════════════════════════════════════════════════════════
# STEP 1: TEST DATASET
# ══════════════════════════════════════════════════════════════════

TEST_QUESTIONS = [
    {
        "question": "What are the common causes of fever?",
        "ground_truth": (
            "Fever is commonly caused by viral infections, "
            "bacterial infections, or inflammatory conditions."
        ),
    },
    {
        "question": "What is the recommended dosage of paracetamol for adults?",
        "ground_truth": (
            "The typical dosage is 500 mg to 650 mg every 4 to 6 hours."
        ),
    },
    {
        "question": "How can hypertension be managed?",
        "ground_truth": (
            "Hypertension can be managed by reducing salt intake, "
            "exercising regularly, maintaining a healthy weight, "
            "and taking medications if needed."
        ),
    },
]


# ══════════════════════════════════════════════════════════════════
# STEP 2: RUN THE RAG PIPELINE
# ══════════════════════════════════════════════════════════════════

def run_pipeline_and_collect(test_questions: list) -> dict:
    print("\n📦 Loading vector store and building RAG chain...")
    vector_store = load_vector_store()
    retriever    = create_retriever(vector_store)
    chain        = build_rag_chain(retriever)

    questions, answers, contexts_list, ground_truths = [], [], [], []

    print(f"\n🔄 Running {len(test_questions)} questions through the pipeline...\n")

    for i, item in enumerate(test_questions, 1):
        question     = item["question"]
        ground_truth = item["ground_truth"]

        print(f"  [{i}/{len(test_questions)}] Q: {question}")
        answer   = chain.invoke(question)
        docs     = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]

        print(f"       Expected : {ground_truth[:70]}")
        print(f"       Actual   : {answer[:70]}")
        print(f"       Chunks   : {len(contexts)} retrieved\n")

        questions.append(question)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(ground_truth)

    return {
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts_list,
        "ground_truth": ground_truths,
    }


# ══════════════════════════════════════════════════════════════════
# STEP 3: CONFIGURE RAGAS JUDGE
# ══════════════════════════════════════════════════════════════════

def setup_ragas_judge():
    """
    Single judge LLM for all metrics — no n=1 workaround needed
    because ResponseRelevancy doesn't use batch sampling.
    """
    judge_llm = LangchainLLMWrapper(
        ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )
    )

    # Embeddings used by ResponseRelevancy to compute cosine similarity
    # between the generated reverse-question and the original question
    judge_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    )

    return judge_llm, judge_embeddings


# ══════════════════════════════════════════════════════════════════
# STEP 4: RUN RAGAS EVALUATION
# ══════════════════════════════════════════════════════════════════

def run_ragas_evaluation(data: dict, judge_llm, judge_embeddings):
    """
    METRIC SUMMARY:
      Faithfulness        → Is every claim in the answer backed by retrieved context?
                            Score = supported claims / total claims
                            Judges: Generator (LLM hallucination check)

      ResponseRelevancy   → Does the answer actually address the question asked?
                            Generates 1 reverse-question from the answer, measures
                            cosine similarity with original question.
                            Judges: Generator (on-topic check)

      ContextPrecision    → Are the most relevant chunks ranked at the top?
                            Score = relevant chunks in top-k / total retrieved
                            Judges: Retriever (ranking quality)

      ContextRecall       → Did the retriever fetch all information needed to answer?
                            Compares retrieved chunks against ground_truth.
                            Judges: Retriever (coverage check)
    """
    metrics = [
        Faithfulness(llm=judge_llm),
        ResponseRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
    ]

    dataset    = Dataset.from_dict(data)
    run_config = RunConfig(timeout=120, max_retries=3, max_workers=1)

    print("\n🧪 Running Ragas evaluation (sequential mode, ~4-5 min)...\n")
    result = evaluate(dataset=dataset, metrics=metrics, run_config=run_config)
    return result


# ══════════════════════════════════════════════════════════════════
# STEP 5: DISPLAY — flat table
# Question | Expected | Actual | Faith | RR | CP | CR
# ══════════════════════════════════════════════════════════════════

def safe_score(value) -> float:
    """Normalise any Ragas output (float/list/None/NaN) → plain float."""
    if isinstance(value, list):
        valid = [v for v in value
                 if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return sum(valid) / len(valid) if valid else float("nan")
    if value is None:
        return float("nan")
    try:
        f = float(value)
        return f if not math.isnan(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def fmt(val: float) -> str:
    return f"{val:.3f}" if not math.isnan(val) else " N/A "


def grade(val: float) -> str:
    if math.isnan(val): return " N/A "
    if val >= 0.8:      return "  ✅ "
    if val >= 0.5:      return "  ⚠️  "
    return                     "  ❌ "


def get_answer_column(df):
    """
    Ragas v1.0 renamed 'answer' → 'response' in the output dataframe.
    Check both so the script works across versions.
    """
    for col in ["response", "answer"]:
        if col in df.columns:
            return col
    raise KeyError(f"No answer column found. Available: {list(df.columns)}")


def get_metric_column(df, candidates: list) -> str:
    """Find which column name Ragas used for a given metric."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def display_results(result, test_questions: list):
    df         = result.to_pandas()
    answer_col = get_answer_column(df)

    # ResponseRelevancy may appear as 'response_relevancy' or 'answer_relevancy'
    rr_col = get_metric_column(df, ["response_relevancy", "answer_relevancy"])

    # ── Column widths ─────────────────────────────────────────────
    QW = 20   # Question
    EW = 24   # Expected
    AW = 24   # Actual
    MW =  7   # each metric column

    # Map display header → actual dataframe column name
    METRICS = [
        ("Faith", "faithfulness"),
        ("RR",    rr_col),           # ResponseRelevancy
        ("CP",    "context_precision"),
        ("CR",    "context_recall"),
    ]
    METRIC_HEADERS = [h for h, _ in METRICS]
    METRIC_COLS    = [c for _, c in METRICS]

    # ── Border builders ───────────────────────────────────────────
    def hline(left, mid, right, fill="─"):
        cols  = [fill*(QW+2), fill*(EW+2), fill*(AW+2)]
        cols += [fill*(MW+2) for _ in METRIC_HEADERS]
        return left + mid.join(cols) + right

    TOP = hline("┌", "┬", "┐")
    DIV = hline("├", "┼", "┤")
    BOT = hline("└", "┴", "┘")

    def cell(text, w):
        return f" {str(text):<{w}} "

    def mcell(text, w):
        return f" {str(text):^{w}} "

    def build_data_row(q_lines, e_lines, a_lines, scores, grades):
        n = max(len(q_lines), len(e_lines), len(a_lines), 2)
        rows = []
        for idx in range(n):
            q    = q_lines[idx] if idx < len(q_lines) else ""
            e    = e_lines[idx] if idx < len(e_lines) else ""
            a    = a_lines[idx] if idx < len(a_lines) else ""
            line = "│" + cell(q, QW) + "│" + cell(e, EW) + "│" + cell(a, AW)
            for s, g in zip(scores, grades):
                if   idx == 0: line += "│" + mcell(s, MW)
                elif idx == 1: line += "│" + mcell(g, MW)
                else:          line += "│" + mcell("",  MW)
            rows.append(line + "│")
        return rows

    # ── Print table ───────────────────────────────────────────────
    total_w = QW + EW + AW + (MW+3)*len(METRIC_HEADERS) + 4
    print("\n" + "═" * total_w)
    print("  📊  RAGAS EVALUATION RESULTS")
    print("═" * total_w)
    print()
    print(TOP)

    header = "│" + cell("Question", QW) + "│" + cell("Expected", EW) + "│" + cell("Actual", AW)
    for h in METRIC_HEADERS:
        header += "│" + mcell(h, MW)
    print(header + "│")
    print(DIV)

    # ── Data rows ─────────────────────────────────────────────────
    all_scores = {c: [] for c in METRIC_COLS}

    for i, df_row in df.iterrows():
        question = test_questions[i]["question"]
        expected = test_questions[i]["ground_truth"]
        actual   = df_row[answer_col]

        score_strs, grade_strs = [], []
        for col in METRIC_COLS:
            s = safe_score(df_row[col]) if col and col in df_row else float("nan")
            all_scores[col].append(s)
            score_strs.append(fmt(s))
            grade_strs.append(grade(s))

        q_lines = textwrap.wrap(question, QW)
        e_lines = textwrap.wrap(expected, EW)
        a_lines = textwrap.wrap(actual,   AW)

        for line in build_data_row(q_lines, e_lines, a_lines, score_strs, grade_strs):
            print(line)
        print(DIV)

    # ── Average row ───────────────────────────────────────────────
    avg_strs, avg_grades = [], []
    for col in METRIC_COLS:
        valid = [s for s in all_scores[col] if not math.isnan(s)]
        avg   = sum(valid) / len(valid) if valid else float("nan")
        avg_strs.append(fmt(avg))
        avg_grades.append(grade(avg))

    avg_s = "│" + cell("AVERAGE", QW) + "│" + cell("", EW) + "│" + cell("", AW)
    for s in avg_strs:
        avg_s += "│" + mcell(s, MW)
    print(avg_s + "│")

    avg_g = "│" + cell("", QW) + "│" + cell("", EW) + "│" + cell("", AW)
    for g in avg_grades:
        avg_g += "│" + mcell(g, MW)
    print(avg_g + "│")

    print(BOT)

    # ── Legend ────────────────────────────────────────────────────
    print()
    print("  Faith = Faithfulness        → Answer grounded in context? (no hallucination)")
    print("  RR    = Response Relevancy  → Does the answer address the question?")
    print("  CP    = Context Precision   → Relevant chunks ranked above noisy ones?")
    print("  CR    = Context Recall      → Did retriever fetch ALL needed info?")
    print()
    print("  ✅ Good = 0.8–1.0   ⚠️  OK = 0.5–0.8   ❌ Poor = 0.0–0.5")
    print()

    # ── Tips for failing metrics ──────────────────────────────────
    fix_map = {
        "faithfulness":       ("Faith < 0.8",  ["Stricter prompt (already done ✓)", "Try a larger LLM"]),
        rr_col:               ("RR < 0.8",     ["More focused system prompt", "Lower temperature"]),
        "context_precision":  ("CP < 0.8",     ["Increase fetch_k in MMR retriever", "Tune chunk_size"]),
        "context_recall":     ("CR < 0.8",     ["Increase k (fetch more chunks)", "Reduce chunk_size", "Hybrid search (dense + BM25)"]),
    }
    failing = [fix_map[col] for col, s in zip(METRIC_COLS, avg_strs)
               if col in fix_map and s.strip() != "N/A" and float(s) < 0.8]
    if failing:
        print("  💡 Tips for scores below 0.8:\n")
        for title, tips in failing:
            print(f"    {title}:")
            for t in tips:
                print(f"      • {t}")
            print()

    return df


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🚀 RAG Evaluation — Ragas v1.0")
    print("=" * 74)

    collected_data              = run_pipeline_and_collect(TEST_QUESTIONS)
    judge_llm, judge_embeds     = setup_ragas_judge()
    result                      = run_ragas_evaluation(collected_data, judge_llm, judge_embeds)
    df                          = display_results(result, TEST_QUESTIONS)

    df.to_csv("ragas_results.csv", index=False)
    print("  💾 Full results saved to ragas_results.csv")
    print("\n  ✅ Done!\n")