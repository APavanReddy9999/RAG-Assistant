import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import math
import textwrap
from dotenv import load_dotenv
from datasets import Dataset

from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas import evaluate, RunConfig

# ── KEY FIX: import from ragas.llms.BASE not ragas.llms ───────────
# ragas.llms.LangchainLLMWrapper  → DeprecationHelper stub (broken)
# ragas.llms.base.LangchainLLMWrapper → the REAL class (works)
from ragas.llms.base import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from rag_pipeline import load_vector_store, create_retriever, build_rag_chain

load_dotenv()


# ══════════════════════════════════════════════════════════════════
# GROQ n>1 FIX — monkey-patch ChatGroq._generate
# ══════════════════════════════════════════════════════════════════
# Ragas calls generate(prompts, n=3) internally.
# Groq rejects n>1. We patch at the lowest level — right before
# the HTTP request is built — so Groq never receives n>1.

def _patch_chatgroq_n():
    original_generate  = ChatGroq._generate
    original_agenerate = ChatGroq._agenerate

    def patched_generate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs["n"] = 1
        return original_generate(self, messages, stop=stop,
                                 run_manager=run_manager, **kwargs)

    async def patched_agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs["n"] = 1
        return await original_agenerate(self, messages, stop=stop,
                                        run_manager=run_manager, **kwargs)

    ChatGroq._generate  = patched_generate
    ChatGroq._agenerate = patched_agenerate
    print("✅ ChatGroq patched — n=1 forced on all API calls")

_patch_chatgroq_n()


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
    Uses ragas.llms.base.LangchainLLMWrapper — the REAL class.

    WHY ragas.llms.base and not ragas.llms?
      ragas.llms.LangchainLLMWrapper is now a DeprecationHelper stub
      that cannot be instantiated or subclassed normally.
      ragas.llms.base.LangchainLLMWrapper is the actual implementation
      class — importing directly from .base bypasses the stub entirely.
    """
    judge_llm = LangchainLLMWrapper(
        ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )
    )

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


def get_col(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"None of {candidates} found. Available: {list(df.columns)}")


def display_results(result, test_questions: list):
    df         = result.to_pandas()
    answer_col = get_col(df, ["response", "answer"])
    rr_col     = get_col(df, ["response_relevancy", "answer_relevancy"])

    QW, EW, AW, MW = 20, 24, 24, 7

    METRICS = [
        ("Faith", "faithfulness"),
        ("RR",    rr_col),
        ("CP",    "context_precision"),
        ("CR",    "context_recall"),
    ]
    HEADERS = [h for h, _ in METRICS]
    COLS    = [c for _, c in METRICS]

    def hline(left, mid, right, fill="─"):
        parts = [fill*(QW+2), fill*(EW+2), fill*(AW+2)] + [fill*(MW+2) for _ in HEADERS]
        return left + mid.join(parts) + right

    TOP = hline("┌","┬","┐")
    DIV = hline("├","┼","┤")
    BOT = hline("└","┴","┘")

    def cell(t, w):  return f" {str(t):<{w}} "
    def mcell(t, w): return f" {str(t):^{w}} "

    def build_row(ql, el, al, scores, grades):
        n = max(len(ql), len(el), len(al), 2)
        rows = []
        for idx in range(n):
            q = ql[idx] if idx < len(ql) else ""
            e = el[idx] if idx < len(el) else ""
            a = al[idx] if idx < len(al) else ""
            line = "│" + cell(q,QW) + "│" + cell(e,EW) + "│" + cell(a,AW)
            for s, g in zip(scores, grades):
                if   idx == 0: line += "│" + mcell(s, MW)
                elif idx == 1: line += "│" + mcell(g, MW)
                else:          line += "│" + mcell("", MW)
            rows.append(line + "│")
        return rows

    total_w = QW + EW + AW + (MW+3)*len(HEADERS) + 4
    print("\n" + "═"*total_w)
    print("  📊  RAGAS EVALUATION RESULTS")
    print("═"*total_w + "\n")
    print(TOP)

    hdr = "│" + cell("Question",QW) + "│" + cell("Expected",EW) + "│" + cell("Actual",AW)
    for h in HEADERS: hdr += "│" + mcell(h, MW)
    print(hdr + "│")
    print(DIV)

    all_scores = {c: [] for c in COLS}

    for i, df_row in df.iterrows():
        question = test_questions[i]["question"]
        expected = test_questions[i]["ground_truth"]
        actual   = df_row[answer_col]

        score_strs, grade_strs = [], []
        for col in COLS:
            s = safe_score(df_row[col]) if col in df_row else float("nan")
            all_scores[col].append(s)
            score_strs.append(fmt(s))
            grade_strs.append(grade(s))

        for line in build_row(
            textwrap.wrap(question, QW),
            textwrap.wrap(expected, EW),
            textwrap.wrap(actual,   AW),
            score_strs, grade_strs
        ):
            print(line)
        print(DIV)

    avg_strs, avg_grades = [], []
    for col in COLS:
        valid = [s for s in all_scores[col] if not math.isnan(s)]
        avg   = sum(valid)/len(valid) if valid else float("nan")
        avg_strs.append(fmt(avg))
        avg_grades.append(grade(avg))

    row_s = "│" + cell("AVERAGE",QW) + "│" + cell("",EW) + "│" + cell("",AW)
    for s in avg_strs: row_s += "│" + mcell(s, MW)
    print(row_s + "│")

    row_g = "│" + cell("",QW) + "│" + cell("",EW) + "│" + cell("",AW)
    for g in avg_grades: row_g += "│" + mcell(g, MW)
    print(row_g + "│")

    print(BOT)
    print()
    print("  Faith = Faithfulness       → Answer grounded in context? (no hallucination)")
    print("  RR    = Response Relevancy → Does the answer address the question?")
    print("  CP    = Context Precision  → Relevant chunks ranked above noisy ones?")
    print("  CR    = Context Recall     → Did retriever fetch ALL needed info?")
    print()
    print("  ✅ Good = 0.8–1.0   ⚠️  OK = 0.5–0.8   ❌ Poor = 0.0–0.5")
    print()

    fixes = {
        "faithfulness":      ("Faith < 0.8", ["Stricter prompt (done ✓)", "Try larger LLM"]),
        rr_col:              ("RR < 0.8",    ["More focused system prompt", "Lower temperature"]),
        "context_precision": ("CP < 0.8",    ["Increase fetch_k in MMR", "Tune chunk_size"]),
        "context_recall":    ("CR < 0.8",    ["Increase k", "Reduce chunk_size", "Hybrid search"]),
    }
    failing = [fixes[col] for col, s in zip(COLS, avg_strs)
               if col in fixes and s.strip() != "N/A" and float(s) < 0.8]
    if failing:
        print("  💡 Tips for scores below 0.8:\n")
        for title, tips in failing:
            print(f"    {title}:")
            for t in tips: print(f"      • {t}")
            print()

    return df


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🚀 RAG Evaluation — Ragas v1.0")
    print("=" * 74)

    collected_data          = run_pipeline_and_collect(TEST_QUESTIONS)
    judge_llm, judge_embeds = setup_ragas_judge()
    result                  = run_ragas_evaluation(collected_data, judge_llm, judge_embeds)
    df                      = display_results(result, TEST_QUESTIONS)

    df.to_csv("ragas_results.csv", index=False)
    print("  💾 Full results saved to ragas_results.csv")
    print("\n  ✅ Done!\n")