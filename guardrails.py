"""
guardrails.py
─────────────────────────────────────────────────────────────────────
Security layer for the Healthcare RAG pipeline.

LOGGING BEHAVIOUR:
  All guard logs go to guardrails.log FILE only.
  Nothing from guardrails prints to the terminal.
  This keeps the terminal output clean for the user.

  WHY LOG TO FILE:
  In production you need an audit trail of every blocked request
  (required for HIPAA compliance) but the user should never see
  internal log messages — only clean answers or blocked messages.
"""

import re
import logging

# ── Log to FILE only — never to terminal ──────────────────────────
# WHY FileHandler not StreamHandler:
# StreamHandler prints to terminal (stdout/stderr).
# FileHandler writes to a .log file silently.
# The user sees only clean output; security team reads the log file.
logger = logging.getLogger("guardrails")
logger.setLevel(logging.INFO)
logger.propagate = False   # stop logs bubbling up to root logger (which prints to terminal)

file_handler = logging.FileHandler("guardrails.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(file_handler)


# ══════════════════════════════════════════════════════════════════
# GUARD RESULT
# ══════════════════════════════════════════════════════════════════

class GuardResult:
    """
    Returned by every guard function.
      passed     → True = safe to proceed, False = blocked
      guard_name → which guard fired (for logs)
      reason     → why it was blocked (for logs)
      message    → what to show the user (polite, informative)
    """
    def __init__(self, passed: bool, guard_name: str,
                 reason: str = "", message: str = ""):
        self.passed     = passed
        self.guard_name = guard_name
        self.reason     = reason
        self.message    = message

    def __repr__(self):
        status = "✅ PASS" if self.passed else "🚫 BLOCK"
        return f"[{status}] {self.guard_name}: {self.reason}"


# ══════════════════════════════════════════════════════════════════
# INPUT GUARD 1 — PROMPT INJECTION DETECTOR
# ══════════════════════════════════════════════════════════════════

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|rules?|prompts?|context)",
    r"forget\s+(all\s+)?(previous|prior|your)\s+(instructions?|rules?|prompts?)",
    r"you\s+are\s+now\s+a?\s*(different|new|another)?\s*(ai|assistant|bot|model)",
    r"new\s+(instructions?|rules?|persona|role|prompt)",
    r"disregard\s+(all\s+)?(previous|prior|your)",
    r"override\s+(your\s+)?(instructions?|rules?|settings?|guidelines?)",
    r"act\s+as\s+(if\s+you\s+are\s+)?(a\s+)?(different|unrestricted|free)",
    r"pretend\s+(you\s+are|to\s+be)\s+",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"dan\s+mode",
    r"reveal\s+(your\s+)?(system\s+)?prompt",
    r"what\s+(are|is)\s+your\s+(system\s+)?prompt",
    r"show\s+me\s+your\s+(instructions?|rules?|system)",
]

def check_prompt_injection(question: str) -> GuardResult:
    q_lower = question.lower().strip()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, q_lower):
            logger.warning(f"PROMPT INJECTION | pattern={pattern} | q={question[:80]}")
            return GuardResult(
                passed=False,
                guard_name="Prompt Injection Guard",
                reason=f"Matched injection pattern: {pattern}",
                message=(
                    "⚠️  I'm a Healthcare Assistant and can only answer "
                    "medical and health-related questions. I cannot modify "
                    "my instructions or reveal system information."
                )
            )
    return GuardResult(passed=True, guard_name="Prompt Injection Guard")


# ══════════════════════════════════════════════════════════════════
# INPUT GUARD 2 — PII REQUEST DETECTOR
# ══════════════════════════════════════════════════════════════════

PII_FIELD_KEYWORDS = [
    "aadhaar", "aadhar", "pan number", "passport",
    "phone number", "mobile number", "contact number",
    "email", "email address",
    "bank account", "account number", "ifsc", "ifsc code",
    "policy number", "insurance number",
    "address", "home address", "residential address",
    "date of birth", "dob", "birth date",
    "patient id",
]

REQUEST_INTENT_KEYWORDS = [
    "what is", "what's", "tell me", "give me", "show me",
    "reveal", "provide", "share", "disclose",
    "find", "lookup", "look up", "fetch", "get",
    "list", "display",
]

def check_pii_request(question: str) -> GuardResult:
    q_lower = question.lower()
    has_pii_field = any(kw in q_lower for kw in PII_FIELD_KEYWORDS)
    has_request   = any(kw in q_lower for kw in REQUEST_INTENT_KEYWORDS)

    if has_pii_field and has_request:
        logger.warning(f"PII REQUEST | q={question[:80]}")
        return GuardResult(
            passed=False,
            guard_name="PII Request Guard",
            reason="Question requests personally identifiable information",
            message=(
                "🔒 I cannot provide personal information such as "
                "Aadhaar numbers, phone numbers, bank details, or "
                "addresses. This information is confidential and protected. "
                "Please contact the hospital administration directly."
            )
        )
    return GuardResult(passed=True, guard_name="PII Request Guard")


# ══════════════════════════════════════════════════════════════════
# INPUT GUARD 3 — PATIENT RECORD REQUEST DETECTOR
# ══════════════════════════════════════════════════════════════════

KNOWN_PATIENT_NAMES = [
    "rahul sharma", "rahul",
    "sneha reddy", "sneha",
    "p-10234", "p-10235",
]

MEDICAL_RECORD_KEYWORDS = [
    "diagnosis", "diagnosed", "condition", "disease", "illness",
    "medication", "medicine", "drug", "prescription", "prescribed",
    "doctor", "physician", "treatment", "therapy",
    "lab report", "test result", "blood test", "hba1c",
    "medical record", "health record", "patient record",
    "insurance", "policy", "financial",
    "last visit", "appointment", "history",
    "restricted", "confidential",
]

def check_patient_record_request(question: str) -> GuardResult:
    q_lower = question.lower()
    has_patient_name   = any(name in q_lower for name in KNOWN_PATIENT_NAMES)
    has_medical_record = any(kw in q_lower for kw in MEDICAL_RECORD_KEYWORDS)

    if has_patient_name and has_medical_record:
        logger.warning(f"PATIENT RECORD REQUEST | q={question[:80]}")
        return GuardResult(
            passed=False,
            guard_name="Patient Record Guard",
            reason="Question requests specific patient medical records",
            message=(
                "🏥 I cannot share individual patient medical records, "
                "diagnoses, medications, or treatment history. "
                "Patient data is strictly confidential under healthcare "
                "privacy regulations. Please consult the treating physician "
                "or hospital records department for authorised access."
            )
        )
    return GuardResult(passed=True, guard_name="Patient Record Guard")


# ══════════════════════════════════════════════════════════════════
# INPUT GUARD 4 — OFF-TOPIC DETECTOR
# ══════════════════════════════════════════════════════════════════

MEDICAL_TOPIC_KEYWORDS = [
    "fever", "headache", "cough", "cold", "pain", "ache", "nausea",
    "vomit", "diarrhea", "fatigue", "tired", "dizzy", "rash", "swelling",
    "bleeding", "burn", "wound", "cut", "nosebleed", "breath", "breathing",
    "diabetes", "hypertension", "blood pressure", "infection", "flu",
    "migraine", "allergy", "asthma", "cancer", "heart", "stroke",
    "inflammation", "virus", "bacteria", "chronic",
    "paracetamol", "ibuprofen", "medicine", "medication", "drug",
    "tablet", "dose", "dosage", "prescription", "antibiotic",
    "treatment", "diagnos", "symptom", "doctor", "hospital", "clinic",
    "nurse", "surgery", "vaccine", "vaccination", "prevent", "health",
    "medical", "patient", "first aid", "emergency", "injury",
    "blood", "heart", "liver", "kidney", "lung", "brain", "bone",
    "muscle", "skin", "stomach", "throat", "nose", "eye", "ear",
    "temperature", "immune",
]

def check_off_topic(question: str) -> GuardResult:
    q_lower = question.lower()

    if len(q_lower.split()) <= 3:
        return GuardResult(passed=True, guard_name="Off-Topic Guard",
                           reason="Short greeting allowed")

    has_medical_topic = any(kw in q_lower for kw in MEDICAL_TOPIC_KEYWORDS)

    if not has_medical_topic:
        logger.info(f"OFF-TOPIC | q={question[:80]}")
        return GuardResult(
            passed=False,
            guard_name="Off-Topic Guard",
            reason="No medical keywords found in question",
            message=(
                "🩺 I'm a Healthcare Assistant and can only answer "
                "medical and health-related questions. Your question "
                "appears to be outside my scope. Please ask me about "
                "symptoms, conditions, medications, or first aid."
            )
        )
    return GuardResult(passed=True, guard_name="Off-Topic Guard")


# ══════════════════════════════════════════════════════════════════
# OUTPUT GUARD 1 — PII LEAK DETECTOR
# ══════════════════════════════════════════════════════════════════

PII_PATTERNS = {
    "Aadhaar Number":   r"\b\d{4}\s\d{4}\s\d{4}\b",
    "Indian Phone":     r"\+91\s?\d{5}\s?\d{5}",
    "Bank Account":     r"\b\d{12}\b",
    "IFSC Code":        r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
    "Insurance Policy": r"\b(SHI|ILI)-\d{10}\b",
    "Email Address":    r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
    "Patient ID":       r"\bP-\d{5}\b",
}

def check_pii_in_output(answer: str) -> GuardResult:
    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, answer):
            logger.warning(f"PII LEAK IN OUTPUT | type={pii_type} | a={answer[:80]}")
            return GuardResult(
                passed=False,
                guard_name="PII Leak Guard",
                reason=f"Answer contains {pii_type}",
                message=(
                    "🔒 I cannot share this information as it contains "
                    "personal or sensitive data. Please contact the "
                    "hospital administration for authorised access."
                )
            )
    return GuardResult(passed=True, guard_name="PII Leak Guard")


# ══════════════════════════════════════════════════════════════════
# OUTPUT GUARD 2 — SENSITIVE PATIENT DATA IN ANSWER
# ══════════════════════════════════════════════════════════════════

SENSITIVE_OUTPUT_KEYWORDS = [
    "diagnosis", "diagnosed", "condition",
    "medication", "prescribed", "taking",
    "bank account", "account number", "ifsc",
    "insurance", "policy number",
    "aadhaar", "aadhar",
    "non-compliance", "restricted", "confidential",
    "do not disclose", "history of",
    "stress-related", "lab report", "test result",
]

def check_sensitive_patient_data_in_output(answer: str) -> GuardResult:
    a_lower = answer.lower()
    has_patient_name    = any(name in a_lower for name in KNOWN_PATIENT_NAMES)
    has_sensitive_topic = any(kw in a_lower for kw in SENSITIVE_OUTPUT_KEYWORDS)

    if has_patient_name and has_sensitive_topic:
        logger.warning(f"SENSITIVE PATIENT DATA IN OUTPUT | a={answer[:80]}")
        return GuardResult(
            passed=False,
            guard_name="Sensitive Patient Data Guard",
            reason="Answer reveals patient-specific sensitive information",
            message=(
                "🏥 I cannot share individual patient medical or financial "
                "records. This information is confidential. Please contact "
                "the appropriate department for authorised access."
            )
        )
    return GuardResult(passed=True, guard_name="Sensitive Patient Data Guard")


# ══════════════════════════════════════════════════════════════════
# OUTPUT GUARD 3 — HALLUCINATION SIGNAL DETECTOR (SOFT WARNING)
# ══════════════════════════════════════════════════════════════════

HALLUCINATION_SIGNALS = [
    "i think", "i believe", "i assume", "i suppose",
    "probably", "might be", "could be",
    "i'm not sure but", "i'm not certain",
    "as far as i know", "to my knowledge",
    "i don't have specific information",
    "based on general knowledge",
    "generally speaking",
]

def check_hallucination_signals(answer: str) -> GuardResult:
    a_lower = answer.lower()
    for signal in HALLUCINATION_SIGNALS:
        if signal in a_lower:
            logger.info(f"HALLUCINATION SIGNAL | signal='{signal}' | a={answer[:80]}")
            return GuardResult(
                passed=False,
                guard_name="Hallucination Signal Guard",
                reason=f"Uncertainty signal detected: '{signal}'",
                message=(
                    answer +
                    "\n\n⚠️  Note: This response may contain uncertain "
                    "information. Please verify with a qualified healthcare "
                    "professional before making any medical decisions."
                )
            )
    return GuardResult(passed=True, guard_name="Hallucination Signal Guard")


# ══════════════════════════════════════════════════════════════════
# MAIN FUNCTIONS — called from rag_pipeline.py
# ══════════════════════════════════════════════════════════════════

def run_input_guards(question: str) -> GuardResult:
    """Run all input guards. Returns on first failure (fail fast)."""
    for guard_fn in [
        check_prompt_injection,
        check_pii_request,
        check_patient_record_request,
        check_off_topic,
    ]:
        result = guard_fn(question)
        if not result.passed:
            return result

    logger.info(f"INPUT GUARDS: all passed | q={question[:60]}")
    return GuardResult(passed=True, guard_name="All Input Guards")


def run_output_guards(answer: str) -> GuardResult:
    """Run all output guards. Hard blocks first, soft warning last."""
    for guard_fn in [
        check_pii_in_output,
        check_sensitive_patient_data_in_output,
    ]:
        result = guard_fn(answer)
        if not result.passed:
            return result

    hallucination = check_hallucination_signals(answer)
    if not hallucination.passed:
        return hallucination

    logger.info(f"OUTPUT GUARDS: all passed | a={answer[:60]}")
    return GuardResult(passed=True, guard_name="All Output Guards")


def print_guard_result(label: str, result: GuardResult):
    """Pretty-print a guard result — used in test_guardrails.py only."""
    status = "✅ ALLOWED" if result.passed else "🚫 BLOCKED"
    print(f"\n{'─'*62}")
    print(f"  {label}")
    print(f"  Status : {status}")
    print(f"  Guard  : {result.guard_name}")
    if not result.passed:
        print(f"  Msg    : {result.message[:100]}...")
    print(f"{'─'*62}")