import pandas as pd
import json
import re
import os
from typing import Dict

# Configure for both datasets
DATASETS = [
    ("out/out_v2.csv", "out/out_v2_with_enhanced_features.csv"),
    ("out/dolly_inference_results_llama2_awq.csv", "out/dolly_inference_results_llama2_awq_with_enhanced_features.csv")
]

# -----------------------------
# Enhanced Feature Extraction
# -----------------------------
PUNCT = set(".,;:!?")

# Keyword dictionaries for intent detection
DETAILED_KEYWORDS = ["detailed", "comprehensive", "elaborate", "explain in depth", "thorough", "extensively", "in detail"]
BRIEF_KEYWORDS = ["brief", "short", "quick", "summary", "tldr", "concise", "briefly", "summarize"]
LIST_KEYWORDS = ["list", "enumerate", "bullet points", "steps", "numbered"]
CREATIVE_KEYWORDS = ["write a story", "create", "imagine", "compose", "generate"]
CODE_KEYWORDS = ["implement", "code", "function", "class", "program", "script", "algorithm"]
ANALYTICAL_KEYWORDS = ["analyze", "compare", "evaluate", "assess", "examine", "pros and cons", "advantages", "disadvantages"]
FORMAT_KEYWORDS = ["table", "json", "xml", "markdown", "csv", "format"]

QUESTION_WORDS = ["who", "what", "where", "when", "why", "how", "which", "whose", "whom"]


def count_sentences(text: str) -> int:
    """Count sentences using basic punctuation."""
    sentence_endings = re.findall(r'[.!?]+', text)
    return max(1, len(sentence_endings))


def count_keyword_matches(text: str, keywords: list) -> int:
    """Count how many keywords from a list appear in text (case insensitive)."""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def extract_enhanced_features(prompt: str) -> Dict[str, float]:
    text = prompt or ""
    n_chars = len(text)
    n_lines = text.count("\n") + 1 if text else 0
    n_words = len(text.split())
    words = text.split()

    # ==========================================
    # BASIC FEATURES (Original)
    # ==========================================
    n_qmarks = text.count("?")
    n_emarks = text.count("!")
    n_backticks = text.count("`")
    n_fences = len(re.findall(r"```", text))
    has_url = int(bool(re.search(r"https?://", text)))
    has_table = int(("|" in text and "---" in text) or ("table" in text.lower()))
    has_list = int(bool(re.search(r"^\s*[-*]\s", text, flags=re.M)))

    math_pattern = r"(\$[^$]+\$)|\\begin\{equation\}|\d+\s*[\+\-\*/]\s*\d+"
    has_math = int(bool(re.search(math_pattern, text)))

    caps_ratio = sum(1 for c in text if c.isupper()) / max(1, n_chars)
    punct_density = sum(1 for c in text if c in PUNCT) / max(1, n_chars)
    avg_line_len = n_chars / max(1, n_lines)
    avg_word_len = sum(len(w) for w in words) / max(1, n_words)

    # ==========================================
    # LINGUISTIC COMPLEXITY
    # ==========================================
    n_sentences = count_sentences(text)
    avg_sentence_len = n_words / max(1, n_sentences)

    # Unique word ratio (vocabulary richness)
    unique_words = len(set(w.lower() for w in words if w.isalnum()))
    unique_word_ratio = unique_words / max(1, n_words)

    # Estimate average syllables per word (simple heuristic)
    vowels = set('aeiouAEIOU')
    total_syllables = sum(max(1, sum(1 for c in word if c in vowels)) for word in words)
    avg_syllables = total_syllables / max(1, n_words)

    # ==========================================
    # STRUCTURAL FEATURES
    # ==========================================
    n_paragraphs = len([p for p in text.split('\n\n') if p.strip()])
    n_paragraphs = max(1, n_paragraphs)

    # Count different list types
    numbered_lists = len(re.findall(r'^\s*\d+[\.\)]\s', text, flags=re.M))
    bullet_lists = len(re.findall(r'^\s*[-*•]\s', text, flags=re.M))

    # Indentation/code blocks
    indented_lines = len(re.findall(r'^\s{4,}', text, flags=re.M))

    # ==========================================
    # INTENT/INSTRUCTION KEYWORDS
    # ==========================================
    has_detailed_request = count_keyword_matches(text, DETAILED_KEYWORDS)
    has_brief_request = count_keyword_matches(text, BRIEF_KEYWORDS)
    has_list_request = count_keyword_matches(text, LIST_KEYWORDS)
    has_creative_request = count_keyword_matches(text, CREATIVE_KEYWORDS)
    has_code_request = count_keyword_matches(text, CODE_KEYWORDS)
    has_analytical_request = count_keyword_matches(text, ANALYTICAL_KEYWORDS)
    has_format_request = count_keyword_matches(text, FORMAT_KEYWORDS)

    # ==========================================
    # QUESTION COMPLEXITY
    # ==========================================
    # Number of questions
    n_questions = text.count('?')

    # Question type diversity
    question_types = sum(1 for qw in QUESTION_WORDS if qw in text.lower())

    # Sub-questions (using "and", "also", "additionally")
    sub_question_markers = text.lower().count(' and ') + text.lower().count(' also ') + text.lower().count(' additionally ')

    # Conditional questions
    has_conditional = int(bool(re.search(r'\bif\b.*\bthen\b', text.lower())))

    # ==========================================
    # SPECIFICITY INDICATORS
    # ==========================================
    # Numbers and dates
    numbers_count = len(re.findall(r'\b\d+\b', text))

    # Proper nouns (simple heuristic: capitalized words mid-sentence)
    proper_nouns = len(re.findall(r'(?<=\s)[A-Z][a-z]+', text))
    proper_noun_density = proper_nouns / max(1, n_words)

    # Examples presence
    has_examples = int(bool(re.search(r'\b(for example|such as|like|e\.g\.|i\.e\.)\b', text.lower())))

    # ==========================================
    # ADDITIONAL USEFUL FEATURES
    # ==========================================
    # Presence of quotes
    has_quotes = int('"' in text or "'" in text or '"' in text or '"' in text)

    # Presence of parentheses (often for clarification)
    parentheses_count = text.count('(') + text.count('[')

    # Colon usage (often precedes lists or explanations)
    colon_count = text.count(':')

    return {
        # Basic features
        "char_len": n_chars,
        "line_count": n_lines,
        "word_count": n_words,
        "question_marks": n_qmarks,
        "exclamation_marks": n_emarks,
        "backticks": n_backticks,
        "code_fences": n_fences,
        "has_url": has_url,
        "has_table": has_table,
        "has_list": has_list,
        "has_math": has_math,
        "caps_ratio": round(caps_ratio, 5),
        "punct_density": round(punct_density, 5),
        "avg_line_len": round(avg_line_len, 3),
        "avg_word_len": round(avg_word_len, 3),

        # Linguistic complexity
        "sentence_count": n_sentences,
        "avg_sentence_len": round(avg_sentence_len, 3),
        "unique_word_ratio": round(unique_word_ratio, 5),
        "avg_syllables": round(avg_syllables, 3),

        # Structural features
        "paragraph_count": n_paragraphs,
        "numbered_lists": numbered_lists,
        "bullet_lists": bullet_lists,
        "indented_lines": indented_lines,

        # Intent keywords
        "detailed_keywords": has_detailed_request,
        "brief_keywords": has_brief_request,
        "list_keywords": has_list_request,
        "creative_keywords": has_creative_request,
        "code_keywords": has_code_request,
        "analytical_keywords": has_analytical_request,
        "format_keywords": has_format_request,

        # Question complexity
        "question_count": n_questions,
        "question_types": question_types,
        "sub_questions": sub_question_markers,
        "has_conditional": has_conditional,

        # Specificity
        "numbers_count": numbers_count,
        "proper_noun_density": round(proper_noun_density, 5),
        "has_examples": has_examples,

        # Additional
        "has_quotes": has_quotes,
        "parentheses_count": parentheses_count,
        "colon_count": colon_count,
    }


# -----------------------------
# Process each dataset
# -----------------------------
for input_path, output_path in DATASETS:
    print(f"\n{'='*60}")
    print(f"Processing: {input_path}")
    print(f"{'='*60}")

    if not os.path.exists(input_path):
        print(f"⚠️  File not found: {input_path}, skipping...")
        continue

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows.")

    # Detect prompt column
    if "prompt_text" in df.columns:
        prompt_col = "prompt_text"
    elif "prompt" in df.columns:
        prompt_col = "prompt"
    else:
        print("⚠️  Could not find 'prompt_text' or 'prompt' column, skipping...")
        continue

    print(f"Using prompt column: {prompt_col}")

    # Compute enhanced features
    print("Extracting enhanced features...")
    df["features_json"] = df[prompt_col].apply(
        lambda p: json.dumps(extract_enhanced_features(str(p)))
    )

    # Save output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Saved to: {output_path}")

print("\n" + "="*60)
print("All datasets processed with enhanced features!")
print("="*60)
