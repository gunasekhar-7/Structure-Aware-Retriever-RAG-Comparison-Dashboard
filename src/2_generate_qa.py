"""
Phase 2: Synthetic QA Dataset Generator for STAIR Fine-Tuning.

Uses Groq Llama-3.3-70B API (with robust offline semantic fallback)
to generate high-quality, realistic developer questions per leaf section.
Splits data into train and validation sets formatted exactly per the STAIR paper.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Attempt to load .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_toc_and_leaves(json_path: Path) -> Tuple[Optional[str], Optional[Dict[str, str]], Optional[str]]:
    """Load the extracted Table of Contents and leaf sections from JSON file."""
    if not json_path.exists():
        print(f"[ERROR] JSON file not found: {json_path}")
        print("  Please run 1_extract_toc.py or sample_pdf_processor.py first.")
        return None, None, None
        
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        toc_string = data.get("toc_prompt", "")
        leaves = data.get("leaves", {})
        source = data.get("source", "Technical Documentation")
        print(f"[INFO] Loaded {len(leaves)} leaf sections from {json_path}")
        return toc_string, leaves, source
    except Exception as e:
        print(f"[ERROR] Failed to parse JSON file {json_path}: {e}")
        return None, None, None


def query_groq_api(prompt: str, api_key: str, model: str = "llama-3.3-70b-versatile", max_retries: int = 3) -> str:
    """Query Groq API with exponential backoff for rate limits."""
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert AI engineer evaluating documentation retrieval systems. "
                    "Generate realistic, challenging developer queries that are answered by the given text. "
                    "Output ONLY valid JSON as requested."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 400
    }

    backoff = 2.0
    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, headers=headers, json=data, timeout=30)
            if response.status_code == 429:
                retry_after = float(response.headers.get("retry-after", backoff))
                print(f"  [WAIT] Rate limit hit. Backing off for {retry_after:.1f}s...")
                time.sleep(retry_after)
                backoff *= 2.0
                continue
            response.raise_for_status()
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                print(f"  [WARNING] Groq API call failed after {max_retries} attempts: {e}")
                return ""
            time.sleep(backoff)
            backoff *= 1.5
    return ""


def parse_questions_from_response(raw_text: str) -> List[str]:
    """Parse JSON list of questions from raw model completion."""
    if not raw_text:
        return []
    
    # Strip markdown code fences if present
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[len("```"):].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [str(q).strip() for q in parsed if str(q).strip()]
    except json.JSONDecodeError:
        pass

    # Fallback regex extraction for bulleted/numbered lines
    lines = raw_text.split("\n")
    questions = []
    for line in lines:
        match = re.match(r"^(?:\d+[\.\)]|\-|\*)\s*(.*?\?)$", line.strip())
        if match:
            questions.append(match.group(1).strip())
    return questions


def generate_semantic_fallback_questions(leaf_title: str, leaf_text: str) -> List[str]:
    """
    High-quality deterministic fallback question generator when Groq API key is not available.
    Generates varied, realistic technical queries based on content analysis.
    """
    questions = []
    clean_title = leaf_title.replace("Section ", "").replace("Chapter ", "").strip()
    
    # Extract key technical tokens from text
    code_terms = re.findall(r"`([^`]+)`", leaf_text)
    caps_terms = re.findall(r"\b([A-Z][a-zA-Z0-9_]{3,})\b", leaf_text)
    keywords = list(dict.fromkeys(code_terms + caps_terms))[:4]
    
    if "async" in leaf_text.lower() or "event loop" in leaf_text.lower():
        questions.append(f"How do I handle asynchronous execution and task lifecycles in {clean_title}?")
        questions.append(f"What happens when blocking calls are made inside an event loop in {clean_title}?")
        questions.append(f"Best practices for structured concurrency and exception groups in {clean_title}?")
    elif "fastapi" in leaf_text.lower() or "route" in leaf_text.lower() or "api" in leaf_text.lower():
        questions.append(f"How to declare path parameters, request bodies, and validation with {clean_title}?")
        questions.append(f"How does dependency injection manage session teardown in {clean_title}?")
        questions.append(f"How do I configure background tasks or security tokens in {clean_title}?")
    elif "collection" in leaf_text.lower() or "data structure" in leaf_text.lower() or "list" in leaf_text.lower():
        questions.append(f"What are the time complexities and performance trade-offs in {clean_title}?")
        questions.append(f"When should I use structural pattern matching versus standard conditionals in {clean_title}?")
        questions.append(f"How are type annotations enforced at runtime in {clean_title}?")
    else:
        kw1 = keywords[0] if len(keywords) > 0 else "the core feature"
        kw2 = keywords[1] if len(keywords) > 1 else "syntax constructs"
        questions.append(f"How do I correctly implement {kw1} in {clean_title}?")
        questions.append(f"What are the architectural considerations when using {kw2} in {clean_title}?")
        questions.append(f"How does {clean_title} behave during runtime error conditions?")

    return questions


def format_stair_training_example(book_title: str, toc_string: str, query: str, target_leaf: str) -> Dict[str, str]:
    """Format a training example exactly according to Table 2 of the STAIR paper."""
    # Ensure leaf target is clean
    clean_target = target_leaf.strip()
    if clean_target.startswith("leaf "):
        clean_target = clean_target[len("leaf "):].strip()
        
    prompt = (
        f"### Book: {book_title}\n"
        f"Pick the best section from the table of contents below which can answer the user query below. "
        f"Only generate the section name, do not generate any explanation!\n"
        f"### Table of Contents\n{toc_string}\n"
        f"### Query: {query}\n"
        f"### Section: leaf "
    )
    return {"prompt": prompt, "completion": clean_target}


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic QA pairs for STAIR fine-tuning.")
    parser.add_argument("--model", type=str, default="llama-3.3-70b-versatile", help="Groq model to use.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of leaf sections to process.")
    args = parser.parse_args()

    project_root = get_project_root()
    data_dir = project_root / "data"
    json_path = data_dir / "toc_and_sections.json"

    toc_string, leaves, book_title = load_toc_and_leaves(json_path)
    if not toc_string or not leaves:
        return

    book_title = book_title or "Python & FastAPI Production Documentation"
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

    if groq_api_key:
        print(f"[INFO] GROQ_API_KEY detected. Using Groq model: {args.model}")
    else:
        print("[INFO] No GROQ_API_KEY found. Utilizing high-quality deterministic semantic fallback generator.")
        print("       (To use Groq API, set GROQ_API_KEY in your .env or shell environment)")

    all_examples: List[Dict[str, str]] = []
    leaf_items = list(leaves.items())
    if args.limit:
        leaf_items = leaf_items[:args.limit]

    print(f"\n[INFO] Generating questions across {len(leaf_items)} leaf sections...")

    for idx, (leaf_title, leaf_text) in enumerate(leaf_items, 1):
        clean_title = leaf_title.strip()
        questions: List[str] = []

        if groq_api_key:
            prompt = (
                f"Documentation Section:\n"
                f"Title: {clean_title}\n"
                f"Content:\n{leaf_text[:1800]}\n\n"
                f"Task: Generate 3 realistic, distinct technical questions that a software engineer would ask, "
                f"which are answered SPECIFICALLY by this section.\n"
                f"Format your output strictly as a JSON list of strings: [\"question 1\", \"question 2\", \"question 3\"]"
            )
            raw_response = query_groq_api(prompt, groq_api_key, model=args.model)
            questions = parse_questions_from_response(raw_response)
            time.sleep(0.3)  # Gentle throttle

        # If API call failed or no key, use semantic fallback generator
        if not questions:
            questions = generate_semantic_fallback_questions(clean_title, leaf_text)

        for q in questions:
            example = format_stair_training_example(book_title, toc_string, q, clean_title)
            all_examples.append(example)

        print(f"  [{idx}/{len(leaf_items)}] {clean_title} -> {len(questions)} queries generated")

    if not all_examples:
        print("[ERROR] No QA pairs generated.")
        return

    # Shuffle and split into Train (80%) and Validation (20%)
    random.seed(42)
    random.shuffle(all_examples)
    
    val_size = max(1, int(len(all_examples) * 0.20))
    train_examples = all_examples[val_size:]
    val_examples = all_examples[:val_size]

    train_path = data_dir / "synthetic_qa_train.jsonl"
    val_path = data_dir / "synthetic_qa_val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for ex in val_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n[SUCCESS] Successfully generated {len(all_examples)} total QA examples.")
    print(f"  - Training set:   {len(train_examples)} examples saved to {train_path.name}")
    print(f"  - Validation set: {len(val_examples)} examples saved to {val_path.name}")
    print(f"\nSample Training Example:")
    print(f"  Prompt snippet:\n    {train_examples[0]['prompt'][-120:]}")
    print(f"  Target completion:\n    {train_examples[0]['completion']}")


if __name__ == "__main__":
    main()
