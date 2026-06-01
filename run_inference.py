import os
import csv
import json
import re
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List

from tqdm import tqdm
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# =========================
# DEFAULT CONFIG
# =========================

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_MODEL_PATH = "/tmp/Qwen3-4B-Thinking-2507"

BAD_BOX_WORDS = {
    "",
    "answer",
    "final answer",
    "ans",
    "[ans]",
    "...",
    "?",
    "your answer",
    "the answer",
    "result",
    "value",
    "choice",
    "letter",
    "blank",
}


# =========================
# ENVIRONMENT
# =========================

def configure_environment() -> None:
    """
    Set environment variables used for the vLLM/DSMLP inference setup.
    """
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ["VLLM_USE_DEEP_GEMM"] = "0"

    os.environ["HF_HOME"] = "/tmp/hf_cache"
    os.environ["HF_HUB_CACHE"] = "/tmp/hf_cache/hub"
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache/transformers"
    os.environ["TMPDIR"] = "/tmp"

    os.makedirs("/tmp/hf_cache", exist_ok=True)


# =========================
# BOX EXTRACTION / CLEANING
# =========================

def extract_boxed_values(text: str) -> List[str]:
    """
    Extract values inside LaTeX \\boxed{...}, including simple nested braces.
    """
    text = str(text)
    values = []
    start_pat = r"\boxed{"
    i = 0

    while True:
        start = text.find(start_pat, i)
        if start == -1:
            break

        j = start + len(start_pat)
        depth = 1
        content_start = j

        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1

        if depth == 0:
            values.append(text[content_start:j - 1].strip())
            i = j
        else:
            break

    return values


def is_placeholder_box(value: str) -> bool:
    cleaned = str(value).strip().lower()
    cleaned = cleaned.replace("\\text{", "").replace("}", "").strip()
    return cleaned in BAD_BOX_WORDS


def get_real_boxes(text: str) -> List[str]:
    boxes = extract_boxed_values(text)
    return [b for b in boxes if not is_placeholder_box(b)]


def clean_to_last_real_box(response: str) -> Optional[str]:
    """
    Return only the last real boxed answer, e.g. \\boxed{4,16}.
    If no real box exists, return None.
    """
    real_boxes = get_real_boxes(response)

    if not real_boxes:
        return None

    last = real_boxes[-1].strip()
    return f"\\boxed{{{last}}}"


def is_bad_response_text(response: str) -> bool:
    """
    Detect missing, placeholder, or likely cutoff/rambling answers.
    """
    if not isinstance(response, str):
        return True

    text = response.strip()

    if text == "":
        return True

    boxes = extract_boxed_values(text)

    if len(boxes) == 0:
        return True

    real_boxes = [b for b in boxes if not is_placeholder_box(b)]

    if len(real_boxes) == 0:
        return True

    lower_text = text.lower()

    fake_patterns = [
        r"\\boxed\{\s*\.\.\.\s*\}",
        r"\\boxed\{\s*\?\s*\}",
        r"\\boxed\{\s*answer\s*\}",
        r"\\boxed\{\s*value\s*\}",
        r"\\boxed\{\s*result\s*\}",
        r"\\boxed\{\s*choice\s*\}",
        r"\\boxed\{\s*letter\s*\}",
    ]

    for pat in fake_patterns:
        if re.search(pat, lower_text):
            return True

    # Long rambling answers were a common failure mode.
    if len(text) > 1500:
        bad_phrases = [
            "wait,",
            "hmm",
            "let me check",
            "let me verify",
            "is this correct",
            "maybe",
            "i think",
            "hold on",
            "actually",
        ]

        if any(p in lower_text for p in bad_phrases):
            return True

    return False


# =========================
# PROMPTS
# =========================

SYSTEM_PROMPT_MATH = (
    "You are solving a math problem for automatic grading. "
    "Be concise and accurate. "
    "Do not repeat the problem. "
    "Do not write self-talk such as wait, hmm, maybe, I think, let me check, or let me verify. "
    "Do not write placeholder words such as answer, value, result, choice, letter, blank, ?, or .... "
    "You may write a few short calculation steps if needed, but avoid long explanations. "
    "End with exactly one boxed final result. "
    "For one blank, use a real value like \\boxed{42}. "
    "For multiple blanks, put the real answers in order inside one box, separated by commas, like \\boxed{4,16}. "
    "The boxed answer must contain actual numbers, expressions, or choices, never placeholder text."
)

SYSTEM_PROMPT_MCQ = (
    "You are solving a multiple-choice math problem for automatic grading. "
    "Be concise and accurate. "
    "Do not repeat the problem. "
    "Do not write self-talk such as wait, hmm, maybe, I think, let me check, or let me verify. "
    "Do not write placeholder words such as answer, value, result, choice, letter, blank, ?, or .... "
    "You may write one short calculation line if needed. "
    "End with exactly one boxed capital option letter, like \\boxed{A}. "
    "The boxed answer must contain the actual option letter only."
)


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    """
    Longfix/v3 used only MCQ vs general math prompting.
    No stats/trig/algebra-specific routing.
    """
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(
            f"{lbl}. {str(opt).strip()}" for lbl, opt in zip(labels, options)
        )
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"

    return SYSTEM_PROMPT_MATH, question


# =========================
# MODEL LOADING
# =========================

def load_model(
    model_id: str = MODEL_ID,
    model_path: str = DEFAULT_MODEL_PATH,
) -> tuple[AutoTokenizer, LLM]:
    """
    Download/cache Qwen model and load with vLLM + bitsandbytes.
    """
    local_model_path = snapshot_download(
        repo_id=model_id,
        local_dir=model_path,
        max_workers=2,
    )

    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=local_model_path,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=False,
        gpu_memory_utilization=0.25,
        max_model_len=4096,
        trust_remote_code=True,
        max_num_seqs=1,
        max_num_batched_tokens=2048,
        enforce_eager=True,
    )

    return tokenizer, llm


# =========================
# GENERATION
# =========================

def get_first_pass_max_tokens(item: Dict[str, Any]) -> int:
    """
    First-pass token budget.
    """
    if item.get("options"):
        return 384
    return 1024


def get_rerun_max_tokens(item: Dict[str, Any]) -> int:
    """
    Longer token budget for bad/malformed rows.
    """
    if item.get("options"):
        return 512
    return 1536


def generate_vllm_response(
    item: Dict[str, Any],
    tokenizer: AutoTokenizer,
    llm: LLM,
    max_tokens: int,
) -> str:
    """
    Generate one response for a question.
    """
    system_prompt, user_prompt = build_prompt(
        item["question"],
        item.get("options"),
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    prompt += (
        "\n/no_think\n"
        "RESPONSE RULES:\n"
        "- Be concise.\n"
        "- No self-talk.\n"
        "- No placeholder boxed answers.\n"
        "- End with exactly one real boxed final answer.\n"
        "Now solve and give the final boxed answer:"
    )

    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        presence_penalty=0.0,
        repetition_penalty=1.05,
    )

    outputs = llm.generate([prompt], params)
    return outputs[0].outputs[0].text


# =========================
# FILE WRITING
# =========================

def write_jsonl(rows_by_id: Dict[int, Dict[str, Any]], path: str) -> None:
    rows = sorted(rows_by_id.values(), key=lambda r: int(r["id"]))

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(rows_by_id: Dict[int, Dict[str, Any]], path: str) -> None:
    rows = sorted(rows_by_id.values(), key=lambda r: int(r["id"]))

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "response"],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow({
                "id": row["id"],
                "response": row.get("response", ""),
            })


# =========================
# MAIN ENTRY POINT
# =========================

def run_inference(
    data_path: str = "data/private.jsonl",
    output_csv: str = "submission.csv",
    output_jsonl: str = "inference_results.jsonl",
    model_id: str = MODEL_ID,
    model_path: str = DEFAULT_MODEL_PATH,
) -> None:
    """
    Full end-to-end inference pipeline.

    This matches the longfix/v3-style process:
    1. Load Qwen/Qwen3-4B-Thinking-2507.
    2. Run first-pass inference on every question.
    3. Detect missing, malformed, placeholder, or likely cutoff answers.
    4. Rerun only those bad rows with a longer token budget.
    5. Clean each final response to the last real boxed answer if possible.
    6. Write final submission CSV.
    """
    configure_environment()

    data = [json.loads(line) for line in open(data_path, encoding="utf-8")]
    data_by_id = {int(item["id"]): item for item in data}

    print(f"Loaded {len(data)} questions from {data_path}")

    tokenizer, llm = load_model(
        model_id=model_id,
        model_path=model_path,
    )

    print("Model loaded.")

    results_by_id: Dict[int, Dict[str, Any]] = {}

    # Resume support
    if Path(output_jsonl).exists():
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    results_by_id[int(row["id"])] = row

        print(f"Resuming from {len(results_by_id)} completed rows.")

    # =========================
    # FIRST PASS
    # =========================

    first_start = time.time()
    first_new = 0

    for item in tqdm(data, desc="First pass"):
        item_id = int(item["id"])

        if item_id in results_by_id:
            continue

        try:
            q_start = time.time()
            raw_response = generate_vllm_response(
                item=item,
                tokenizer=tokenizer,
                llm=llm,
                max_tokens=get_first_pass_max_tokens(item),
            )
            elapsed = time.time() - q_start

            cleaned = clean_to_last_real_box(raw_response)
            final_response = cleaned if cleaned is not None else raw_response

            row = {
                "id": item_id,
                "question": item["question"],
                "response": final_response,
                "raw_response": raw_response,
                "elapsed_seconds": elapsed,
                "pass": "first",
                "max_new_tokens": get_first_pass_max_tokens(item),
            }

        except Exception as e:
            row = {
                "id": item_id,
                "question": item.get("question", ""),
                "response": "",
                "error": repr(e),
                "traceback": traceback.format_exc(),
                "pass": "first_error",
            }

        results_by_id[item_id] = row
        first_new += 1

        write_jsonl(results_by_id, output_jsonl)
        write_csv(results_by_id, output_csv)

        if first_new % 10 == 0:
            elapsed_total = time.time() - first_start
            avg = elapsed_total / max(first_new, 1)
            remaining = len(data) - len(results_by_id)
            eta_hours = avg * remaining / 3600

            print(
                f"First pass completed {len(results_by_id)}/{len(data)} | "
                f"Avg {avg:.1f}s/question | "
                f"ETA {eta_hours:.2f} hours"
            )

    # =========================
    # IDENTIFY BAD ROWS
    # =========================

    bad_ids = []

    for item in data:
        item_id = int(item["id"])
        row = results_by_id.get(item_id, {})
        response = row.get("response", "")

        if is_bad_response_text(response):
            bad_ids.append(item_id)

    print(f"Bad / missing / placeholder / cutoff responses to rerun: {len(bad_ids)}")
    print("First 30 bad ids:", bad_ids[:30])

    # =========================
    # LONGFIX RERUN BAD ROWS
    # =========================

    rerun_start = time.time()
    rerun_count = 0

    for item_id in tqdm(bad_ids, desc="Longfix rerun"):
        item = data_by_id[item_id]
        old_response = results_by_id.get(item_id, {}).get("response", "")
        old_bad = is_bad_response_text(old_response)

        try:
            q_start = time.time()
            raw_response = generate_vllm_response(
                item=item,
                tokenizer=tokenizer,
                llm=llm,
                max_tokens=get_rerun_max_tokens(item),
            )
            elapsed = time.time() - q_start

            cleaned = clean_to_last_real_box(raw_response)
            new_response = cleaned if cleaned is not None else raw_response
            new_bad = is_bad_response_text(new_response)

            # Keep old if new is bad but old was okay.
            if new_bad and not old_bad:
                final_response = old_response
                rerun_status = "kept_old_response"
            else:
                final_response = new_response
                rerun_status = "used_new_response"

            row = {
                "id": item_id,
                "question": item["question"],
                "response": final_response,
                "raw_response": raw_response,
                "elapsed_seconds": elapsed,
                "pass": "longfix_rerun",
                "rerun_status": rerun_status,
                "max_new_tokens": get_rerun_max_tokens(item),
            }

        except Exception as e:
            row = {
                "id": item_id,
                "question": item.get("question", ""),
                "response": old_response,
                "error_on_rerun": repr(e),
                "traceback_on_rerun": traceback.format_exc(),
                "pass": "longfix_rerun_error",
                "max_new_tokens": get_rerun_max_tokens(item),
            }

        results_by_id[item_id] = row
        rerun_count += 1

        write_jsonl(results_by_id, output_jsonl)
        write_csv(results_by_id, output_csv)

        if rerun_count % 10 == 0:
            elapsed_total = time.time() - rerun_start
            avg = elapsed_total / max(rerun_count, 1)
            remaining = len(bad_ids) - rerun_count
            eta_hours = avg * remaining / 3600

            print(
                f"Reran {rerun_count}/{len(bad_ids)} bad rows | "
                f"Avg {avg:.1f}s/rerun | "
                f"ETA {eta_hours:.2f} hours"
            )

    # =========================
    # FINAL CLEANING PASS
    # =========================

    for item_id, row in list(results_by_id.items()):
        response = row.get("response", "")
        cleaned = clean_to_last_real_box(response)

        if cleaned is not None:
            row["response"] = cleaned
            row["final_cleaning"] = "cleaned_to_last_real_box"
        else:
            row["final_cleaning"] = "kept_raw_no_clean_box"

        results_by_id[item_id] = row

    write_jsonl(results_by_id, output_jsonl)
    write_csv(results_by_id, output_csv)

    # =========================
    # FINAL DIAGNOSTICS
    # =========================

    num_total = len(results_by_id)
    num_boxed = sum(
        "\\boxed" in str(row.get("response", ""))
        for row in results_by_id.values()
    )
    num_empty = sum(
        str(row.get("response", "")).strip() == ""
        for row in results_by_id.values()
    )

    still_bad_ids = []

    for row in results_by_id.values():
        if is_bad_response_text(row.get("response", "")):
            still_bad_ids.append(row["id"])

    print("=" * 80)
    print("INFERENCE DONE")
    print("Total rows:", num_total)
    print("Boxed responses:", num_boxed, "/", len(data))
    print("Empty responses:", num_empty)
    print("Still bad:", len(still_bad_ids))
    print("First bad ids:", still_bad_ids[:30])
    print("Output CSV:", output_csv)
    print("Debug JSONL:", output_jsonl)


if __name__ == "__main__":
    run_inference(
        data_path="data/private.jsonl",
        output_csv="submission.csv",
        output_jsonl="inference_results.jsonl",
    )
