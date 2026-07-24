"""Open-ended answer parsing/matching for MMMU's non-multiple-choice subset.

Vendored (near-verbatim) from the official MMMU eval script, same source
lmms_eval.tasks.mmmu.utils already vendors it from:
https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py

Kept as a standalone copy so mmmu_offline.py doesn't depend on the
`lmms_eval` package being installed.
"""
import re
from typing import List


def extract_numbers(string: str) -> List[str]:
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"

    numbers_with_commas = re.findall(pattern_commas, string)
    numbers_scientific = re.findall(pattern_scientific, string)
    numbers_simple = re.findall(pattern_simple, string)
    return numbers_with_commas + numbers_scientific + numbers_simple


def check_is_number(string: str) -> bool:
    try:
        float(string.replace(",", ""))
        return True
    except ValueError:
        return False


def normalize_str(string: str):
    string = string.strip()
    if check_is_number(string):
        string = string.replace(",", "")
        return [round(float(string), 2)]
    string = string.lower()
    if len(string) == 1:
        return [" " + string, string + " "]  # avoid trivial substring matches
    return [string]


def parse_open_response(response: str) -> list:
    """Return a list of predicted strings/numbers extracted from free text."""

    def get_key_subresponses(response):
        response = response.strip().strip(".").lower()
        sub_responses = re.split(r"\.\s(?=[A-Z])|\n", response)
        indicators_of_keys = ["could be ", "so ", "is ", "thus ", "therefore ", "final ", "answer ", "result "]
        key_responses = []
        for index, resp in enumerate(sub_responses):
            if index == len(sub_responses) - 1:
                indicators_of_keys = indicators_of_keys + ["="]
            shortest_key_response = None
            for indicator in indicators_of_keys:
                if indicator in resp:
                    candidate = resp.split(indicator)[-1].strip()
                    if not shortest_key_response or len(candidate) < len(shortest_key_response):
                        shortest_key_response = candidate
            if shortest_key_response and shortest_key_response.strip() not in [":", ",", ".", "!", "?", ";", ":", "'"]:
                key_responses.append(shortest_key_response)
        if len(key_responses) == 0:
            return [response]
        return key_responses

    key_responses = get_key_subresponses(response)
    pred_list = key_responses.copy()
    for resp in key_responses:
        pred_list.extend(extract_numbers(resp))

    tmp_pred_list = []
    for p in pred_list:
        tmp_pred_list.extend(normalize_str(p))
    return list(set(tmp_pred_list))


def eval_open(gold_i, pred_list: list) -> bool:
    """Check whether any normalized ground-truth answer matches any parsed prediction."""
    if isinstance(gold_i, list):
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(normalize_str(answer))
    else:
        norm_answers = normalize_str(gold_i)

    for pred in pred_list:
        if isinstance(pred, str):
            for norm_ans in norm_answers:
                if isinstance(norm_ans, str) and norm_ans in pred:
                    return True
        else:
            if pred in norm_answers:
                return True
    return False
