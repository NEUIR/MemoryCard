import datetime
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

TASK_TYPES = ["TR", "AR", "VS", "NQA", "ER", "PQA", "SSC", "AO", "AC"]

TASK_TYPE_MAP = {
    "plotQA":          "PQA",
    "findNeedle":      "NQA",
    "needle":          "NQA",
    "ego":             "ER",
    "count":           "AC",
    "order":           "AO",
    "anomaly_reco":    "AR",
    "topic_reasoning": "TR",
    "subPlot":         "SSC",
    "sub_scene":       "SSC",
    "summary":         "VS",
}


MLVU_VIDEO_ROOT = os.environ.get(
    "MLVU_VIDEO_ROOT",
    "/mlvu/video",
)


def mlvu_doc_to_visual(doc):
    video_name = doc["video_name"]
    base = os.path.basename(video_name)
    cands = [
        os.path.join(MLVU_VIDEO_ROOT, base),
        os.path.join(MLVU_VIDEO_ROOT, video_name),
    ]
    for p in cands:
        if os.path.exists(p):
            return [p]
    sys.exit(f"video path for {video_name!r} not found under {MLVU_VIDEO_ROOT}")


def mlvu_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    option_prompt = ""
    question = doc["question"] + "\nOnly give the best option.\n"
    full_prompt = option_prompt + "\n" + question + "\n" + "Best option: ("
    return full_prompt


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""

    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]


def mlvu_process_results(doc, results):

    pred = results[0]
    pred_ans = extract_characters_regex(pred)

    raw_task_type = doc["task_type"]
    task_type = TASK_TYPE_MAP.get(raw_task_type, raw_task_type)

    data_dict = {"question_id": doc["question"], "task_type": task_type, "pred_answer": pred_ans, "answer": doc["answer"]}
    return {f"mlvu_perception_score": data_dict}


def mlvu_aggregate_results(results):
    category2score = {}
    for task_type in TASK_TYPES:
        category2score[task_type] = {"correct": 0, "answered": 0}

    other = {"correct": 0, "answered": 0}

    for result in results:
        task_type = result["task_type"]
        if task_type in category2score:
            category2score[task_type]["answered"] += 1
            category2score[task_type]["correct"] += result["pred_answer"] == result["answer"]
        else:
            other["answered"] += 1
            other["correct"] += result["pred_answer"] == result["answer"]

    for task_cate in TASK_TYPES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        eval_logger.info(f"Evaluation on Task Categories: {task_cate}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    if other["answered"] > 0:
        eval_logger.info(f"Evaluation on Task Categories: OTHER: {100 * other['correct'] / other['answered'] : .1f}%  ({other['correct']}/{other['answered']})")

    total_correct = 0
    total_answered = 0
    for k, v in category2score.items():
        total_correct += v["correct"]
        total_answered += v["answered"]
    total_correct += other["correct"]
    total_answered += other["answered"]
    eval_logger.info(f"Overall Performance: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    return 100 * total_correct / total_answered if total_answered > 0 else 0