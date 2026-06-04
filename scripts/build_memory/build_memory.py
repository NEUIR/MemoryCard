#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from transformers import AutoModelForImageTextToText, AutoProcessor


MEMORY_CONFIG: Dict[str, Dict[str, int]] = {
    "short":  {"total_kf_budget": 128, "session_min": 16, "session_max": 32},
    "medium": {"total_kf_budget": 128, "session_min": 16, "session_max": 32},
    "long":   {"total_kf_budget": 128, "session_min": 16, "session_max": 32},
}
ALLOWED_LABELS = ("short", "medium", "long")
DEFAULT_LABEL = "medium"


HEADER_H        = 120     # burnt-in text strip on top of each card
MIN_KF_PER_SESSION = 1
MAX_KF_PER_SESSION = 16
KFS_PER_CARD    = 1      # each card carries a SINGLE keyframe (no grid)

# Heard (ASR snippet) rendering — burnt onto the card's header so the
# downstream answering model has access to the speech without spending
# any text tokens. Constrained to 1 line to leave room for topic.
HEARD_MAX_CHARS = 400
TOPIC_MAX_CHARS = 100

# Visual-token cost reminder
#   high frames (top-4)            -> W/2 x H/2  ≈ 2x2 = 4x fewer tokens
#   mid  frames (top-5..12)        -> W/4 x H/4  ≈ 16x fewer tokens
#   low  frames (top-13..44)       -> W/8 x H/8  ≈ 64x fewer tokens
# total budget is identical to Q-Frame at default high/mid/low = 4/8/32

VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".avi", ".mov"}


def normalize_text(s: Any) -> str:
    s = str(s or "")
    s = s.replace("\u2019", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = re.sub(r"\s+", " ", s).strip()
    return s


def fmt_hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def fmt_hms_short(seconds: float) -> str:
    """Compact mm:ss or h:mm:ss for header captions."""
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(round(seconds - h * 3600 - m * 60))
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def run_cmd(cmd: List[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)


def ffprobe_duration(video_path: str) -> Optional[float]:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        p = run_cmd(cmd, timeout=30)
    except Exception:
        return None
    if p.returncode != 0:
        return None
    try:
        return float((p.stdout or "").strip())
    except Exception:
        return None


def normalize_label(s: Any) -> str:
    s = str(s or "").strip().lower()
    if s in ALLOWED_LABELS:
        return s
    if s in {"sub", "subshort", "sshort"}:
        return "short"
    if s in {"med", "mid"}:
        return "medium"
    if s in {"long_video", "vlong", "verylong"}:
        return "long"
    return DEFAULT_LABEL


# dataset / ASR loading

def _resolve_video_path(video_data_dir: str, file_id: str) -> str:
    if not video_data_dir:
        return ""
    vdir = Path(video_data_dir)
    fid = str(file_id or "").strip()
    if not fid:
        return ""
    cand = vdir / fid
    if cand.suffix:
        return str(cand) if cand.exists() else ""
    for ext in [".mp4", ".mkv", ".webm", ".mov", ".avi"]:
        p = vdir / f"{fid}{ext}"
        if p.exists():
            return str(p)
    return ""


def list_videos(video_dir: str) -> List[Path]:
    p = Path(video_dir)
    if not p.exists():
        return []
    files = [x for x in p.iterdir() if x.is_file() and x.suffix.lower() in VIDEO_EXTS]
    return sorted(files, key=lambda x: x.name)


def load_videos_from_jsonl(
    jsonl_path: str,
    video_data_dir: str,
    video_id_field: str = "video_id",
    video_fileid_field: str = "videoID",
) -> List[Dict[str, Any]]:
    """Read the Video-MME jsonl. Multiple QA pairs share the same video_id;
    we deduplicate by video_id and keep the dataset's `duration` label."""
    items: List[Dict[str, Any]] = []
    seen = set()
    p = Path(jsonl_path)
    if (not p.exists()) or p.stat().st_size == 0:
        return items

    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            vid = str(obj.get(video_id_field, "") or "").strip() or str(obj.get(video_fileid_field, "") or "").strip()
            if not vid or vid in seen:
                continue

            video_path = str(obj.get("video_path") or obj.get("video") or "").strip()
            if not video_path:
                fileid = str(obj.get(video_fileid_field, "") or "").strip()
                video_path = _resolve_video_path(video_data_dir, fileid)
            if not video_path or (not os.path.exists(video_path)):
                continue

            seen.add(vid)
            items.append({
                "vid":            vid,
                "video_name":     Path(video_path).name,
                "video_path":     video_path,
                "duration_label": normalize_label(obj.get("duration")),
            })
    return items


def _read_parquet_rows(parquet_path: str) -> List[Dict[str, Any]]:
    """Read a parquet file -> list of dict rows. Tries pyarrow first
    (no pandas dependency), then pandas, then fastparquet."""
    last_err = None
    try:
        import pyarrow.parquet as pq  # type: ignore
        tbl = pq.read_table(parquet_path)
        return tbl.to_pylist()
    except Exception as e:
        last_err = e
    try:
        import pandas as pd  # type: ignore
        df = pd.read_parquet(parquet_path)
        return df.to_dict(orient="records")
    except Exception as e:
        last_err = e
    try:
        from fastparquet import ParquetFile  # type: ignore
        pf = ParquetFile(parquet_path)
        df = pf.to_pandas()
        return df.to_dict(orient="records")
    except Exception as e:
        last_err = e
    raise RuntimeError(
        "could not read parquet — install one of: "
        "`pip install pyarrow` (recommended) or `pip install fastparquet`. "
        f"last error: {last_err}"
    )


def load_videos_from_parquet(
    parquet_path: str,
    video_data_dir: str,
    video_id_field: str = "video_id",
    video_fileid_field: str = "videoID",
) -> List[Dict[str, Any]]:
    """Read a Video-MME parquet (HF-style schema) and return one record per
    unique video. Compatible with the same field-name conventions as the
    jsonl loader (video_id / videoID / video_path / duration)."""
    items: List[Dict[str, Any]] = []
    seen = set()
    p = Path(parquet_path)
    if (not p.exists()) or p.stat().st_size == 0:
        return items

    rows = _read_parquet_rows(parquet_path)
    if not rows:
        return items

    for row in rows:
        vid = str(row.get(video_id_field, "") or "").strip() or \
              str(row.get(video_fileid_field, "") or "").strip()
        if not vid or vid in seen:
            continue

        video_path = str(row.get("video_path") or row.get("video") or "").strip()
        if not video_path:
            fileid = str(row.get(video_fileid_field, "") or "").strip()
            if fileid:
                video_path = _resolve_video_path(video_data_dir, fileid)
        if not video_path or (not os.path.exists(video_path)):
            continue

        seen.add(vid)
        items.append({
            "vid":            vid,
            "video_name":     Path(video_path).name,
            "video_path":     video_path,
            "duration_label": normalize_label(row.get("duration")),
        })
    return items


def load_video_items(
    video_dir: str,
    data_jsonl: str,
    video_data_dir: str,
    video_id_field: str,
    video_fileid_field: str,
) -> List[Dict[str, Any]]:
    if data_jsonl:
        # auto-detect by extension; accept .parquet OR .jsonl
        ext = Path(data_jsonl).suffix.lower()
        if ext == ".parquet":
            return load_videos_from_parquet(
                parquet_path=data_jsonl,
                video_data_dir=video_data_dir,
                video_id_field=video_id_field,
                video_fileid_field=video_fileid_field,
            )
        return load_videos_from_jsonl(
            jsonl_path=data_jsonl,
            video_data_dir=video_data_dir,
            video_id_field=video_id_field,
            video_fileid_field=video_fileid_field,
        )
    vids = list_videos(video_dir)
    return [{"vid": vp.stem, "video_name": vp.name, "video_path": str(vp),
             "duration_label": DEFAULT_LABEL} for vp in vids]


def load_asr_for_video(asr_root: Path, vid: str) -> Dict[str, Any]:
    p = asr_root / f"{vid}.json"
    if not p.exists():
        return {"full_text": "", "segments": [], "duration_sec": None, "duration_label": ""}
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return {"full_text": "", "segments": [], "duration_sec": None, "duration_label": ""}
    return {
        "full_text":      normalize_text(obj.get("full_text") or ""),
        "segments":       list(obj.get("segments") or []),
        "duration_sec":   obj.get("duration_sec"),
        "duration_label": normalize_label(obj.get("duration_label")),
    }


def asr_text_with_timestamps(segments: List[Dict[str, Any]], max_chars: int = 12000) -> str:
    if not segments:
        return "(no ASR available)"
    lines: List[str] = []
    total = 0
    for seg in segments:
        st = float(seg.get("start", 0.0) or 0.0)
        ed = float(seg.get("end",   st)  or st)
        txt = normalize_text(seg.get("text") or "")
        if not txt:
            continue
        line = f"[{st:.2f}-{ed:.2f}] {txt}"
        if total + len(line) + 1 > max_chars:
            lines.append("... (transcript truncated)")
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines) if lines else "(no ASR available)"


def slice_asr_segments_by_window(
    segments: List[Dict[str, Any]],
    start_sec: float,
    end_sec: float,
) -> Tuple[List[Dict[str, Any]], str]:
    """Slice ASR segments to a session window using STRICT midpoint
    membership: a segment belongs to [start_sec, end_sec) iff its midpoint
    falls inside that window. This guarantees:

      (1) Every ASR segment belongs to exactly ONE session (no double-
          counting, no inter-session leakage).
      (2) The "Heard" text burnt onto a card never includes speech from
          neighboring sessions, so the card's textual anchor is strictly
          time-aligned to its visual content.

    The previous implementation included any segment that *overlapped*
    the window, which meant a segment spanning [3.92, 9.52] would be
    fully included in BOTH a window ending at 5.0 AND a window starting
    at 5.0 — causing the "Heard" text on adjacent cards to bleed into
    each other and look mis-aligned with the visible session timestamp.
    """
    out: List[Dict[str, Any]] = []
    parts: List[str] = []
    for seg in segments:
        st = float(seg.get("start", 0.0) or 0.0)
        ed = float(seg.get("end",   st)  or st)
        # use provided mid if available, else compute one
        mid_raw = seg.get("mid")
        if mid_raw is None:
            mid = 0.5 * (st + ed)
        else:
            try:
                mid = float(mid_raw)
            except Exception:
                mid = 0.5 * (st + ed)
        # midpoint must fall inside [start_sec, end_sec)
        if mid < start_sec or mid >= end_sec:
            continue
        txt = normalize_text(seg.get("text") or "")
        if not txt:
            continue
        out.append({"start": round(st, 4), "end": round(ed, 4), "text": txt})
        parts.append(txt)
    return out, normalize_text(" ".join(parts))


# Qwen3-VL chat — OFFICIAL pipeline only
def build_qwen3_vl(model_dir: str, attn_impl: str = "sdpa"):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"[INFO] loading Qwen3-VL from {model_dir} (dtype={dtype}, attn={attn_impl})", flush=True)
    kwargs: Dict[str, Any] = dict(dtype=dtype, device_map="auto")
    if attn_impl and attn_impl.lower() != "auto":
        kwargs["attn_implementation"] = attn_impl
    model = AutoModelForImageTextToText.from_pretrained(model_dir, **kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_dir)
    return model, processor


def vlm_chat(
    model,
    processor,
    messages: List[Dict[str, Any]],
    max_new_tokens: int = 4096,
    do_sample: bool = False,
    temperature: float = 0.0,
) -> str:
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    gen_kwargs: Dict[str, Any] = dict(max_new_tokens=int(max_new_tokens))
    if do_sample:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = float(temperature)
    else:
        gen_kwargs["do_sample"] = False
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **gen_kwargs)
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    out_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )
    return out_text[0] if out_text else ""


# self-read prompt + parsing
SELFREAD_SYSTEM = (
    "You are a careful video analyst. You will read an entire video together "
    "with its time-aligned speech transcript and segment it into a fixed "
    "number of semantically coherent SESSIONS. Each session will become one "
    "memory card. You DO NOT know the downstream question — focus only on "
    "describing the video itself."
)

SELFREAD_USER_TEMPLATE = """The video is {duration_sec:.1f} seconds long.
Time-aligned speech transcript (may be empty if the video is silent):

{asr_block}

TASK — segment this video into between {min_sessions} and {max_sessions}
semantically coherent sessions. The TOTAL number of keyframes across ALL
sessions must EXACTLY equal {total_kf_budget}. You decide how to allocate
this {total_kf_budget}-keyframe BUDGET across sessions:

  * Sessions with rich VISUAL change (many scene cuts, multiple subjects,
    complex actions, lots of on-screen text/labels, varied composition)
    deserve MORE keyframes — up to {max_kf_per_session} per session.
  * Visually static or repetitive sessions (single static shot, talking
    head, fixed scene, slide) need FEWER keyframes — minimum
    {min_kf_per_session} per session.
  * The sum of "keyframes" lengths over all sessions MUST be EXACTLY
    {total_kf_budget}. Count carefully before answering.

Hard constraints:
- Sessions are CONTIGUOUS, NON-OVERLAPPING, in time order, and together
  cover [0, {duration_sec:.1f}].
- For each session output:
    * "start"   : float seconds, >= 0
    * "end"     : float seconds, <= {duration_sec:.1f}, strictly > start
    * "topic"   : <= 12 words, the session's topic
    * "summary" : ONE short sentence, <= 12 words. Describe what
                  VISUALLY happens in this session — what is on screen,
                  who is doing what, where. Do NOT transcribe what is
                  spoken; the summary is a visual anchor, not a caption
                  of the speech.
    * "keyframes": between {min_kf_per_session} and {max_kf_per_session}
                   float timestamps in seconds, strictly inside
                   (start, end), ordered ascending.
                   CRITICAL: keyframes within a session MUST be VISUALLY
                   DIVERSE and TIME-SPREAD across the session — together
                   they should show the FULL range of what changes during
                   that session. Do NOT cluster them in one moment. Each
                   keyframe should look meaningfully DIFFERENT from the
                   others (different camera angle, different subject,
                   different action stage, different on-screen text, etc.).
                   Prefer moments containing on-screen text, product
                   labels, faces, gestures, or scene transitions — these
                   carry the most decision-relevant information.

Return ONLY a single JSON object with this exact schema. No prose, no
markdown fences, no comments:

{{
  "sessions": [
    {{
      "start": 0.0,
      "end":   12.3,
      "topic": "...",
      "summary": "...",
      "keyframes": [t1, t2, ...]
    }}
  ]
}}
"""


def _strip_json_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_json_object(s: str) -> str:
    s = _strip_json_fences(s)
    start = s.find("{")
    if start < 0:
        return ""
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return ""


def _salvage_truncated_sessions(raw: str) -> List[Dict[str, Any]]:
    """When the model's JSON output is truncated mid-array (e.g. ran out of
    max_new_tokens), the top-level `}` never closes and the standard parser
    returns nothing. This function walks through the raw text, finds every
    well-formed `{ ... }` block that looks like a session, and returns the
    parsed list. Order preserved.
    """
    s = _strip_json_fences(raw or "")
    sessions: List[Dict[str, Any]] = []
    i = 0
    while i < len(s):
        # find next opening '{' that has at least 'start' inside
        j = s.find("{", i)
        if j < 0:
            break
        depth = 0
        end = -1
        in_str = False
        esc = False
        for k in range(j, len(s)):
            ch = s[k]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = k
                    break
        if end < 0:
            i = j + 1
            continue
        block = s[j:end + 1]
        # only keep blocks that look like session dicts (have 'start' and 'end')
        if '"start"' in block and '"end"' in block:
            try:
                obj = json.loads(block)
                if isinstance(obj, dict) and "start" in obj and "end" in obj:
                    sessions.append(obj)
            except Exception:
                pass
        i = end + 1
    return sessions


def parse_selfread_output(raw: str, duration_sec: float) -> List[Dict[str, Any]]:
    blob = _extract_json_object(raw or "")
    sessions: List[Dict[str, Any]] = []
    if blob:
        try:
            obj = json.loads(blob)
            cand = obj.get("sessions") if isinstance(obj, dict) else None
            if isinstance(cand, list):
                sessions = cand
        except Exception:
            sessions = []

    if not sessions:
        salvaged = _salvage_truncated_sessions(raw or "")
        if salvaged:
            sessions = salvaged

    if not sessions:
        return []

    parsed: List[Dict[str, Any]] = []
    for s in sessions:
        if not isinstance(s, dict):
            continue
        try:
            st = float(s.get("start", 0.0))
            ed = float(s.get("end",   0.0))
        except Exception:
            continue
        if not (ed > st):
            continue

        kfs_raw = s.get("keyframes")
        if kfs_raw is None:
            kfs_raw = s.get("display_keyframes") or s.get("retrieval_keyframes") or []
        kfs: List[float] = []
        if isinstance(kfs_raw, list):
            for v in kfs_raw:
                try:
                    kfs.append(float(v))
                except Exception:
                    pass

        parsed.append({
            "start":     max(0.0, st),
            "end":       min(float(duration_sec), ed),
            "topic":     normalize_text(s.get("topic")   or ""),
            "summary":   normalize_text(s.get("summary") or ""),
            "keyframes": kfs,
        })
    return parsed


def sanitize_sessions(
    sessions: List[Dict[str, Any]],
    duration_sec: float,
    session_min: int,
    session_max: int,
    total_kf_budget: int,
    min_kf_per_session: int = MIN_KF_PER_SESSION,
    max_kf_per_session: int = MAX_KF_PER_SESSION,
) -> List[Dict[str, Any]]:
    """Clean sessions and enforce:
       1) session count in [session_min, session_max]
       2) sum(len(keyframes)) == total_kf_budget
       3) each session has between min_kf_per_session and max_kf_per_session
       4) per-session keyframes are time-spread inside (start, end)
    The number of sessions is NOT forced to a single target — Q-Frame's
    Card budget is at the keyframe level, not the session level."""
    if not sessions:
        return []

    # ------- step 1: clean ranges, drop too-short, sort, no-overlap -------
    cleaned = []
    for s in sessions:
        st = max(0.0, float(s.get("start", 0.0)))
        ed = min(float(duration_sec), float(s.get("end", 0.0)))
        if ed - st < 0.5:
            continue
        cleaned.append({**s, "start": st, "end": ed})
    if not cleaned:
        return []
    cleaned.sort(key=lambda x: float(x["start"]))

    # tile [0, duration]
    fixed: List[Dict[str, Any]] = []
    cursor = 0.0
    for s in cleaned:
        st = max(cursor, float(s["start"]))
        ed = max(st + 0.1, float(s["end"]))
        fixed.append({**s, "start": st, "end": ed})
        cursor = ed
    if fixed and fixed[-1]["end"] < duration_sec - 0.5:
        fixed[-1]["end"] = float(duration_sec)

    # ------- step 2: enforce session count in [session_min, session_max] -------
    while len(fixed) < session_min:
        # split the longest session in half
        idx = max(range(len(fixed)), key=lambda i: fixed[i]["end"] - fixed[i]["start"])
        s = fixed[idx]
        mid = (s["start"] + s["end"]) / 2.0
        left = {**s, "end": mid}
        right = {**s, "start": mid, "topic": s.get("topic", ""),
                 "summary": s.get("summary", ""), "keyframes": []}
        fixed = fixed[:idx] + [left, right] + fixed[idx+1:]

    while len(fixed) > session_max:
        # merge the shortest adjacent pair
        pair_idx = min(range(len(fixed)-1),
                       key=lambda i: (fixed[i]["end"] - fixed[i]["start"])
                                   + (fixed[i+1]["end"] - fixed[i+1]["start"]))
        a, b = fixed[pair_idx], fixed[pair_idx+1]
        merged = {
            "start":   a["start"],
            "end":     b["end"],
            "topic":   a.get("topic")   or b.get("topic")   or "",
            "summary": (a.get("summary","") + " " + b.get("summary","")).strip(),
            "keyframes": list(a.get("keyframes") or []) + list(b.get("keyframes") or []),
        }
        fixed = fixed[:pair_idx] + [merged] + fixed[pair_idx+2:]

    # ------- step 3: clamp + clean keyframes inside each session -------
    for s in fixed:
        st, ed = float(s["start"]), float(s["end"])
        kfs = []
        for v in (s.get("keyframes") or []):
            try:
                t = float(v)
            except Exception:
                continue
            if st < t < ed:
                kfs.append(t)
        kfs = sorted(set(round(t, 3) for t in kfs))
        # cap at max_kf_per_session
        if len(kfs) > max_kf_per_session:
            idxs = [round(i * (len(kfs) - 1) / (max_kf_per_session - 1))
                    for i in range(max_kf_per_session)]
            kfs = [kfs[i] for i in idxs]
        s["keyframes"] = kfs

    # ------- step 4: enforce total_kf_budget across sessions -------
    requested_per_session = [len(s["keyframes"]) for s in fixed]

    def _add_uniform(s: Dict[str, Any], n_target: int) -> None:
        """Set s['keyframes'] to n_target time-spread values inside (start,end)."""
        st, ed = float(s["start"]), float(s["end"])
        if n_target <= 0:
            s["keyframes"] = []
            return
        margin = (ed - st) / (n_target + 1)
        s["keyframes"] = [round(st + (i + 1) * margin, 3) for i in range(n_target)]

    # ensure minimum per session
    for s in fixed:
        if len(s["keyframes"]) < min_kf_per_session:
            _add_uniform(s, min_kf_per_session)

    total_now = sum(len(s["keyframes"]) for s in fixed)

    # If we have too many, trim from sessions with the most kfs first
    while total_now > total_kf_budget:
        candidates = [i for i, s in enumerate(fixed)
                      if len(s["keyframes"]) > min_kf_per_session]
        if not candidates:
            break
        idx = max(candidates, key=lambda i: len(fixed[i]["keyframes"]))
        kfs = fixed[idx]["keyframes"]
        if len(kfs) >= 2:
            gaps = []
            for j in range(len(kfs)):
                left_gap  = kfs[j] - kfs[j-1] if j > 0 else float("inf")
                right_gap = kfs[j+1] - kfs[j] if j < len(kfs) - 1 else float("inf")
                gaps.append(min(left_gap, right_gap))
            drop_j = min(range(len(kfs)), key=lambda j: gaps[j])
            kfs.pop(drop_j)
        else:
            kfs.pop()
        total_now -= 1

    # If we have too few, add to sessions the model originally favoured
    while total_now < total_kf_budget:
        candidates = [i for i, s in enumerate(fixed)
                      if len(s["keyframes"]) < max_kf_per_session]
        if not candidates:
            break
        idx = max(candidates, key=lambda i: (
            requested_per_session[i],
            fixed[i]["end"] - fixed[i]["start"],
        ))
        s = fixed[idx]
        st, ed = float(s["start"]), float(s["end"])
        kfs = list(s["keyframes"])
        bounds = [st] + kfs + [ed]
        gaps = [bounds[j+1] - bounds[j] for j in range(len(bounds) - 1)]
        big = max(range(len(gaps)), key=lambda j: gaps[j])
        new_t = round((bounds[big] + bounds[big+1]) / 2.0, 3)
        kfs.append(new_t)
        s["keyframes"] = sorted(set(kfs))
        total_now = sum(len(s["keyframes"]) for s in fixed)

    # ------- step 5: enforce time-spread within each session -------
    for s in fixed:
        st_s, ed_s = float(s["start"]), float(s["end"])
        sess_len = max(1e-3, ed_s - st_s)
        kfs = s["keyframes"]
        if len(kfs) < 2:
            continue
        kf_span = max(kfs) - min(kfs)
        coverage = kf_span / sess_len
        gaps = [kfs[i+1] - kfs[i] for i in range(len(kfs) - 1)]
        if min(gaps) <= 0:
            _add_uniform(s, len(kfs))
            continue
        gap_ratio = max(gaps) / max(min(gaps), 1e-3)
        if coverage < 0.5 or gap_ratio > 4.0:
            _add_uniform(s, len(kfs))

    return fixed


def fallback_uniform_sessions(
    duration_sec: float,
    num_sessions: int,
    total_kf_budget: int = 128,
) -> List[Dict[str, Any]]:
    """Used when the model's self-read output is unparseable. Splits the
    video into `num_sessions` equal sessions and gives each session an
    even share of `total_kf_budget` keyframes (time-spread inside the
    session). Used only as a last resort."""
    num_sessions = max(1, int(num_sessions))
    step = float(duration_sec) / num_sessions

    # divide budget as evenly as possible across sessions
    base = total_kf_budget // num_sessions
    extra = total_kf_budget - base * num_sessions
    per_session = [base + (1 if i < extra else 0) for i in range(num_sessions)]

    out: List[Dict[str, Any]] = []
    for i in range(num_sessions):
        st = i * step
        ed = duration_sec if i == num_sessions - 1 else (i + 1) * step
        n_kf = max(1, per_session[i])
        margin = (ed - st) / (n_kf + 1)
        kfs = [round(st + (k + 1) * margin, 3) for k in range(n_kf)]
        out.append({
            "start":     round(st, 4),
            "end":       round(ed, 4),
            "topic":     "",
            "summary":   "",
            "keyframes": kfs,
        })
    return out


# pick exactly N timestamps in (start, end)
def uniform_timestamps(start: float, end: float, n: int) -> List[float]:
    n = max(1, int(n))
    if n == 1:
        return [round((start + end) / 2.0, 3)]
    step = (end - start) / (n + 1)
    return [round(start + (i + 1) * step, 3) for i in range(n)]


def pick_n_timestamps_in(raw: List[float], start: float, end: float, n: int) -> List[float]:
    cleaned = sorted({round(float(t), 3) for t in raw if start < float(t) < end})

    if len(cleaned) > n:
        m = len(cleaned)
        if n == 1:
            idxs = [m // 2]
        else:
            idxs = sorted({int(round(i * (m - 1) / (n - 1))) for i in range(n)})
            j = 0
            while len(idxs) < n and j < m:
                if j not in idxs:
                    idxs.append(j); idxs.sort()
                j += 1
            idxs = idxs[:n]
        cleaned = [cleaned[i] for i in idxs]
    elif len(cleaned) < n:
        uni = uniform_timestamps(start, end, n=n + 2)[1:-1]
        for t in uni:
            if all(abs(t - c) > 0.05 for c in cleaned):
                cleaned.append(round(t, 3))
                if len(cleaned) >= n:
                    break
        cleaned = sorted(cleaned)[:n]
    return cleaned[:n]


# raw frame export (high-res, full-resolution from original video)
def ffmpeg_dump_frame(video_path: str, ts_sec: float, out_jpg: str,
                     timeout: int = 30, q: int = 2) -> Tuple[bool, str]:
    Path(out_jpg).parent.mkdir(parents=True, exist_ok=True)
    ts_sec = max(0.0, float(ts_sec))
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y",
        "-ss", f"{ts_sec:.3f}",
        "-i",  video_path,
        "-frames:v", "1",
        "-q:v", str(int(q)),
        out_jpg,
    ]
    try:
        p = run_cmd(cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"ffmpeg_timeout_{timeout}s"
    except Exception as e:
        return False, repr(e)
    if p.returncode != 0:
        return False, (p.stderr or "").strip()
    if (not os.path.exists(out_jpg)) or os.path.getsize(out_jpg) == 0:
        return False, "jpg not created or empty"
    return True, ""


# CARD composition (PIL)
_FONT_CACHE: Dict[int, ImageFont.ImageFont] = {}


def _get_font(size: int) -> ImageFont.ImageFont:
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                f = ImageFont.truetype(p, size)
                _FONT_CACHE[size] = f
                return f
            except Exception:
                pass
    f = ImageFont.load_default()
    _FONT_CACHE[size] = f
    return f


def _resize_keep_ratio_pad(img: Image.Image, w: int, h: int) -> Image.Image:
    iw, ih = img.size
    scale = min(w / max(1, iw), h / max(1, ih))
    nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
    img2 = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    canvas.paste(img2, ((w - nw) // 2, (h - nh) // 2))
    return canvas


def _wrap_text_to_width(draw: ImageDraw.ImageDraw, text: str,
                       font: ImageFont.ImageFont, max_width: int) -> List[str]:
    """Greedy word-wrap so each line fits in max_width pixels."""
    words = (text or "").split()
    if not words:
        return [""]
    lines: List[str] = []
    cur = words[0]
    for w in words[1:]:
        candidate = cur + " " + w
        try:
            bbox = draw.textbbox((0, 0), candidate, font=font)
            tw = bbox[2] - bbox[0]
        except Exception:
            tw = font.getsize(candidate)[0] if hasattr(font, "getsize") else len(candidate) * 8
        if tw <= max_width:
            cur = candidate
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def _truncate_lines(lines: List[str], max_lines: int) -> List[str]:
    if len(lines) <= max_lines:
        return lines
    kept = lines[:max_lines]
    # add ellipsis to the last kept line
    kept[-1] = kept[-1].rstrip()
    if not kept[-1].endswith("..."):
        kept[-1] = kept[-1] + " ..."
    return kept


def render_keyframe_card(
    *,
    out_path: str,
    slot_id: str,
    slot_index: int,
    total_slots: int,
    kf_index: int,
    total_kfs_in_slot: int,
    start_sec: float,
    end_sec: float,
    keyframe_jpg: str,
    keyframe_ts: float,
    topic: str,
    heard: str,
) -> Tuple[bool, str]:
    """Render ONE Q-Frame–style memory card to disk.

    Layout: a small dark header strip (~HEADER_H px) burnt on top of the
    keyframe image. The keyframe itself is preserved at its native
    resolution; the header carries the session's slot id, time range,
    topic, and an ASR snippet (Heard) so the downstream answering model
    can use both the visual and the language signals from a single image
    — without spending any text tokens on metadata.

    Sizing rationale: each card is one keyframe with header. Since Step 3
    will downsample retrieved cards by W/2, W/4, W/8 (tier-dependent),
    we do not impose a fixed canvas size. We just preserve the source
    frame at native dump resolution and prepend the header.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # ---- load keyframe at native resolution ----
    try:
        kf = Image.open(keyframe_jpg).convert("RGB")
    except Exception as e:
        return False, f"open_fail:{keyframe_jpg}:{e!r}"
    kw, kh = kf.size

    # ---- build canvas: header_h + keyframe ----
    canvas_w = kw
    canvas_h = HEADER_H + kh
    canvas = Image.new("RGB", (canvas_w, canvas_h), (28, 32, 40))
    draw = ImageDraw.Draw(canvas)

    # ---- header text ----
    pad_x = 12
    cur_y = 6

    meta_font   = _get_font(16)
    topic_font  = _get_font(20)
    heard_font  = _get_font(22)

    # line 1: meta (slot + kf index + time range)
    meta_text = (f"Slot {slot_index + 1}/{total_slots}   "
                 f"KF {kf_index + 1}/{total_kfs_in_slot}   |   "
                 f"{fmt_hms_short(start_sec)} - {fmt_hms_short(end_sec)}   "
                 f"@ {fmt_hms_short(keyframe_ts)}")
    draw.text((pad_x, cur_y), meta_text, fill=(180, 200, 230), font=meta_font)
    cur_y += 22

    # line 2: topic (truncate to TOPIC_MAX_CHARS)
    topic_text = (topic or "").strip()
    if len(topic_text) > TOPIC_MAX_CHARS:
        topic_text = topic_text[:TOPIC_MAX_CHARS].rstrip() + "..."
    if topic_text:
        draw.text((pad_x, cur_y), "Topic: " + topic_text,
                  fill=(255, 255, 255), font=topic_font)
        cur_y += 22

    # # line 3: heard (ASR snippet, truncated; 1 line only)
    heard_text = normalize_text(heard or "")
    if len(heard_text) > HEARD_MAX_CHARS:
        heard_text = heard_text[:HEARD_MAX_CHARS].rstrip() + "..."
    if heard_text:
        wrapped = _wrap_text_to_width(draw, "Heard: " + heard_text,
                                       heard_font, canvas_w - 2 * pad_x)
        line_h = 26   # heard_font is 22 px, leave ~4 px line gap
        for ln in wrapped:
            if cur_y + line_h > HEADER_H - 4:
                # no room for a fresh line — append "..." to the last drawn line
                break
            draw.text((pad_x, cur_y), ln,
                      fill=(255, 255, 255), font=heard_font)
            cur_y += line_h

    # ---- paste keyframe below header ----
    canvas.paste(kf, (0, HEADER_H))

    try:
        canvas.save(out_path, format="JPEG", quality=92)
    except Exception as e:
        return False, repr(e)
    return True, ""


# self-read driver
def build_selfread_messages(
    video_path: str,
    duration_sec: float,
    asr_block: str,
    min_sessions: int,
    max_sessions: int,
    total_kf_budget: int,
    min_kf_per_session: int,
    max_kf_per_session: int,
) -> List[Dict[str, Any]]:
    user_text = SELFREAD_USER_TEMPLATE.format(
        duration_sec       = float(duration_sec),
        asr_block          = asr_block,
        min_sessions       = int(min_sessions),
        max_sessions       = int(max_sessions),
        total_kf_budget    = int(total_kf_budget),
        min_kf_per_session = int(min_kf_per_session),
        max_kf_per_session = int(max_kf_per_session),
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": SELFREAD_SYSTEM}]},
        {"role": "user", "content": [
            {"type": "video", "video": video_path},
            {"type": "text",  "text": user_text},
        ]},
    ]


def run_selfread(
    model,
    processor,
    video_path: str,
    duration_sec: float,
    asr_block: str,
    cfg: Dict[str, int],
    max_new_tokens: int,
) -> Tuple[List[Dict[str, Any]], str]:
    total_budget = int(cfg["total_kf_budget"])
    s_min = int(cfg["session_min"])
    s_max = int(cfg["session_max"])

    messages = build_selfread_messages(
        video_path         = video_path,
        duration_sec       = duration_sec,
        asr_block          = asr_block,
        min_sessions       = s_min,
        max_sessions       = s_max,
        total_kf_budget    = total_budget,
        min_kf_per_session = MIN_KF_PER_SESSION,
        max_kf_per_session = MAX_KF_PER_SESSION,
    )
    raw = vlm_chat(model, processor, messages,
                   max_new_tokens=max_new_tokens, do_sample=False)
    sessions = parse_selfread_output(raw, duration_sec=duration_sec)
    sessions = sanitize_sessions(
        sessions,
        duration_sec       = duration_sec,
        session_min        = s_min,
        session_max        = s_max,
        total_kf_budget    = total_budget,
        min_kf_per_session = MIN_KF_PER_SESSION,
        max_kf_per_session = MAX_KF_PER_SESSION,
    )
    if not sessions:
        # last-resort fallback: uniform sessions, even kf split
        n_default_sessions = max(s_min, min(s_max, 32))
        sessions = fallback_uniform_sessions(
            duration_sec,
            num_sessions    = n_default_sessions,
            total_kf_budget = total_budget,
        )
    return sessions, raw


# build memory for one video
def build_memory_for_video(
    *,
    model,
    processor,
    item: Dict[str, Any],
    asr_obj: Dict[str, Any],
    out_root: Path,
    selfread_max_new_tokens: int,
    frame_timeout_sec: int,
    keep_intermediate_frames: bool,
) -> Dict[str, Any]:
    vid        = str(item["vid"])
    video_path = str(item["video_path"])
    video_name = str(item.get("video_name") or Path(video_path).name)

    # 1) duration_label: prefer dataset jsonl label, then ASR JSON, else medium.
    duration_label = (item.get("duration_label")
                      or asr_obj.get("duration_label")
                      or DEFAULT_LABEL)
    duration_label = normalize_label(duration_label)
    cfg = MEMORY_CONFIG[duration_label]

    # 2) duration_sec.
    duration_sec = asr_obj.get("duration_sec")
    if duration_sec is None:
        duration_sec = ffprobe_duration(video_path)
    if duration_sec is None or duration_sec <= 0:
        raise RuntimeError(f"could not determine duration for {video_path}")

    # 3) self-read.
    asr_block = asr_text_with_timestamps(asr_obj.get("segments") or [],
                                         max_chars=12000)
    t0 = time.perf_counter()
    sessions, selfread_raw = run_selfread(
        model=model,
        processor=processor,
        video_path=video_path,
        duration_sec=float(duration_sec),
        asr_block=asr_block,
        cfg=cfg,
        max_new_tokens=selfread_max_new_tokens,
    )
    selfread_sec = round(time.perf_counter() - t0, 4)

    # 4) per-slot: dump per-session keyframes; for EACH keyframe render
    cards_root  = out_root / "cards"  / vid    # final pool entries: total_kf_budget cards
    frames_root = out_root / "frames" / vid    # raw keyframes (auto-cleaned)
    cards_root.mkdir(parents=True, exist_ok=True)
    frames_root.mkdir(parents=True, exist_ok=True)

    memory_slots: List[Dict[str, Any]] = []
    frame_errors: List[Dict[str, Any]] = []
    card_errors:  List[Dict[str, Any]] = []
    t1 = time.perf_counter()
    total_slots = len(sessions)

    for i, sess in enumerate(sessions):
        slot_id = f"slot_{i:04d}"
        st = float(sess["start"])
        ed = float(sess["end"])

        kf_ts = sorted(float(t) for t in (sess.get("keyframes") or [])
                       if st < float(t) < ed)
        if not kf_ts:
            kf_ts = [round((st + ed) / 2.0, 3)]

        seg_in_slot, full_in_slot = slice_asr_segments_by_window(
            asr_obj.get("segments") or [], st, ed,
        )
        topic   = normalize_text(sess.get("topic")   or "")
        summary = normalize_text(sess.get("summary") or "")

        kf_records: List[Dict[str, Any]] = []
        for k, ts in enumerate(kf_ts):
            raw_fname = f"{slot_id}_kf_{k:02d}_{ts:09.3f}.raw.jpg"
            raw_fpath = str(frames_root / raw_fname)
            ok, err = ffmpeg_dump_frame(video_path, ts, raw_fpath,
                                        timeout=frame_timeout_sec)
            if not ok:
                frame_errors.append({"slot_id": slot_id, "k": k,
                                     "ts": round(float(ts), 3), "error": err})
                continue

            card_fname = f"{slot_id}_kf_{k:02d}.jpg"
            card_path  = str(cards_root / card_fname)
            ok, err = render_keyframe_card(
                out_path           = card_path,
                slot_id            = slot_id,
                slot_index         = i,
                total_slots        = total_slots,
                kf_index           = k,
                total_kfs_in_slot  = len(kf_ts),
                start_sec          = st,
                end_sec            = ed,
                keyframe_jpg       = raw_fpath,
                keyframe_ts        = float(ts),
                topic              = topic,
                heard              = full_in_slot,
            )
            if not ok:
                card_errors.append({"slot_id": slot_id, "k": k, "error": err})
                card_path = ""

            kf_records.append({
                "k":          k,
                "ts":         round(float(ts), 3),
                "card_path":  card_path,
            })

            if not keep_intermediate_frames:
                try:
                    os.remove(raw_fpath)
                except Exception:
                    pass

        memory_slots.append({
            "slot_id":                  slot_id,
            "start":                    round(st, 4),
            "end":                      round(ed, 4),
            "duration_sec":             round(ed - st, 4),
            "topic":                    topic,
            "summary":                  summary,
            "session_asr_raw":          full_in_slot,
            "session_asr_segments_raw": seg_in_slot,
            "keyframes":                kf_records,   # [{k, ts, card_path}]
        })

    frame_dump_sec = round(time.perf_counter() - t1, 4)

    if not keep_intermediate_frames:
        try:
            for p in frames_root.iterdir():
                break
            else:
                frames_root.rmdir()
        except Exception:
            pass

    pool_card_paths: List[str] = []
    for s in memory_slots:
        for kf in s["keyframes"]:
            if kf.get("card_path"):
                pool_card_paths.append(kf["card_path"])

    memory_record = {
        "video_id":       vid,
        "video_name":     video_name,
        "video_path":     video_path,
        "duration_sec":   round(float(duration_sec), 4),
        "duration_label": duration_label,
        "config_used":    {
            "total_kf_budget":     cfg["total_kf_budget"],
            "session_min":         cfg["session_min"],
            "session_max":         cfg["session_max"],
            "min_kf_per_session":  MIN_KF_PER_SESSION,
            "max_kf_per_session":  MAX_KF_PER_SESSION,
            "header_h":            HEADER_H,
        },
        "memory_pool_size":  len(pool_card_paths),
        "pool_card_paths":   pool_card_paths,
        "memory_slots":   memory_slots,
        "frame_errors":   frame_errors,
        "card_errors":    card_errors,
    }

    timing_record = {
        "selfread_sec":    selfread_sec,
        "frame_dump_sec":  frame_dump_sec,
    }
    return memory_record, timing_record


# main
def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root",      type=str, required=True)
    ap.add_argument("--vlm_model_dir", type=str, required=True)
    ap.add_argument("--video_dir",      type=str, default="")
    ap.add_argument("--data_jsonl",     type=str, default="")
    ap.add_argument("--video_data_dir", type=str, default="")
    ap.add_argument("--jsonl_video_id_field",     type=str, default="video_id")
    ap.add_argument("--jsonl_video_fileid_field", type=str, default="videoID")
    ap.add_argument("--worker_id",   type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--selfread_max_new_tokens", type=int, default=4096)
    ap.add_argument("--frame_timeout_sec",       type=int, default=30)
    ap.add_argument("--attn_impl", type=str, default="sdpa",
                    choices=["sdpa", "eager", "flash_attention_2", "auto"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep_intermediate_frames", action="store_true",
                    help="Keep the 4 raw per-slot JPEGs after the card is built.")
    return ap.parse_args()


def main() -> None:
    args = get_args()
    out_root  = Path(args.out_root)
    asr_root  = out_root / "asr"
    mem_root  = out_root / "memory"
    log_root  = out_root / "timing"
    raw_root  = out_root / "selfread_raw"
    mem_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    items = load_video_items(
        video_dir=args.video_dir,
        data_jsonl=args.data_jsonl,
        video_data_dir=args.video_data_dir,
        video_id_field=args.jsonl_video_id_field,
        video_fileid_field=args.jsonl_video_fileid_field,
    )
    items = [x for i, x in enumerate(items) if i % max(1, args.num_workers) == args.worker_id]
    print(f"[INFO] worker={args.worker_id}/{args.num_workers} items={len(items)}", flush=True)

    model, processor = build_qwen3_vl(args.vlm_model_dir, attn_impl=args.attn_impl)

    timing_path = log_root / f"step2_worker{args.worker_id}.jsonl"
    tfw = open(timing_path, "a", encoding="utf-8")
    try:
        for item in tqdm(items, desc=f"Video-MME memory worker {args.worker_id}/{args.num_workers}"):
            vid = str(item["vid"])
            out_json = mem_root / f"{vid}.json"
            if out_json.exists() and (not args.overwrite):
                continue

            video_path = str(item["video_path"])
            print(f"[START] worker={args.worker_id} video_id={vid} "
                  f"label={item.get('duration_label')} path={video_path}", flush=True)
            t_all = time.perf_counter()

            asr_obj = load_asr_for_video(asr_root, vid)

            try:
                row, timing_extra = build_memory_for_video(
                    model=model,
                    processor=processor,
                    item=item,
                    asr_obj=asr_obj,
                    out_root=out_root,
                    selfread_max_new_tokens=int(args.selfread_max_new_tokens),
                    frame_timeout_sec=int(args.frame_timeout_sec),
                    keep_intermediate_frames=bool(args.keep_intermediate_frames),
                )
                ok, err = True, ""
            except Exception as e:
                ok, err = False, repr(e)
                traceback.print_exc()
                row = {
                    "video_id": vid,
                    "video_name": str(item.get("video_name") or Path(video_path).name),
                    "video_path": video_path,
                    "duration_sec": None,
                    "duration_label": item.get("duration_label", DEFAULT_LABEL),
                    "memory_slots": [],
                    "error": err,
                }
                timing_extra = {"selfread_sec": None, "frame_dump_sec": None}

            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(row, f, ensure_ascii=False, indent=2)

            total_sec = round(time.perf_counter() - t_all, 4)
            tfw.write(json.dumps({
                "video_id":         vid,
                "video_path":       video_path,
                "ok":               ok,
                "error":            err,
                "duration_sec":     row.get("duration_sec"),
                "duration_label":   row.get("duration_label"),
                "memory_pool_size": row.get("memory_pool_size", 0),
                "num_frame_errors": len(row.get("frame_errors") or []),
                "num_card_errors":  len(row.get("card_errors")  or []),
                "selfread_sec":     timing_extra.get("selfread_sec"),
                "frame_dump_sec":   timing_extra.get("frame_dump_sec"),
                "step2_total_sec":  total_sec,
            }, ensure_ascii=False) + "\n")
            tfw.flush()

            print(
                f"[DONE]  worker={args.worker_id} video_id={vid} ok={ok} "
                f"label={row.get('duration_label')} cards={row.get('memory_pool_size', 0)} "
                f"total={total_sec:.2f}s",
                flush=True,
            )
    finally:
        tfw.close()


if __name__ == "__main__":
    main()