#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
import torch
from qwen_asr import Qwen3ASRModel


VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".avi", ".mov"}

_SENT_END_RE = re.compile(r'[。！？!?\.]+$')
_SOFT_PUNC_RE = re.compile(r'[，,；;：:]$')


def normalize_text(s: Any) -> str:
    s = str(s or "")
    s = s.replace("\u2019", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = re.sub(r"\s+", " ", s).strip()
    return s


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


def ffprobe_has_audio(video_path: str) -> bool:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_type",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        p = run_cmd(cmd, timeout=30)
    except Exception:
        return False
    if p.returncode != 0:
        return False
    return "audio" in ((p.stdout or "").strip().lower())


def infer_duration_label(duration_sec: Optional[float], fallback: str = "") -> str:
    fb = str(fallback or "").strip().lower()
    if fb in {"short", "medium", "long"}:
        return fb
    if duration_sec is None:
        return fb
    try:
        d = float(duration_sec)
    except Exception:
        return fb
    if d < 120:
        return "short"
    if d < 240:
        return "medium"
    return "long"


def extract_audio_chunk(video_path: str, wav_path: str, start_sec: float, end_sec: float, timeout_sec: int) -> Tuple[bool, str]:
    Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.01, float(end_sec) - float(start_sec))
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y",
        "-ss", f"{float(start_sec):.3f}",
        "-t", f"{float(duration):.3f}",
        "-i", video_path,
        "-map", "0:a:0?",
        "-vn", "-ac", "1", "-ar", "16000",
        wav_path,
    ]
    try:
        p = run_cmd(cmd, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        return False, f"ffmpeg_timeout_{timeout_sec}s"
    except Exception as e:
        return False, repr(e)
    if p.returncode != 0:
        return False, (p.stderr or "").strip()
    if (not os.path.exists(wav_path)) or os.path.getsize(wav_path) == 0:
        return False, "wav not created or empty"
    return True, ""


# dataset loading
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
                "vid": vid,
                "video_name": Path(video_path).name,
                "video_path": video_path,
                "duration_label": str(obj.get("duration") or "").strip().lower(),
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
        return load_videos_from_jsonl(
            jsonl_path=data_jsonl,
            video_data_dir=video_data_dir,
            video_id_field=video_id_field,
            video_fileid_field=video_fileid_field,
        )
    vids = list_videos(video_dir)
    return [{"vid": vp.stem, "video_name": vp.name, "video_path": str(vp), "duration_label": ""} for vp in vids]


# qwen3-asr helpers
def normalize_language(language: str) -> Optional[str]:
    key = str(language or "").strip().lower()
    if key in {"", "auto", "none", "null"}:
        return None
    mapping = {
        "zh": "Chinese",
        "zh-cn": "Chinese",
        "zh_cn": "Chinese",
        "chinese": "Chinese",
        "en": "English",
        "english": "English",
        "yue": "Cantonese",
        "cantonese": "Cantonese",
        "ja": "Japanese",
        "japanese": "Japanese",
        "fr": "French",
        "french": "French",
        "de": "German",
        "german": "German",
        "es": "Spanish",
        "spanish": "Spanish",
        "ko": "Korean",
        "korean": "Korean",
    }
    return mapping.get(key, str(language).strip())


def build_asr_context_prompt(hotwords: str) -> str:
    hotwords = normalize_text(hotwords)
    return hotwords


def build_qwen3_asr_model(args: argparse.Namespace) -> Qwen3ASRModel:
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return Qwen3ASRModel.from_pretrained(
        args.qwen3_asr_model_dir,
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=int(args.batch_size),
        max_new_tokens=int(args.max_new_tokens),
        forced_aligner=args.forced_aligner_model_dir,
        forced_aligner_kwargs=dict(
            dtype=dtype,
            device_map=device,
        ),
    )


def iter_time_windows(duration_sec: float, chunk_sec: float, overlap_sec: float) -> List[Tuple[float, float]]:
    duration_sec = max(0.0, float(duration_sec))
    chunk_sec = max(1.0, float(chunk_sec))
    overlap_sec = max(0.0, float(overlap_sec))
    if duration_sec <= 0:
        return [(0.0, 0.0)]
    step = max(0.1, chunk_sec - overlap_sec)
    windows: List[Tuple[float, float]] = []
    cur = 0.0
    while cur < duration_sec:
        st = cur
        ed = min(duration_sec, st + chunk_sec)
        if windows and st < windows[-1][1] and abs(ed - windows[-1][1]) < 1e-6:
            break
        windows.append((round(st, 4), round(ed, 4)))
        if ed >= duration_sec:
            break
        cur += step
    return windows


def _timestamp_item_to_dict(ts_obj: Any, offset_sec: float) -> Optional[Dict[str, Any]]:
    try:
        txt = normalize_text(getattr(ts_obj, "text", "") or ts_obj.get("text") or "")
    except Exception:
        txt = normalize_text(getattr(ts_obj, "text", "") or "")
    if not txt:
        return None

    def _get(name: str, fallback: float = 0.0) -> float:
        try:
            val = getattr(ts_obj, name)
        except Exception:
            try:
                val = ts_obj.get(name)
            except Exception:
                val = fallback
        try:
            return float(val)
        except Exception:
            return float(fallback)

    st = _get("start_time", 0.0) + float(offset_sec)
    ed = _get("end_time", st) + float(offset_sec)
    if ed < st:
        ed = st
    return {
        "start": round(st, 4),
        "end": round(ed, 4),
        "mid": round((st + ed) / 2.0, 4),
        "duration_sec": round(max(0.0, ed - st), 4),
        "text": txt,
    }


def _contains_cjk(s: str) -> bool:
    s = str(s or "")
    for ch in s:
        if "\u4e00" <= ch <= "\u9fff":
            return True
    return False


def merge_timestamp_items_by_punc(
    items: List[Dict[str, Any]],
    max_gap_sec: float = 0.5,
    max_seg_sec: float = 6.0,
    max_tokens_per_seg: int = 25,
    split_on_soft_punc: bool = False,
) -> List[Dict[str, Any]]:
    items = [x for x in items if normalize_text(x.get("text") or "")]
    if not items:
        return []

    merged: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    cur_tokens: List[str] = []
    cur_count = 0

    def flush() -> None:
        nonlocal cur, cur_tokens, cur_count
        if cur is None or not cur_tokens:
            cur = None
            cur_tokens = []
            cur_count = 0
            return

        is_cjk = any(_contains_cjk(t) for t in cur_tokens)
        text = "".join(cur_tokens) if is_cjk else " ".join(cur_tokens)
        text = normalize_text(text)

        cur["text"] = text
        cur["mid"] = round((float(cur["start"]) + float(cur["end"])) / 2.0, 4)
        cur["duration_sec"] = round(max(0.0, float(cur["end"]) - float(cur["start"])), 4)
        merged.append(cur)

        cur = None
        cur_tokens = []
        cur_count = 0

    for x in items:
        st = float(x.get("start", 0.0))
        ed = float(x.get("end", st))
        txt = normalize_text(x.get("text") or "")
        if not txt:
            continue

        if cur is None:
            cur = {
                "start": round(st, 4),
                "end": round(ed, 4),
            }
            cur_tokens = [txt]
            cur_count = 1
        else:
            prev_end = float(cur["end"])
            gap = max(0.0, st - prev_end)
            seg_len_if_merge = max(ed, prev_end) - float(cur["start"])

            should_break = (
                gap > max_gap_sec or
                seg_len_if_merge > max_seg_sec or
                cur_count >= max_tokens_per_seg
            )

            if should_break:
                flush()
                cur = {
                    "start": round(st, 4),
                    "end": round(ed, 4),
                }
                cur_tokens = [txt]
                cur_count = 1
            else:
                cur["end"] = round(max(ed, prev_end), 4)
                cur_tokens.append(txt)
                cur_count += 1

        if _SENT_END_RE.search(txt):
            flush()
        elif split_on_soft_punc and _SOFT_PUNC_RE.search(txt):
            flush()

    flush()

    for i, x in enumerate(merged):
        x["seg_id"] = i

    return merged


def transcribe_audio_chunks(
    model: Qwen3ASRModel,
    chunk_audio_paths: List[str],
    chunk_windows: List[Tuple[float, float]],
    language: Optional[str],
    context_prompt: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    chunk_records: List[Dict[str, Any]] = []
    flat_segments: List[Dict[str, Any]] = []
    full_text_parts: List[str] = []
    if not chunk_audio_paths:
        return chunk_records, flat_segments, ""

    lang_arg = None if language is None else [language] * len(chunk_audio_paths)
    ctx_arg = [context_prompt] * len(chunk_audio_paths)
    results = model.transcribe(
        audio=chunk_audio_paths,
        language=lang_arg,
        context=ctx_arg,
        return_time_stamps=True,
    )

    global_seg_id = 0
    for chunk_id, (r, (chunk_st, chunk_ed)) in enumerate(zip(results, chunk_windows)):
        chunk_text = normalize_text(getattr(r, "text", "") or "")
        chunk_lang = normalize_text(getattr(r, "language", "") or "")

        ts_list = getattr(r, "time_stamps", None)
        raw_ts_dicts: List[Dict[str, Any]] = []
        if ts_list is not None:
            for ts in ts_list:
                item = _timestamp_item_to_dict(ts, offset_sec=chunk_st)
                if item is None:
                    continue
                raw_ts_dicts.append(item)

        ts_dicts = merge_timestamp_items_by_punc(
            raw_ts_dicts,
            max_gap_sec=0.5,
            max_seg_sec=6.0,
            max_tokens_per_seg=25,
            split_on_soft_punc=False,
        )

        for item in ts_dicts:
            item["seg_id"] = global_seg_id
            global_seg_id += 1
            flat_segments.append(item)

        if not chunk_text and ts_dicts:
            has_cjk = any(_contains_cjk(x["text"]) for x in ts_dicts)
            chunk_text = "".join([x["text"] for x in ts_dicts]).strip() if has_cjk else " ".join([x["text"] for x in ts_dicts]).strip()
            chunk_text = normalize_text(chunk_text)

        chunk_records.append({
            "chunk_id": chunk_id,
            "start": round(float(chunk_st), 4),
            "end": round(float(chunk_ed), 4),
            "mid": round((float(chunk_st) + float(chunk_ed)) / 2.0, 4),
            "duration_sec": round(max(0.0, float(chunk_ed) - float(chunk_st)), 4),
            "language": chunk_lang,
            "text": chunk_text,
            "time_stamps": ts_dicts,
        })
        if chunk_text:
            full_text_parts.append(chunk_text)

    return chunk_records, flat_segments, normalize_text(" ".join(full_text_parts))


# main
def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--qwen3_asr_model_dir", type=str, required=True)
    ap.add_argument("--forced_aligner_model_dir", type=str, required=True)
    ap.add_argument("--video_dir", type=str, default="")
    ap.add_argument("--data_jsonl", type=str, default="")
    ap.add_argument("--video_data_dir", type=str, default="")
    ap.add_argument("--jsonl_video_id_field", type=str, default="video_id")
    ap.add_argument("--jsonl_video_fileid_field", type=str, default="videoID")
    ap.add_argument("--worker_id", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--language", type=str, default="auto")
    ap.add_argument("--hotwords", type=str, default="")
    ap.add_argument("--chunk_sec", type=float, default=30.0)
    ap.add_argument("--chunk_overlap_sec", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--audio_timeout_sec", type=int, default=120)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep_wav", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = get_args()
    out_root = Path(args.out_root)
    asr_root = out_root / "asr"
    audio_root = out_root / "audio_chunks"
    log_root = out_root / "timing"
    asr_root.mkdir(parents=True, exist_ok=True)
    audio_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    items = load_video_items(
        video_dir=args.video_dir,
        data_jsonl=args.data_jsonl,
        video_data_dir=args.video_data_dir,
        video_id_field=args.jsonl_video_id_field,
        video_fileid_field=args.jsonl_video_fileid_field,
    )
    items = [x for i, x in enumerate(items) if i % max(1, args.num_workers) == args.worker_id]
    print(f"[INFO] worker={args.worker_id}/{args.num_workers} items={len(items)}")

    asr_model = build_qwen3_asr_model(args)
    language = normalize_language(args.language)
    context_prompt = build_asr_context_prompt(args.hotwords)

    timing_path = log_root / f"step1_worker{args.worker_id}.jsonl"
    tfw = open(timing_path, "a", encoding="utf-8")
    try:
        for item in tqdm(items, desc=f"Video-MME Qwen3-ASR worker {args.worker_id}/{args.num_workers}"):
            vid = str(item["vid"])
            video_path = str(item["video_path"])
            video_name = str(item.get("video_name") or Path(video_path).name)
            out_json = asr_root / f"{vid}.json"
            if out_json.exists() and (not args.overwrite):
                continue

            print(f"[START] worker={args.worker_id} video_id={vid} path={video_path}", flush=True)
            t_all = time.perf_counter()
            timing: Dict[str, float] = {}
            audio_ok = False
            audio_error = ""
            full_text = ""
            segments: List[Dict[str, Any]] = []
            chunk_records: List[Dict[str, Any]] = []

            t0 = time.perf_counter()
            duration_sec = ffprobe_duration(video_path)
            has_audio_stream = ffprobe_has_audio(video_path)
            timing["probe_sec"] = round(time.perf_counter() - t0, 6)

            chunk_audio_paths: List[str] = []
            chunk_windows: List[Tuple[float, float]] = []
            success_chunk_windows: List[Tuple[float, float]] = []
            chunk_errors: List[Dict[str, Any]] = []
            extract_chunk_timings: List[Dict[str, Any]] = []

            if duration_sec is None:
                audio_error = "duration_probe_failed"
            elif not has_audio_stream:
                audio_error = "no_audio_stream"
            else:
                t0 = time.perf_counter()
                chunk_windows = iter_time_windows(
                    duration_sec=float(duration_sec),
                    chunk_sec=float(args.chunk_sec),
                    overlap_sec=float(args.chunk_overlap_sec),
                )
                timing["plan_chunks_sec"] = round(time.perf_counter() - t0, 6)

                t0 = time.perf_counter()
                vid_audio_dir = audio_root / vid
                vid_audio_dir.mkdir(parents=True, exist_ok=True)
                for chunk_id, (st, ed) in enumerate(chunk_windows):
                    wav_path = str(vid_audio_dir / f"chunk_{chunk_id:04d}_{st:09.3f}_{ed:09.3f}.wav")
                    t_chunk = time.perf_counter()
                    ok, err = extract_audio_chunk(video_path, wav_path, st, ed, timeout_sec=args.audio_timeout_sec)
                    extract_sec = round(time.perf_counter() - t_chunk, 6)
                    extract_chunk_timings.append({
                        "chunk_id": chunk_id,
                        "start": round(float(st), 4),
                        "end": round(float(ed), 4),
                        "duration_sec": round(max(0.0, float(ed) - float(st)), 4),
                        "extract_sec": extract_sec,
                        "ok": bool(ok),
                        "error": str(err or ""),
                    })
                    if ok:
                        chunk_audio_paths.append(wav_path)
                        success_chunk_windows.append((st, ed))
                    else:
                        chunk_errors.append({
                            "chunk_id": chunk_id,
                            "start": st,
                            "end": ed,
                            "error": err,
                        })
                timing["extract_audio_sec"] = round(time.perf_counter() - t0, 6)

                if not chunk_audio_paths:
                    audio_error = chunk_errors[0]["error"] if chunk_errors else "no_audio_chunks_extracted"
                else:
                    t0 = time.perf_counter()
                    try:
                        chunk_records, segments, full_text = transcribe_audio_chunks(
                            model=asr_model,
                            chunk_audio_paths=chunk_audio_paths,
                            chunk_windows=success_chunk_windows,
                            language=language,
                            context_prompt=context_prompt,
                        )
                        audio_ok = bool(full_text or segments or chunk_records)
                    except Exception as e:
                        audio_error = repr(e)
                    timing["asr_infer_sec"] = round(time.perf_counter() - t0, 6)

            t0 = time.perf_counter()
            if (not args.keep_wav) and chunk_audio_paths:
                for p in chunk_audio_paths:
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            timing["cleanup_wav_sec"] = round(time.perf_counter() - t0, 6)

            row = {
                "video_id": vid,
                "video_name": video_name,
                "video_path": video_path,
                "duration_sec": round(float(duration_sec), 4) if duration_sec is not None else None,
                "duration_label": infer_duration_label(duration_sec, item.get("duration_label") or ""),
                "audio_ok": bool(audio_ok),
                "audio_error": str(audio_error or ""),
                "has_audio_stream": bool(has_audio_stream),
                "language_hint": language,
                "asr_context_prompt": context_prompt,
                "chunk_sec": float(args.chunk_sec),
                "chunk_overlap_sec": float(args.chunk_overlap_sec),
                "full_text": full_text,
                "segments": segments,
                "chunks": chunk_records,
            }
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(row, f, ensure_ascii=False, indent=2)

            timing_obj = {
                "probe_sec": timing.get("probe_sec", 0.0),
                "plan_chunks_sec": timing.get("plan_chunks_sec", 0.0),
                "extract_audio_sec": timing.get("extract_audio_sec", 0.0),
                "asr_infer_sec": timing.get("asr_infer_sec", 0.0),
                "cleanup_wav_sec": timing.get("cleanup_wav_sec", 0.0),
                "step1_total_sec": round(time.perf_counter() - t_all, 6),
            }
            tfw.write(json.dumps({
                "video_id": vid,
                "video_name": video_name,
                "video_path": video_path,
                "audio_ok": row["audio_ok"],
                "audio_error": row["audio_error"],
                "duration_sec": row["duration_sec"],
                "has_audio_stream": row["has_audio_stream"],
                "num_chunk_windows": len(chunk_windows),
                "num_success_audio_chunks": len(chunk_audio_paths),
                "num_failed_audio_chunks": len(chunk_errors),
                "num_chunks": len(chunk_records),
                "num_segments": len(segments),
                "timing": timing_obj,
                "extract_chunk_timings": extract_chunk_timings,
                "chunk_errors": chunk_errors,
            }, ensure_ascii=False) + "\n")
            tfw.flush()
            print(
                f"[DONE] worker={args.worker_id} video_id={vid} audio_ok={row['audio_ok']} chunks={len(chunk_records)} asr_chars={len(full_text)}",
                flush=True,
            )
    finally:
        tfw.close()


if __name__ == "__main__":
    main()