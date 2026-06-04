import json
import os
import random
import time as _time
import warnings
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"


import sys
sys.path.append("/Long-CLIP-main")
from model import longclip

_CLIP_MODEL_PATH = "/LongCLIP-L/longclip-L.pt"
clip_model, clip_processor = longclip.load(_CLIP_MODEL_PATH, device=device)


def TextImageMatching(text, images, task=None, tau=1.0):
    if task == "videomme":
        system_prompt = (
            "Select the best answer to the following multiple-choice question "
            "based on the video and the subtitles. Respond with only the "
            "letter (A, B, C, or D) of the correct option.\n"
        )
        question = text[len(system_prompt):].split("\n")[0]
    elif task == "videomme_w_subtitle":
        question = text.split("\n")[-6]
    elif "longvideobench" in task:
        question = text.split("\n")[0]
    elif "mlvu" in task:
        question = text.split("\n")[1][10:]
    else:
        raise ValueError("unsupport task.")

    with torch.no_grad(), torch.cuda.amp.autocast():
        text = longclip.tokenize([question]).to(device)
        images = torch.stack([clip_processor(Image.fromarray(image)) for image in images]).to(device)
        image_features = clip_model.encode_image(images)
        text_features = clip_model.encode_text(text)
        logits_per_text = text_features @ image_features.T

    probs = (logits_per_text / tau).softmax(dim=1)[0]
    probs = torch.log(probs) - torch.log(-torch.log(torch.rand(len(images), device=probs.device) + 1e-10) + 1e-10)
    indices = np.argsort(-probs.cpu().detach().numpy())
    return indices


def _list_pool_cards(cards_root: str, vid: str, max_n: int) -> List[str]:
    d = Path(cards_root) / str(vid)
    if not d.is_dir():
        return []
    paths = sorted(p for p in d.iterdir() if p.suffix.lower() == ".jpg")
    paths = [str(p) for p in paths][:max_n]
    return paths


def _load_cards_as_ndarray(card_paths: List[str]) -> np.ndarray:
    if not card_paths:
        return np.zeros((1, 336, 336, 3), dtype=np.uint8)
    first = Image.open(card_paths[0]).convert("RGB")
    cw, ch = first.size
    arrs = []
    for p in card_paths:
        im = Image.open(p).convert("RGB")
        if im.size != (cw, ch):
            im = im.resize((cw, ch), Image.Resampling.LANCZOS)
        arrs.append(np.array(im))
    return np.stack(arrs, axis=0)


def _dump_retrieval_overview(
    visual_tmp: List[Optional[Image.Image]],
    indices: np.ndarray,
    high_n: int,
    mid_n: int,
    low_n: int,
    out_dir: str,
    video_id_hint: str = "",
    question_hint: str = "",
) -> None:
    try:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        high_imgs: List[Tuple[int, Image.Image]] = []
        mid_imgs:  List[Tuple[int, Image.Image]] = []
        low_imgs:  List[Tuple[int, Image.Image]] = []
        for rank, idx in enumerate(indices[:high_n + mid_n + low_n]):
            im = visual_tmp[idx] if idx < len(visual_tmp) else None
            if im is None:
                continue
            if rank < high_n:
                high_imgs.append((idx, im))
            elif rank < high_n + mid_n:
                mid_imgs.append((idx, im))
            else:
                low_imgs.append((idx, im))

        DISPLAY_W_HIGH, DISPLAY_W_MID, DISPLAY_W_LOW = 320, 220, 130
        BAND_PAD, ROW_PAD, BAND_LABEL_H = 16, 6, 28

        def _resize_to_w(im: Image.Image, target_w: int) -> Image.Image:
            w, h = im.size
            if w == 0:
                return im
            target_h = max(1, int(round(h * (target_w / w))))
            return im.resize((target_w, target_h), Image.Resampling.LANCZOS)

        high_disp = [(i, _resize_to_w(im, DISPLAY_W_HIGH)) for i, im in high_imgs]
        mid_disp  = [(i, _resize_to_w(im, DISPLAY_W_MID))  for i, im in mid_imgs]
        low_disp  = [(i, _resize_to_w(im, DISPLAY_W_LOW))  for i, im in low_imgs]

        def _row_dims(disp_list, max_per_row):
            if not disp_list:
                return 0, 0, 0
            tile_h = max(im.size[1] for _, im in disp_list)
            tile_w = max(im.size[0] for _, im in disp_list)
            n = len(disp_list)
            cols = min(n, max_per_row)
            rows = (n + cols - 1) // cols
            band_w = cols * tile_w + (cols + 1) * ROW_PAD
            band_h = rows * tile_h + (rows + 1) * ROW_PAD + BAND_LABEL_H
            return band_w, band_h, tile_h

        high_w, high_h, _ = _row_dims(high_disp, max_per_row=8)
        mid_w,  mid_h,  _ = _row_dims(mid_disp,  max_per_row=8)
        low_w,  low_h,  _ = _row_dims(low_disp,  max_per_row=8)

        canvas_w = max(high_w, mid_w, low_w) + 2 * BAND_PAD
        canvas_h = (high_h + mid_h + low_h) + 4 * BAND_PAD + 60
        canvas_w = max(canvas_w, 600)
        canvas_h = max(canvas_h, 200)
        canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 247))

        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(canvas)
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            tag_font   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        except Exception:
            title_font = label_font = tag_font = ImageFont.load_default()

        title = f"vid={video_id_hint}  picked={len(high_imgs)}H+{len(mid_imgs)}M+{len(low_imgs)}L"
        if question_hint:
            qh = question_hint.strip().replace("\n", " ")
            if len(qh) > 90:
                qh = qh[:90] + "..."
            title = title + "   |   Q: " + qh
        draw.rectangle([0, 0, canvas_w, 50], fill=(28, 32, 40))
        draw.text((10, 14), title, fill=(255, 255, 255), font=title_font)

        cur_y = 60

        def _paint_band(disp_list, tier_name, max_per_row):
            nonlocal cur_y
            if not disp_list:
                return
            tile_h = max(im.size[1] for _, im in disp_list)
            tile_w = max(im.size[0] for _, im in disp_list)
            band_w = max_per_row * tile_w + (max_per_row + 1) * ROW_PAD
            rows_n = (len(disp_list) + max_per_row - 1) // max_per_row
            band_h = rows_n * tile_h + (rows_n + 1) * ROW_PAD + BAND_LABEL_H
            band_x0 = (canvas_w - band_w) // 2
            draw.rectangle([band_x0, cur_y, band_x0 + band_w, cur_y + BAND_LABEL_H], fill=(60, 70, 90))
            draw.text((band_x0 + 8, cur_y + 6), f"{tier_name}  ({len(disp_list)} cards)",
                      fill=(255, 255, 255), font=label_font)
            inner_y = cur_y + BAND_LABEL_H + ROW_PAD
            for i, (orig_idx, im) in enumerate(disp_list):
                col = i %  max_per_row
                row = i // max_per_row
                x = band_x0 + ROW_PAD + col * (tile_w + ROW_PAD)
                y = inner_y + row * (tile_h + ROW_PAD)
                paste_x = x + (tile_w - im.size[0]) // 2
                paste_y = y + (tile_h - im.size[1]) // 2
                canvas.paste(im, (paste_x, paste_y))
                tag = f"#{i + 1}  pool[{orig_idx}]"
                draw.rectangle([x, y, x + 84, y + 16], fill=(0, 0, 0))
                draw.text((x + 3, y + 1), tag, fill=(255, 255, 255), font=tag_font)
            cur_y += band_h + BAND_PAD

        _paint_band(high_disp, "HIGH (W/2 H/2)", max_per_row=8)
        _paint_band(mid_disp,  "MID  (W/4 H/4)", max_per_row=8)
        _paint_band(low_disp,  "LOW  (W/8 H/8)", max_per_row=8)

        ts = _time.strftime("%Y%m%d_%H%M%S")
        safe_vid = "".join(c if c.isalnum() or c in "._-" else "_"
                           for c in (video_id_hint or "novid"))[:64]
        out_path = Path(out_dir) / f"retrieval_{ts}_{safe_vid}.jpg"
        canvas.save(str(out_path), format="JPEG", quality=88)
    except Exception as e:
        eval_logger.warning(f"[debug-dump] failed: {e!r}")


# Model
@register_model("minicpm_v_w_cards_videomme")
class MiniCPM_V_Videomme(lmms):
    """MiniCPM-V + memory-cards retrieval for VideoMME (matches Qwen3-VL retrieval)."""

    def __init__(
        self,
        pretrained: str = "openbmb/MiniCPM-V-4_5",
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = torch.bfloat16,
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = True,
        # ---- card-retrieval hyperparams (match cards_videomme_qwen3.sh) ----
        max_num_frames: int = 128,
        sample_frames: int = 8,
        high_frames: int = 4,
        mid_frames: int = 8,
        low_frames: int = 32,
        cards_root: str = "/memory/videomme/cards",
        skip_videos_without_cards: Union[bool, str] = False,
        skip_marker: str = "SKIP_NO_CARDS",
        debug_dump_dir: str = "",
        debug_dump_every: int = 1,
        # ---- MiniCPM-V chat params -----------------------------------------
        max_slice_nums: int = 2,
        use_image_id: bool = False,
        attn_implementation: str = "sdpa",
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = device

        import transformers.modeling_utils as _mu
        if not hasattr(_mu.PreTrainedModel, "_minicpm_compat_patched"):
            if not hasattr(_mu.PreTrainedModel, "all_tied_weights_keys"):
                def _safe_all_tied_weights_keys(self):
                    legacy = getattr(self, "_tied_weights_keys", None) or []
                    return {k: k for k in legacy}
                _mu.PreTrainedModel.all_tied_weights_keys = property(_safe_all_tied_weights_keys)
            _mu.PreTrainedModel._minicpm_compat_patched = True

        self._model = AutoModel.from_pretrained(
            pretrained,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            torch_dtype=dtype,
        ).to(dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
        self._config = self._model.config
        self.model.eval()
        self.model.tie_weights()

        self.batch_size_per_gpu = int(batch_size)

        # cards
        self.max_num_frames = int(max_num_frames)
        self.sample_frames = int(sample_frames)
        self.high_frames = int(high_frames)
        self.mid_frames = int(mid_frames)
        self.low_frames = int(low_frames)
        self.cards_root = str(cards_root)
        if isinstance(skip_videos_without_cards, str):
            self.skip_videos_without_cards = skip_videos_without_cards.lower() in ("true", "1", "yes")
        else:
            self.skip_videos_without_cards = bool(skip_videos_without_cards)
        self.skip_marker = str(skip_marker)
        self.debug_dump_dir = str(debug_dump_dir or "").strip()
        try:
            self.debug_dump_every = max(1, int(debug_dump_every))
        except Exception:
            self.debug_dump_every = 1
        self._dump_seen = 0
        self._last_resolved_vid: Optional[str] = None

        # MiniCPM-V chat params
        self.max_slice_nums = int(max_slice_nums)
        self.use_image_id = bool(use_image_id)

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided."
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
            if accelerator.distributed_type in (DistributedType.FSDP, DistributedType.DEEPSPEED):
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

        try:
            videos_with_cards = sum(
                1 for p in Path(self.cards_root).iterdir() if p.is_dir()
            ) if Path(self.cards_root).is_dir() else 0
        except Exception:
            videos_with_cards = -1
        eval_logger.info(
            f"[minicpm-v cards] cards_root={self.cards_root} "
            f"#videos_with_cards={videos_with_cards} "
            f"skip_no_cards={self.skip_videos_without_cards} "
            f"H/M/L={self.high_frames}/{self.mid_frames}/{self.low_frames} "
            f"max_pool={self.max_num_frames} "
            f"max_slice_nums={self.max_slice_nums} "
            f"debug_dump_dir={self.debug_dump_dir or '(off)'} "
            f"debug_dump_every={self.debug_dump_every}"
        )

    @property
    def config(self): return self._config
    @property
    def tokenizer(self): return self._tokenizer
    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model
    @property
    def eot_token_id(self): return self.tokenizer.eos_token_id
    @property
    def max_length(self): return self._max_length
    @property
    def batch_size(self): return self.batch_size_per_gpu
    @property
    def device(self): return self._device
    @property
    def rank(self): return self._rank
    @property
    def world_size(self): return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests):
        raise NotImplementedError

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    # ---- ID resolution + card-pool loading (copied from Qwen3-VL cards) ----
    def load_card_pool(self, video_path: str):
        """Load on-disk card pool for `video_path`.

        Tries (in order):
          1. cards_root/{stem}/         (stem = file stem, == YouTube id)
          2. memory/*.json reverse lookup -> cards_root/{video_id}/

        Returns (frames_ndarray[N,H,W,3], frame_idx, frame_time_str, video_time)
        or (None, None, None, None) if no cards exist.
        """
        stem = Path(video_path).stem
        candidate = Path(self.cards_root) / stem
        if not candidate.is_dir():
            mem_root = Path(self.cards_root).parent / "memory"
            for p in mem_root.glob("*.json"):
                try:
                    with open(p, "r") as fh:
                        rec = json.load(fh)
                    if rec.get("video_path") == video_path or \
                       Path(rec.get("video_path", "")).name == Path(video_path).name:
                        candidate = Path(self.cards_root) / str(rec["video_id"])
                        break
                except Exception:
                    continue

        card_paths = _list_pool_cards(str(candidate.parent), candidate.name,
                                      max_n=self.max_num_frames)
        if not card_paths:
            self._last_resolved_vid = None
            return None, None, None, None

        self._last_resolved_vid = candidate.name
        frames = _load_cards_as_ndarray(card_paths)
        n = frames.shape[0]
        frame_idx = list(range(n))
        frame_time = ",".join([f"{i:.2f}s" for i in frame_idx])
        return frames, frame_idx, frame_time, float(n)

    def load_video(self, video_path, max_frames_num, fps=1, force_sample=False):
        if max_frames_num == 0:
            return np.zeros((1, 336, 336, 3))
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        fps = round(vr.get_avg_fps() / fps)
        frame_idx = [i for i in range(0, len(vr), fps)]
        frame_time = [i / fps for i in frame_idx]
        if len(frame_idx) > max_frames_num or force_sample:
            sample_fps = max_frames_num
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i / vr.get_avg_fps() for i in frame_idx]
        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames, frame_idx, frame_time, video_time

    # ---- main loop --------------------------------------------------------
    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)
            gen_kwargs = all_gen_kwargs[0]

            until = [self.tok_decode(self.eot_token_id)]
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]

            assert self.batch_size_per_gpu == 1, "MiniCPM-V cards only supports batch_size 1"
            assert len(visuals) == 1

            context = contexts[0]
            if "<image>" in context:
                context = context.replace("<image>", "")
            visual = visuals[0]

            if not (isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov"))):
                eval_logger.warning(f"unexpected visual type {type(visual)}; skipping")
                res.append("")
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), "")
                pbar.update(1)
                continue

            # ---- Step 1: load card pool ---------------------------------------
            pool, frame_idx, frame_time, video_time = self.load_card_pool(visual)
            if pool is None:
                if self.skip_videos_without_cards:
                    eval_logger.warning(f"[cards] no cards for {Path(visual).stem}, skipping with marker.")
                    res.append(self.skip_marker)
                    self.cache_hook.add_partial("generate_until", (context, gen_kwargs), self.skip_marker)
                    pbar.update(1)
                    continue
                # fallback: raw-video uniform sampling
                pool, frame_idx, frame_time, video_time = self.load_video(visual, self.max_num_frames)

            # ---- Step 2: LongCLIP retrieval + 3-tier resize -------------------
            try:
                indices = TextImageMatching(context, pool, task=task, tau=0.8)

                visual_tmp: List[Optional[Image.Image]] = [None] * len(pool)
                pil_pool = [Image.fromarray(v).convert("RGB") for v in pool]
                width, height = pil_pool[0].size
                for idx in indices[:self.high_frames]:
                    visual_tmp[idx] = pil_pool[idx].resize(
                        (width // 2, height // 2), Image.Resampling.LANCZOS)
                for idx in indices[self.high_frames: self.high_frames + self.mid_frames]:
                    visual_tmp[idx] = pil_pool[idx].resize(
                        (width // 4, height // 4), Image.Resampling.LANCZOS)
                for idx in indices[self.high_frames + self.mid_frames:
                                   self.high_frames + self.mid_frames + self.low_frames]:
                    visual_tmp[idx] = pil_pool[idx].resize(
                        (width // 8, height // 8), Image.Resampling.LANCZOS)

                if self.debug_dump_dir and self._rank == 0:
                    self._dump_seen += 1
                    if (self._dump_seen - 1) % self.debug_dump_every == 0:
                        vid_hint = self._last_resolved_vid or Path(visual).stem
                        _dump_retrieval_overview(
                            visual_tmp=visual_tmp,
                            indices=indices,
                            high_n=self.high_frames,
                            mid_n=self.mid_frames,
                            low_n=self.low_frames,
                            out_dir=self.debug_dump_dir,
                            video_id_hint=str(vid_hint),
                            question_hint=context,
                        )

                frames_for_lm = [v for v in visual_tmp if v is not None]
            except Exception as e:
                eval_logger.info(f"[cards] retrieval failed: {e!r}; falling back to uniform sample")
                if len(pool) >= self.sample_frames:
                    pool = pool[sorted(random.sample(range(len(pool)), self.sample_frames))]
                h_, w_, _ = pool[0].shape
                frames_for_lm = [Image.fromarray(v).convert("RGB").resize(
                    (w_ // 2, h_ // 2), Image.Resampling.LANCZOS) for v in pool]

            # ---- Step 3: MiniCPM-V chat ---------------------------------------
            msgs = [{"role": "user", "content": frames_for_lm + [context]}]

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            try:
                response = self.model.chat(
                    image=None,
                    msgs=msgs,
                    tokenizer=self.tokenizer,
                    sampling=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_image_id=self.use_image_id,
                    max_slice_nums=self.max_slice_nums,
                )
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                response = ""

            for term in until:
                if len(term) > 0 and isinstance(response, str):
                    response = response.split(term)[0]

            res.append(response)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), response)
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests):
        raise NotImplementedError