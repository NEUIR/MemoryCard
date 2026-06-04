import base64
import json
import os
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import load_video_decord

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


import random
device = "cuda" if torch.cuda.is_available() else "cpu"


import time as _time
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

        DISPLAY_W_HIGH = 320
        DISPLAY_W_MID  = 220
        DISPLAY_W_LOW  = 130
        BAND_PAD       = 16
        ROW_PAD        = 6
        BAND_LABEL_H   = 28

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
        canvas_h = (high_h + mid_h + low_h) + 4 * BAND_PAD + 60   # title strip
        canvas_w = max(canvas_w, 600)
        canvas_h = max(canvas_h, 200)
        canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 247))

        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(canvas)
        try:
            title_font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            label_font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            tag_font   = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        except Exception:
            title_font = label_font = tag_font = ImageFont.load_default()

        # title strip
        title = f"vid={video_id_hint}  picked={len(high_imgs)}H+{len(mid_imgs)}M+{len(low_imgs)}L"
        if question_hint:
            qh = question_hint.strip().replace("\n", " ")
            if len(qh) > 90:
                qh = qh[:90] + "..."
            title = title + "   |   Q: " + qh
        draw.rectangle([0, 0, canvas_w, 50], fill=(28, 32, 40))
        draw.text((10, 14), title, fill=(255, 255, 255), font=title_font)

        cur_y = 60

        def _paint_band(disp_list: List[Tuple[int, Image.Image]],
                        tier_name: str, max_per_row: int) -> None:
            nonlocal cur_y
            if not disp_list:
                return
            tile_h = max(im.size[1] for _, im in disp_list)
            tile_w = max(im.size[0] for _, im in disp_list)
            band_w = max_per_row * tile_w + (max_per_row + 1) * ROW_PAD
            band_h = ((len(disp_list) + max_per_row - 1) // max_per_row) * tile_h \
                     + (((len(disp_list) + max_per_row - 1) // max_per_row) + 1) * ROW_PAD \
                     + BAND_LABEL_H

            # band header
            band_x0 = (canvas_w - band_w) // 2
            draw.rectangle([band_x0, cur_y,
                            band_x0 + band_w, cur_y + BAND_LABEL_H],
                           fill=(60, 70, 90))
            draw.text((band_x0 + 8, cur_y + 6),
                      f"{tier_name}  ({len(disp_list)} cards)",
                      fill=(255, 255, 255), font=label_font)
            inner_y = cur_y + BAND_LABEL_H + ROW_PAD

            for i, (orig_idx, im) in enumerate(disp_list):
                col = i %  max_per_row
                row = i // max_per_row
                x = band_x0 + ROW_PAD + col * (tile_w + ROW_PAD)
                y = inner_y + row * (tile_h + ROW_PAD)
                # paste the card
                paste_x = x + (tile_w - im.size[0]) // 2
                paste_y = y + (tile_h - im.size[1]) // 2
                canvas.paste(im, (paste_x, paste_y))
                # rank tag
                tag = f"#{i + 1}  pool[{orig_idx}]"
                draw.rectangle([x, y, x + 84, y + 16], fill=(0, 0, 0))
                draw.text((x + 3, y + 1), tag,
                          fill=(255, 255, 255), font=tag_font)
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


import sys
sys.path.append("/Long-CLIP-main")
from model import longclip

model_path = "/LongCLIP-L/longclip-L.pt"
clip_model, clip_processor = longclip.load(model_path, device=device)


def TextImageMatching(text, images, task=None, tau=1.0):
    if task == "videomme":
        system_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.\n"
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
    """List the rendered Q-Frame–style cards for one video, in disk order
    (slot_XXXX_kf_YY.jpg). We keep at most `max_n` cards (= max_num_frames
    in Q-Frame's lingo), to mirror the candidate pool size.
    """
    d = Path(cards_root) / str(vid)
    if not d.is_dir():
        return []
    paths = sorted(p for p in d.iterdir() if p.suffix.lower() == ".jpg")
    paths = [str(p) for p in paths][:max_n]
    return paths


def _load_cards_as_ndarray(card_paths: List[str]) -> np.ndarray:

    if not card_paths:
        return np.zeros((1, 336, 336, 3), dtype=np.uint8)

    # use the first card's size as the canonical size
    first = Image.open(card_paths[0]).convert("RGB")
    cw, ch = first.size

    arrs = []
    for p in card_paths:
        im = Image.open(p).convert("RGB")
        if im.size != (cw, ch):
            im = im.resize((cw, ch), Image.Resampling.LANCZOS)
        arrs.append(np.array(im))
    return np.stack(arrs, axis=0)


@register_model("qwen2_vl_w_cards_videomme")
class Qwen2_VL_Videomme(lmms):

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        use_flash_attention_2: Optional[bool] = False,
        max_pixels: int = 12845056,
        min_pixels: int = 3136,
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
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        if use_flash_attention_2:
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                pretrained,
                torch_dtype="auto",
                device_map=self.device_map,
                attn_implementation="flash_attention_2",
            ).eval()
        else:
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                pretrained, torch_dtype="auto", device_map=self.device_map,
            ).eval()
        self.processor = AutoProcessor.from_pretrained(
            pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        self.sample_frames = sample_frames
        self.high_frames = high_frames
        self.mid_frames = mid_frames
        self.low_frames = low_frames
        self.cards_root = str(cards_root)
        if isinstance(skip_videos_without_cards, str):
            self.skip_videos_without_cards = skip_videos_without_cards.lower() in ("true", "1", "yes")
        else:
            self.skip_videos_without_cards = bool(skip_videos_without_cards)
        self.skip_marker = str(skip_marker)
        # debug dump: only main process writes to avoid race
        self.debug_dump_dir = str(debug_dump_dir or "").strip()
        try:
            self.debug_dump_every = max(1, int(debug_dump_every))
        except Exception:
            self.debug_dump_every = 1
        self._dump_seen = 0
        self._last_resolved_vid: Optional[str] = None

        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._word_size = 1

        try:
            videos_with_cards = sum(
                1 for p in Path(self.cards_root).iterdir() if p.is_dir()
            ) if Path(self.cards_root).is_dir() else 0
        except Exception:
            videos_with_cards = -1
        eval_logger.info(
            f"[qframe-cards] cards_root={self.cards_root} "
            f"#videos_with_cards={videos_with_cards} "
            f"skip_no_cards={self.skip_videos_without_cards} "
            f"high/mid/low={self.high_frames}/{self.mid_frames}/{self.low_frames} "
            f"max_pool={self.max_num_frames} "
            f"debug_dump_dir={self.debug_dump_dir or '(off)'} "
            f"debug_dump_every={self.debug_dump_every}"
        )

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _resolve_video_id_from_path(self, video_path: str) -> str:

        return Path(video_path).stem

    def load_card_pool(self, video_path: str):

        stem = Path(video_path).stem
        candidate = Path(self.cards_root) / stem
        if not candidate.is_dir():
            # try memory/{vid}.json reverse lookup
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

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            gen_kwargs = all_gen_kwargs[0]
            until = [self.tokenizer.decode(self.eot_token_id)]
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            messages = []

            kept_contexts: List[str] = []
            slot_outcome: List[Tuple[str, object]] = []
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")
                message = [{"role": "system", "content": "You are a helpful assistant."}]

                if len(visuals) > 0:
                    visual = visuals[i] if i < len(visuals) else None
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):

                        visual, frame_idx, frame_time, video_time = self.load_card_pool(visual)

                        if visual is None:
                            # video has no cards on disk
                            if self.skip_videos_without_cards:
                                slot_outcome.append(("skip", self.skip_marker))
                                continue

                            visual, frame_idx, frame_time, video_time = self.load_video(
                                visuals[i], self.max_num_frames)

                        try:
                            indices = TextImageMatching(context, visual, task=task, tau=0.8)

                            visual_tmp = [None] * len(visual)
                            visual = [Image.fromarray(v).convert("RGB") for v in visual]
                            width, height = visual[0].size
                            for idx in indices[:self.high_frames]:
                                visual_tmp[idx] = visual[idx].resize(
                                    (width // 2, height // 2), Image.Resampling.LANCZOS)
                            for idx in indices[self.high_frames: self.high_frames + self.mid_frames]:
                                visual_tmp[idx] = visual[idx].resize(
                                    (width // 4, height // 4), Image.Resampling.LANCZOS)
                            for idx in indices[self.high_frames + self.mid_frames:
                                               self.high_frames + self.mid_frames + self.low_frames]:
                                visual_tmp[idx] = visual[idx].resize(
                                    (width // 8, height // 8), Image.Resampling.LANCZOS)

                            # === read-only debug dump (no model input change) ===
                            if self.debug_dump_dir and self._rank == 0:
                                self._dump_seen += 1
                                if (self._dump_seen - 1) % self.debug_dump_every == 0:
                                    vid_hint = self._last_resolved_vid or Path(visuals[i]).stem
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

                            visual = [v for v in visual_tmp if v is not None]
                        except Exception as e:
                            eval_logger.info(f"{e}")
                            if len(visual) >= self.sample_frames:
                                visual = visual[sorted(random.sample(range(len(visual)), self.sample_frames))]
                            height, width, _ = visual[0].shape
                            visual = [Image.fromarray(v).convert("RGB").resize(
                                (width // 2, height // 2), Image.Resampling.LANCZOS) for v in visual]

                        image_content = []
                        for base64_image in visual:
                            buffer = BytesIO()
                            base64_image.save(buffer, format="JPEG")
                            base64_bytes = base64.b64encode(buffer.getvalue())
                            base64_string = base64_bytes.decode("utf-8")
                            image_content.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"})
                        message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})

                    elif isinstance(visual, Image.Image):
                        base64_image = visual.convert("RGB")
                        buffer = BytesIO()
                        base64_image.save(buffer, format="JPEG")
                        base64_bytes = base64.b64encode(buffer.getvalue())
                        base64_string = base64_bytes.decode("utf-8")
                        message.append({"role": "user", "content": [{"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"}, {"type": "text", "text": context}]})
                    elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):
                        image_content = []
                        for v in visual:
                            base64_image = v.convert("RGB")
                            buffer = BytesIO()
                            base64_image.save(buffer, format="JPEG")
                            base64_bytes = base64.b64encode(buffer.getvalue())
                            base64_string = base64_bytes.decode("utf-8")
                            image_content.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"})
                        message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})
                    else:
                        message.append({"role": "user", "content": [{"type": "text", "text": context}]})
                else:
                    message.append({"role": "user", "content": [{"type": "text", "text": context}]})

                messages.append(message)
                kept_contexts.append(context)
                slot_outcome.append(("real", len(messages) - 1))

            if not messages:
                for kind, payload in slot_outcome:
                    if kind == "skip":
                        res.append(payload)
                        self.cache_hook.add_partial(
                            "generate_until", (None, gen_kwargs), payload)
                        pbar.update(1)
                continue

            texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
            image_inputs, video_inputs = process_vision_info(messages)
            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]
                indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                if total_frames - 1 not in indices:
                    indices = np.append(indices, total_frames - 1)
                video_inputs[0] = video_inputs[0][indices]
            inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 128
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            pad_token_id = self.tokenizer.pad_token_id

            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )

            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for slot_i, (kind, payload) in enumerate(slot_outcome):
                if kind == "skip":
                    res.append(payload)
                    self.cache_hook.add_partial(
                        "generate_until",
                        (contexts[slot_i] if slot_i < len(contexts) else None,
                         gen_kwargs),
                        payload,
                    )
                else:
                    msg_idx = payload                        # int
                    ans = answers[msg_idx]
                    ctx = kept_contexts[msg_idx]
                    res.append(ans)
                    self.cache_hook.add_partial(
                        "generate_until", (ctx, gen_kwargs), ans)
                pbar.update(1)
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")