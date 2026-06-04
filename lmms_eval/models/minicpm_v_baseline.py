import warnings
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger


@register_model("minicpm_v_baseline")
class MiniCPM_V_Baseline(lmms):

    def __init__(
        self,
        pretrained: str = "openbmb/MiniCPM-V-4_5",
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = torch.bfloat16,
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = True,
        max_num_frames: int = 8,
        frame_mode: str = "video",          # "video" or "image"
        max_slice_nums: int = 2,            # set 1 if CUDA OOM and frames > 448*448
        use_image_id: bool = False,         # MUST be False for video mode
        attn_implementation: str = "sdpa",  # "sdpa" or "flash_attention_2"
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        assert frame_mode in ("video", "image"), f"frame_mode must be 'video' or 'image', got {frame_mode}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = device

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
        self.max_num_frames = int(max_num_frames)
        self.frame_mode = frame_mode
        self.max_slice_nums = int(max_slice_nums)
        self.use_image_id = bool(use_image_id)

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected DistributedType.DEEPSPEED. Make sure zero stage is 0.")
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

        eval_logger.info(
            f"[MiniCPM-V] frame_mode={self.frame_mode}, max_num_frames={self.max_num_frames}, "
            f"max_slice_nums={self.max_slice_nums}, use_image_id={self.use_image_id}"
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

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for MiniCPM_V_Baseline")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _uniform_sample_frames(self, video_path: str, num_frames: int) -> List[Image.Image]:

        vr = decord.VideoReader(video_path, num_threads=1)
        total_frames = len(vr)
        if total_frames <= 0:
            raise RuntimeError(f"empty video: {video_path}")

        trim_ratios = [1.0, 0.95, 0.9, 0.8, 0.7, 0.5]
        last_err = None
        for ratio in trim_ratios:
            end = max(num_frames, int(total_frames * ratio) - 1)
            end = min(end, total_frames - 1)
            indices = np.linspace(0, end, num_frames, dtype=int)
            try:
                frames_np = vr.get_batch(indices.tolist()).asnumpy()
                if ratio < 1.0:
                    eval_logger.warning(
                        f"[decord-eof] {video_path}: tail truncated to {ratio:.0%} of frames"
                    )
                return [Image.fromarray(f.astype("uint8")).convert("RGB") for f in frames_np]
            except Exception as e:
                last_err = e
                continue

        eval_logger.warning(f"[decord-eof] {video_path}: per-index fallback after {last_err!r}")
        end = max(num_frames, total_frames // 2)
        indices = np.linspace(0, min(end, total_frames - 1), num_frames, dtype=int).tolist()
        frames = []
        for idx in indices:
            try:
                frames.append(vr[idx].asnumpy())
            except Exception:
                if frames:
                    frames.append(frames[-1])  # pad with last good frame
                else:
                    continue
        if not frames:
            raise RuntimeError(f"could not decode any frame from {video_path}: {last_err!r}")
        return [Image.fromarray(f.astype("uint8")).convert("RGB") for f in frames]

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
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            assert self.batch_size_per_gpu == 1, "MiniCPM-V baseline only supports batch_size_per_gpu == 1"
            assert len(visuals) == 1, "MiniCPM-V baseline expects exactly one visual per request"

            context = contexts[0]
            if "<image>" in context:
                context = context.replace("<image>", "")

            visual = visuals[0]

            try:
                if isinstance(visual, str) and visual.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                    frames = self._uniform_sample_frames(visual, self.max_num_frames)
                elif isinstance(visual, Image.Image):
                    frames = [visual.convert("RGB")]
                else:
                    raise ValueError(f"Unsupported visual type: {type(visual)}")
            except Exception as e:
                eval_logger.error(f"[frame-sample] failed for {visual!r}: {e!r}; skipping sample")
                res.append("")
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), "")
                pbar.update(1)
                continue

            msgs = [{"role": "user", "content": frames + [context]}]

            chat_kwargs = {}
            if self.frame_mode == "video":
                chat_kwargs["use_image_id"] = self.use_image_id   # False
                chat_kwargs["max_slice_nums"] = self.max_slice_nums  # 2 (or 1 if OOM)

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
                    **chat_kwargs,
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

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
