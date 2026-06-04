import base64
from io import BytesIO
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("qwen3_vl")
class Qwen3_VL(lmms):

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        use_flash_attention_2: Optional[bool] = False,
        max_num_frames: int = 8,
        frame_mode: str = "video",            # "video" or "image"
        video_max_pixels: int = 128 * 28 * 28,  
        image_max_pixels: int = 512 * 28 * 28,    
        video_min_pixels: int = 16 * 28 * 28,
        image_min_pixels: int = 16 * 28 * 28,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        assert frame_mode in ("video", "image"), f"frame_mode must be 'video' or 'image', got {frame_mode}"

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
            self._model = AutoModelForImageTextToText.from_pretrained(
                pretrained,
                dtype=torch.bfloat16,
                device_map=self.device_map,
                attn_implementation="flash_attention_2",
            ).eval()
        else:
            self._model = AutoModelForImageTextToText.from_pretrained(
                pretrained,
                dtype=torch.bfloat16,
                device_map=self.device_map,
            ).eval()

        self.processor = AutoProcessor.from_pretrained(pretrained)
        self.processor.tokenizer.padding_side = "left"

        self.max_num_frames = int(max_num_frames)
        self.frame_mode = frame_mode
        self.video_max_pixels = int(video_max_pixels)
        self.image_max_pixels = int(image_max_pixels)
        self.video_min_pixels = int(video_min_pixels)
        self.image_min_pixels = int(image_min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)

        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self._model)
            else:
                self._model = accelerator.prepare_model(self._model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

        eval_logger.info(
            f"[Qwen3-VL] frame_mode={self.frame_mode}, max_num_frames={self.max_num_frames}, "
            f"video_max_pixels={self.video_max_pixels}, image_max_pixels={self.image_max_pixels}"
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

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen3_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _uniform_sample_frames(self, video_path: str, num_frames: int) -> List[Image.Image]:
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)

        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames_np = vr.get_batch(indices.tolist()).asnumpy()
        return [Image.fromarray(f).convert("RGB") for f in frames_np]

    def _build_message(self, video_path: str, context: str):
        frames = self._uniform_sample_frames(video_path, self.max_num_frames)

        if self.frame_mode == "video":
            user_content = [
                {
                    "type": "video",
                    "video": frames,
                    "max_pixels": self.video_max_pixels,   # per-frame budget
                    "min_pixels": self.video_min_pixels,
                },
                {"type": "text", "text": context},
            ]
        else:  # image mode
            user_content = []
            for f in frames:
                buffer = BytesIO()
                f.save(buffer, format="JPEG")
                b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                user_content.append({
                    "type": "image",
                    "image": f"data:image/jpeg;base64,{b64}",
                    "max_pixels": self.image_max_pixels,   # per-image budget
                    "min_pixels": self.image_min_pixels,
                })
            user_content.append({"type": "text", "text": context})

        return [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": user_content},
        ]

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
            for i, context in enumerate(contexts):
                visual = visuals[i] if i < len(visuals) else None

                if isinstance(visual, str) and visual.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    msg = self._build_message(visual, context)
                else:
                    # text-only fallback (videomme/mlvu shouldn't hit this)
                    msg = [
                        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                        {"role": "user", "content": [{"type": "text", "text": context}]},
                    ]
                messages.append(msg)

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )

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
            answers = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError