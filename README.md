# MemoryCard: Video Memory Augmentation for Long-Video Question Answering

Source code for our paper:
**MemoryCard: Video Memory Augmentation for Long-Video Question Answering**

Click the links below to view our paper and project resources:

<a href='https://arxiv.org/abs/2606.05917'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://github.com/NEUIR/MemoryCard'><img src='https://img.shields.io/badge/Code-GitHub-green'></a>

If you find this work useful, please cite our paper and give us a shining star 🌟

```bibtex
@misc{yang2026memorycardtopicawaremultimodalclue,
      title={MemoryCard: Topic-Aware Multi-Modal Clue Compression for Long-Video Question Answering}, 
      author={Qing Yang and Pengcheng Huang and Xinze Li and Zhenghao Liu and Yukun Yan and Yu Gu and Ge Yu and Gang Li and Maosong Sun},
      year={2026},
      eprint={2606.05917},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.05917}, 
}
```

## Overview

![](figs/framework.png)

**MemoryCard** is a video-memory-based augmentation framework for long-video question answering.
Long videos usually contain sparse and temporally scattered evidence, while most frames are redundant. Instead of using isolated frames as evidence, MemoryCard organizes a video into semantically coherent **Memory Cards**. Each card contains event-level visual context, representative moments, and aligned speech cues, enabling VLMs to answer long-video questions with compact and high-density multimodal evidence.

The full pipeline contains three stages:

```text
Long Video
   │
   ├── Step 1: Extract ASR
   │       └── OUT_ROOT/asr/{video_id}.json
   │
   ├── Step 2: Build Memory Cards
   │       ├── OUT_ROOT/cards/{video_id}/slot_XXXX_kf_YY.jpg
   │       ├── OUT_ROOT/memory/{video_id}.json
   │       └── OUT_ROOT/selfread_raw/{video_id}.txt
   │
   └── Step 3: Evaluate with lmms-eval
           └── qwen3_vl_w_cards_videomme
```

**Note:** This README uses **Qwen3-VL** as the backbone model and **Video-MME** as the example benchmark. The pipeline can be adapted to other VLMs or long-video QA benchmarks by modifying the corresponding model wrapper, dataset loader, and running scripts.

## Set Up 🛠️

**Use `git clone` to download this project**

```bash
git clone https://github.com/NEUIR/MemoryCard.git
cd MemoryCard
```

**Create the environment**

We provide the conda environment file used in our experiments.

```bash
conda env create -f memorycard.yml
conda activate memorycard
```

You also need to prepare the following external dependencies and checkpoints:

```text
- ffmpeg
- Qwen3-VL
- Qwen3-ASR
- Qwen3 ForcedAligner
- LongCLIP
```

Please make sure the corresponding model paths are correctly set at the top of the running scripts in `scripts/launch/`.


## Using MemoryCard

### (1) Extract ASR with Qwen3-ASR 🎧

First, edit the path section at the top of:

```bash
scripts/launch/run_step1_extract_asr.sh
```

You need to set:

```bash
MEMORYCARD_ROOT=/path/to/MemoryCard
ASR_REPO_DIR=/path/to/Qwen3-ASR-main
QWEN3_ASR_MODEL_DIR=/path/to/Qwen3-ASR-1.7B
FORCED_ALIGNER_MODEL_DIR=/path/to/Qwen3-ForcedAligner-0.6B
DATA_JSONL=/path/to/Video-MME/test-00000-of-00001.jsonl
VIDEO_DATA_DIR=/path/to/Video-MME/data
OUT_ROOT=/path/to/output/videomme_memory
```

Then run:

```bash
conda activate memorycard
bash scripts/launch/run_step1_extract_asr.sh
```

The generated ASR files will be saved to:

```text
OUT_ROOT/asr/{video_id}.json
```

### (2) Build Memory Cards with Qwen3-VL 🧠

First, edit the path section at the top of:

```bash
scripts/launch/run_step2_build_memory.sh
```

You need to set:

```bash
MEMORYCARD_ROOT=/path/to/MemoryCard
QWEN3_VL_REPO=/path/to/Qwen3-VL-main
VLM_MODEL_DIR=/path/to/Qwen3-VL-8B-Instruct
DATA_JSONL=/path/to/Video-MME/videomme/test-00000-of-00001.parquet
VIDEO_DATA_DIR=/path/to/Video-MME/data
OUT_ROOT=/path/to/output/videomme_memory
```

Then run:

```bash
conda activate memorycard
bash scripts/launch/run_step2_build_memory.sh
```

The generated Memory Cards will be saved to:

```text
OUT_ROOT/cards/{video_id}/slot_XXXX_kf_YY.jpg
OUT_ROOT/memory/{video_id}.json
OUT_ROOT/selfread_raw/{video_id}.txt
```

### (3) Evaluate with lmms-eval 🚀

First, edit the path section at the top of:

```bash
scripts/launch/run_step3_eval_cards.sh
```

You need to set:

```bash
MEMORYCARD_ROOT=/path/to/MemoryCard
PRETRAINED=/path/to/Qwen3-VL-8B-Instruct
LONGCLIP_REPO=/path/to/Long-CLIP-main
LONGCLIP_MODEL=/path/to/longclip-L.pt
CARDS_ROOT=/path/to/output/videomme_memory/cards
OUTPUT_PATH=/path/to/output/eval_logs
```

Then run:

```bash
conda activate memorycard
bash scripts/launch/run_step3_eval_cards.sh
```


By default, we use the following Memory Card retrieval budget:

```text
max_num_frames = 128
high_frames    = 4
mid_frames     = 8
low_frames     = 32
sample_frames  = 8
```

You can modify these values directly in `scripts/launch/run_step3_eval_cards.sh`.

## Baselines

We provide two Qwen3-VL baselines.

### Image-frame baseline

Edit paths in:

```bash
scripts/launch/run_baseline_image.sh
```

Then run:

```bash
conda activate memorycard
bash scripts/launch/run_baseline_image.sh
```

### Video-frame baseline

Edit paths in:

```bash
scripts/launch/run_baseline_video.sh
```

Then run:

```bash
conda activate memorycard
bash scripts/launch/run_baseline_video.sh
```



### About LongCLIP

The retrieval module uses LongCLIP to match questions with Memory Cards. Please make sure these two paths are correctly set in the Step 3 script:

```bash
LONGCLIP_REPO=/path/to/Long-CLIP-main
LONGCLIP_MODEL=/path/to/longclip-L.pt
```

### Debug Visualization 🔍

Step 3 can save visualization images for retrieved cards. The selected high-, mid-, and low-resolution cards are saved to:

```text
OUTPUT_PATH/debug_retrieval_cards/{LOG_SUFFIX}/
```

To disable debug dump, set this variable in the Step 3 script:

```bash
DEBUG_DUMP_DIR=""
```

## Contact 📬

If you have questions, suggestions, or bug reports, please contact:

```text
yangqing_neu@outlook.com
```
