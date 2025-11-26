# DCE-YOLOv8m-SAR: Lightweight Multi-Dataset Person Detection for Search and Rescue

This repository contains a **lightweight, YOLOv8-based detector with DCE modules** (Divided Context Extraction) trained for **single-class person detection** across **four SAR-relevant datasets**:

* SARD
* HERIDAL
* VisDrone (person class)
* NTUT4K (hard negatives / background)

## Live-Demo

https://github.com/user-attachments/assets/15847854-d3c2-4a57-9351-057317f846e8

The training pipeline is **two-stage**:

1. **Stage 1** – composite training on all four datasets.
2. **Stage 2** – hard-negative fine-tuning with NTUT4K oversampling to strongly reduce false positives.

Evaluation is done with the official **Ultralytics YOLOv8** validation API plus a **false-positive–only protocol** on NTUT4K.

The goal is a **scientifically transparent, reproducible baseline** for SAR person detection that achieves:

* competitive accuracy on SARD and VisDrone,
* acceptable performance on HERIDAL (without being specifically optimised for it),
* and **≈3× fewer false positives** on NTUT4K after Stage 2,

while remaining **fast and parameter-efficient**.

---

## 1. Architecture

The model is based on **YOLOv8m** with architectural modifications inspired by:

> J. An, D. H. Lee, M. D. Putro, B. W. Kim,
> *DCE-YOLOv8: Lightweight and Accurate Object Detection for Drone Vision*,
> IEEE Access, 2024.

Key components:

* **DCE Blocks** (Divided Context Extraction) in the backbone:

  * Split feature channels into a **processing branch** and an **identity branch**.
  * Process only a subset of channels with 3×3 convolutions.
  * Fuse processed and identity features via 1×1 convolution.
  * Reduces computation while preserving contextual information, especially for small or distant persons.

* **ERB Blocks** (Efficient Residual Bottlenecks):

  * Channel reduction via 1×1 conv → sequence of bottlenecks → 1×1 expansion.
  * Replace heavier C2f/C3 blocks to reduce parameters and memory while retaining representational power.

* **SCDown Modules**:

  * 1×1 conv followed by depthwise 3×3 conv with stride 2.
  * Efficient spatial downsampling tailored to lightweight backbones.

* **Detection head**:

  * Single class: `person`.
  * Three detection scales (P3–P5) suitable for drone/SAR imagery.

All custom modules are integrated into the Ultralytics model parser via a custom `tasks.py` and `dce_modules.py`, and instantiated from a custom model YAML (e.g. `cfg/yolov8m_dce.yaml`).

---

## 2. Datasets

All datasets are converted to YOLO format with a **single class** (`person` / human).

### 2.1 SARD (Search and Rescue Dataset)

* Original SAR dataset for person detection in aerial imagery.
* Key reference:
  N. A. Bachir, *Human Detection from Aerial Imagery for Search and Rescue Operations*, M.Sc. thesis, UAEU, 2022.
* Size in this project:

  * Train: **4 041 images / 4 041 labels**
  * Val: **1 144 images / 1 144 labels**

### 2.2 HERIDAL

* Drone-based SAR dataset with challenging backgrounds, small targets and clutter.
* Used in benchmarking works on person detection for SAR missions.
* Size in this project:

  * Train: **1 124 images / 1 124 labels**
  * Val: **313 images / 313 labels**

### 2.3 VisDrone (person-only)

* VisDrone2019 detection dataset, restricted here to the **`person` / `pedestrian`** class.
* Size in this project:

  * Train: **6 471 images / 6 471 labels**
  * Val: **548 images / 548 labels**

### 2.4 NTUT4K (Hard Negatives)

* NTUT4K aerial imagery, used primarily as **background / hard negative**.
* In this project:

  * NTUT4K **train** is included in Stage 1 and Stage 2 training as background/hard negative imagery.
  * NTUT4K **val** is used in evaluation as a **false-positive–only benchmark**: any prediction counts as a false positive.
* Size in this project:

  * Train: **360 images / 219 labels**
  * Val: **154 images / 40 labels (used as background FP-check)**

---

## 3. Repository Structure

A suggested layout (matching the implementation):

```text
.
├── cfg/
│   ├── yolov8m_dce.yaml           # DCE-YOLOv8m architecture (single class)
│   ├── sar_composite.yaml         # Stage 1 composite dataset definition
│   ├── sar_hardneg_stage2.yaml    # Stage 2 hard-negative dataset definition
│   └── old/
│       ├── sardyolo.yaml          # Per-dataset eval YAML (SARD)
│       ├── heridalyolo.yaml       # Per-dataset eval YAML (HERIDAL)
│       ├── visdrone_person.yaml   # Per-dataset eval YAML (VisDrone, person-only)
│       └── ntut4kyolo.yaml        # Per-dataset eval YAML (NTUT4K)
├── src/
│   ├── dce_modules.py             # DCE, ERB, SCDown modules 
│   ├── tasks.py                   # Custom Ultralytics tasks with DCE integration
│   ├── train_stage1_sar_dce.py    # Stage 1 training script
│   ├── train_stage2_hardneg.py    # Stage 2 hard-negative fine-tuning
│   └── benchmark_models.py        # Unified evaluation on all four datasets
└── ...
```

Path names can be adapted as needed; the important parts are the model YAML, dataset YAMLs, custom modules, and the three pipeline scripts.

---

## 4. Environment and Dependencies

Experiments were run with the following key dependencies:

* Python 3.11.4
* **PyTorch 2.1.1 + CUDA 12.1**
* **Ultralytics 8.3.231**
* `opencv-python==4.8.0.76`
* `numpy==1.26.3`
* `PyYAML==5.4.1`

A (truncated) example of the environment:

```text
torch==2.1.1+cu121
torchvision==0.16.1+cu121
ultralytics==8.3.231
opencv-python==4.8.0.76
numpy==1.26.3
PyYAML==5.4.1
...
```

The training scripts set:

* `seed = 0`
* `deterministic = True`

for improved reproducibility (within the usual limits of GPU nondeterminism).

---

## 5. Training Pipeline

### 5.1 Stage 1 – Composite Multi-Dataset Training

* **Model**: `cfg/yolov8m_dce.yaml`

* **Data**: `cfg/sar_composite.yaml` with:

  * Train mixture (with repetition factors per dataset):

    * VisDrone train ×2
    * HERIDAL train ×2
    * SARD train ×1
    * NTUT4K train ×1
  * Val mixture:

    * SARD val
    * HERIDAL val
    * VisDrone val
    * (NTUT4K **not** used in Val for Stage 1)

* **Optimizer**: AdamW

* **Key hyperparameters** (Stage 1):

  * `epochs = 100`
  * `batch = 6`
  * `imgsz = 1280` (training)
  * `lr0 = 0.001`, `lrf = 0.01`
  * `warmup_epochs = 5`
  * `patience = 30`
  * `weight_decay = 5e-4`

* **Initialization**: COCO-pretrained `yolov8m.pt` (Ultralytics).

Example command:

```bash
python src/train_stage1_sar_dce.py \
  --model-yaml cfg/yolov8m_dce.yaml \
  --data cfg/sar_composite.yaml \
  --epochs 100 \
  --batch 6 \
  --imgsz 1280 \
  --project experiments/stage1_sar \
  --name dce_yolov8m_composite_run \
  --device 0
```

The best checkpoint from Stage 1 is stored as, e.g.:

```text
experiments/stage1_sar/dce_yolov8m_composite_run/weights/best.pt
```

### 5.2 Stage 2 – Hard-Negative Fine-Tuning (NTUT-Focused)

Stage 2 starts from the **best Stage 1 checkpoint** and re-weights the training data to emphasise **hard negatives** (NTUT4K) and background suppression.

* **Model**: `cfg/yolov8m_dce.yaml`

* **Pretrained weights**: Stage 1 best checkpoint

* **Data**: `cfg/sar_hardneg_stage2.yaml`

  * Train mixture:

    * NTUT4K train ×4 (strong oversampling)
    * SARD train ×1
    * VisDrone train ×2
    * HERIDAL train ×1

  * Val mixture:

    * SARD val
    * HERIDAL val
    * VisDrone val
    * (NTUT4K still not used in Val; NTUT evaluation is FP-only later)

* **Hyperparameters** (Stage 2):

  * `epochs = 15` (early stopped around epoch 11)

  * `batch = 6`

  * `imgsz = 1280`

  * `optimizer = AdamW`

  * `lr0 = 0.0004`, `lrf = 0.01`

  * `warmup_epochs = 1`

  * `patience = 10`

  * **Augmentation**:

    * `auto_augment = randaugment`
    * `mosaic = 0.5`
    * `mixup = 0.0`
    * `copy_paste = 0.0`
    * `hsv_h = 0.015`, `hsv_s = 0.8`, `hsv_v = 0.5`
    * `degrees = 10`, `translate = 0.1`, `scale = 0.6`
    * `fliplr = 0.5`, `flipud = 0.3`, `erasing = 0.3`

Example command:

```bash
python src/train_stage2_hardneg.py \
  --model-yaml cfg/yolov8m_dce.yaml \
  --pretrained experiments/stage1_sar/dce_yolov8m_composite_run/weights/best.pt \
  --data cfg/sar_hardneg_stage2.yaml \
  --epochs 15 \
  --batch 6 \
  --imgsz 1280 \
  --project experiments/stage2_hardneg \
  --name dce_yolov8m_hardneg_run \
  --device 0
```

The final model used for benchmarking is referred to as, for example:

```text
checkpoints/stage_2_dce_sar_best_of_all_round_2.pt
```

---

## 6. Evaluation Protocol

### 6.1 Unified Benchmark Script

Evaluation across all four datasets is performed via:

```bash
python src/benchmark_models.py \
  --model stage_2_dce_sar_best_of_all_round_2=checkpoints/stage_2_dce_sar_best_of_all_round_2.pt \
  --dataset SARD=cfg/old/sardyolo.yaml \
  --dataset HERIDAL=cfg/old/heridalyolo.yaml \
  --dataset VISDRONE=cfg/old/visdrone_person.yaml \
  --dataset NTUT4K=cfg/old/ntut4kyolo.yaml \
  --imgsz 800 \
  --batch 32 \
  --conf 0.25 \
  --iou 0.50 \
  --plots
```

For **SARD, HERIDAL, VisDrone**, the script calls:

* `YOLO(model).val(...)` with:

  * `split = "val"`, `imgsz = 800`, `batch = 32`
  * `conf = 0.25`, `iou = 0.50`

From the Ultralytics `val` object, the following metrics are recorded:

* `precision = box.p[0]`
* `recall = box.r[0]`
* `mAP50 = box.map50`
* `mAP50-95 = box.map`
* `fitness` (Ultralytics’ combined score)

For **NTUT4K**, a special **FP-only** evaluation is used:

* The script runs `model.predict()` over NTUT4K **val images** with the same `imgsz`, `conf`, and `iou`.
* All detections are interpreted as **false positives** (no GTs are used).
* The following metrics are computed:

  * `avg_fp_per_image`
  * `total_preds`
  * `max_preds_single_image`

This protocol explicitly measures how aggressively the model fires on **pure background** or non-target patterns.

---

## 7. Results

### 7.1 Stage 1 vs Stage 2 (ours)

The following table compares **Stage 1** and **Stage 2** on the validation splits of SARD, HERIDAL, and VisDrone (GT-based), plus FP-only stats on NTUT4K.

**Per-dataset metrics (val, conf = 0.25, IoU = 0.50, imgsz = 800)**

|      Dataset | Stage |  Precision |     Recall |      mAP50 |   mAP50-95 |
| -----------: | :---: | ---------: | ---------: | ---------: | ---------: |
|     **SARD** |   1   |     0.8852 |     0.7221 |     0.8234 |     0.3762 |
|              | **2** |     0.8702 |     0.7286 |     0.8256 | **0.4280** |
|  **HERIDAL** |   1   |     0.5661 |     0.3070 |     0.4371 |     0.1805 |
|              | **2** | **0.7050** |     0.1801 |     0.4212 | **0.1896** |
| **VISDRONE** |   1   |     0.7345 | **0.4897** |     0.6249 |     0.2516 |
|              | **2** | **0.7997** |     0.4476 | **0.6338** | **0.2920** |

**NTUT4K (FP-only, val)**

| Stage | images | total preds | avg FP / image | max preds per image |
| :---: | -----: | ----------: | -------------: | ------------------: |
| **1** |    154 |         146 |          0.948 |                  16 |
| **2** |    154 |          45 |      **0.292** |                   7 |

**Aggregated (weighted by val-set size over SARD + HERIDAL + VisDrone):**

* Precision:

  * Stage 1: ≈ 0.79
  * Stage 2: **≈ 0.83**
* Recall:

  * Stage 1: ≈ 0.59
  * Stage 2: ≈ 0.57
* **mAP50**:

  * Stage 1: ≈ 0.709
  * Stage 2: **≈ 0.710** (essentially unchanged)
* **mAP50-95**:

  * Stage 1: ≈ 0.312
  * Stage 2: **≈ 0.354** (≈ +13.5% relative)

In summary:

> Stage 2 hard-negative fine-tuning **reduces false positives by ~3× on NTUT4K**, improves **mAP50-95** on all three GT-datasets, **increases overall precision**, and maintains the same mAP50 with a moderate recall trade-off – matching the requirements of practical SAR deployments where false positives are particularly costly.

---

## 8. Relation to Prior Work

This project is inspired by and builds directly on:

* **DCE-YOLOv8**
  J. An, D. H. Lee, M. D. Putro, B. W. Kim,
  *DCE-YOLOv8: Lightweight and Accurate Object Detection for Drone Vision*,
  IEEE Access, vol. 12, pp. 170898–170912, 2024,
  doi:10.1109/ACCESS.2024.3481410.

and SAR-oriented object detection literature, in particular:

* **Benchmarking YOLOv5 for SAR human detection (SARD & HERIDAL)**
  N. Bachir, Q. A. Memon,
  *Benchmarking YOLOv5 models for improved human detection in search and rescue missions*,
  Journal of Electronic Science and Technology, vol. 22, no. 1, 100243, 2024,
  doi:10.1016/j.jnlest.2024.100243.

* **SARD Dataset and YOLOv5 on SARD**
  N. A. Bachir,
  *Human Detection from Aerial Imagery for Search and Rescue Operations*,
  M.Sc. thesis, United Arab Emirates University, 2022.

The main methodological differences are:

1. **Multi-dataset training** of a single detector over SARD, HERIDAL, VisDrone, and NTUT4K.
2. **Integration of DCE/ERB/SCDown modules** into a YOLOv8m-scale architecture specialised for single-class SAR person detection.
3. A **two-stage training regime** with explicit **hard-negative fine-tuning** and an **FP-only evaluation** on NTUT4K to quantify background robustness.

---

## 9. Reproducibility Notes

To reproduce the reported results:

1. Prepare the datasets with the given train/val splits in YOLO format (single class `person`).
2. Create and adjust the dataset YAMLs:

   * `cfg/sar_composite.yaml`
   * `cfg/sar_hardneg_stage2.yaml`
   * per-dataset eval YAMLs in `cfg/old/`.
3. Install dependencies (matching at least `ultralytics==8.3.231` and `torch==2.1.1+cu121`).
4. Run Stage 1 training with `train_stage1_sar_dce.py`.
5. Run Stage 2 fine-tuning with `train_stage2_hardneg.py` using the Stage 1 best checkpoint as `--pretrained`.
6. Evaluate with `benchmark_models.py` on all four datasets as shown above.

For exact reproducibility, it is recommended to:

* Export `pip freeze` to `environment.txt`.
* Version-control:

  * model YAML (`yolov8m_dce.yaml`),
  * dataset YAMLs,
  * Ultralytics-specific config changes (e.g. custom `tasks.py`).
* Log the full `args.yaml` and `results.csv` for each training run.

---

## 10. Citation

If you use this repository, please cite the underlying DCE-YOLOv8 and SAR detection works, and consider citing the corresponding publication once available.

```bibtex
@article{an2024dceyolov8,
  title   = {DCE-YOLOv8: Lightweight and Accurate Object Detection for Drone Vision},
  author  = {An, Jinsu and Lee, Donghee and Putro, Muhamad Dwisnanto and Kim, Byeong Woo},
  journal = {IEEE Access},
  volume  = {12},
  pages   = {170898--170912},
  year    = {2024},
  doi     = {10.1109/ACCESS.2024.3481410}
}

@article{bachir2024yolov5sar,
  title   = {Benchmarking YOLOv5 models for improved human detection in search and rescue missions},
  author  = {Bachir, Namat and Memon, Qurban Ali},
  journal = {Journal of Electronic Science and Technology},
  volume  = {22},
  number  = {1},
  pages   = {100243},
  year    = {2024},
  doi     = {10.1016/j.jnlest.2024.100243}
}
```




