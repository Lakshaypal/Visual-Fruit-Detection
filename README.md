# Fruit Detection from Plant Images

A small computer-vision project that runs an Ultralytics YOLO checkpoint on plant photographs and saves annotated images. A second script processes front/back image pairs.

## What the scripts do

- `scripts/detect.py`: runs inference on the JPG images in `data/` and writes annotated results.
- `scripts/process_all.py`: finds numbered `front_imgN.jpg` images, looks for their matching `back_imgN.jpg`, and runs inference on both views.

The current paired-view script does **not** match detections across views or compute a unique-fruit count. The repository demonstrates detection and paired-image processing, rather than a validated multi-view counting algorithm.

## Setup

Use Python 3.10+ and run from the repository root:

```sh
git clone https://github.com/Lakshaypal/Fruit-detecting-system.git
cd Fruit-detecting-system
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Retrieve the committed checkpoint and sample images, including Git LFS content where applicable:

```sh
git lfs pull
```

The scripts resolve data, weights and output paths relative to the repository, so they also work when launched from another working directory.

```sh
python scripts/detect.py
python scripts/process_all.py
```

The checkpoint is `models/best.pt`; outputs are written to `results/`. Supply only model checkpoints you trust. No training pipeline, held-out accuracy benchmark or cross-view correspondence model is included.

## Stack

Python, Ultralytics YOLO, OpenCV and NumPy. Existing annotated outputs are included as examples.
