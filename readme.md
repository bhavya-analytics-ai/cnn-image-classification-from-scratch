# CNN Toilet & Seat Cover Classifier

End-to-end Convolutional Neural Network built from scratch for toilet brand and seat cover type classification. Trained on two datasets with and without data augmentation, evaluated with full classification metrics, and deployed via Streamlit GUI and FastAPI.

---

## What it does

- Classifies toilet images into 3 brands: Aquapro, Montecarlo, Smart
- Classifies seat cover images into 2 types: Elongated, Round
- Compares baseline vs augmented model performance
- Deployed as an interactive GUI and a REST API

---

## Architecture

Built from scratch — no transfer learning, no pretrained weights.

```
Input (3 × 224 × 224)
→ Conv(32) → ReLU → MaxPool
→ Conv(64) → ReLU → MaxPool
→ Conv(128) → ReLU → MaxPool
→ Flatten
→ FC(256) → ReLU → Dropout
→ Output Layer
```

- Toilet classifier: 3 output neurons (Aquapro, Montecarlo, Smart)
- Seat cover classifier: 2 output neurons (Elongated, Round)

---

## Preprocessing

Images come in varying sizes and aspect ratios. To prevent distortion:

- Resize longest side to 224px
- Pad remaining sides to reach 224×224
- Normalize pixel values

Two pipelines:
- **Base transform** — no augmentation
- **Augmented transform** — random flip, rotation, color jitter

---

## Training

| Setting | Value |
|---------|-------|
| Loss | CrossEntropyLoss |
| Optimizer | Adam (lr=0.001) |
| Epochs | 10 |
| Batch size | 32 |
| Hardware | CUDA / Apple MPS |

Four models trained:
- Toilet — no augmentation
- Toilet — with augmentation
- Seat cover — no augmentation
- Seat cover — with augmentation

---

## Results

**Toilet classifier:** No augmentation → best model. Near-perfect performance on test set.

**Seat cover classifier:** No augmentation → best model. Augmentation reduced performance due to small dataset size and shape sensitivity (Elongated vs Round is a subtle geometric difference that augmentation distorts).

Misclassified samples saved to `misclassified_*/` for analysis.

---

## Dataset Structure

```
data/
└── splits/
    ├── Seat Cover/
    │   ├── train/ (Elongated, Round)
    │   ├── val/
    │   └── test/
    └── Toilets/
        ├── train/ (Aquapro, Montecarlo, Smart)
        ├── val/
        └── test/
```

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Trained Models

Saved in `models/`:

| File | Dataset | Augmentation |
|------|---------|-------------|
| `toilet_no_aug_final.pth` | Toilet | No |
| `toilet_aug_final.pth` | Toilet | Yes |
| `seat_no_aug_final.pth` | Seat Cover | No |
| `seat_aug_final.pth` | Seat Cover | Yes |

---

## Run

**Streamlit GUI:**
```bash
streamlit run app_cnn.py
```

Upload an image → select dataset → get prediction, confidence score, and class probability bars.

**FastAPI:**
```bash
uvicorn api_cnn:app --reload
```
```
http://127.0.0.1:8000/docs
```

---

## Demo

- `assets/demo_toilet.gif` — toilet classification demo
- `assets/demo_seatcover.gif` — seat cover classification demo

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Author

Bhavya Pandya — Master's Student, Artificial Intelligence
