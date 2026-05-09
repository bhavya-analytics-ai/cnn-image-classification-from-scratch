from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
import io
import csv, os
from datetime import datetime

# =====================================================
# APP
# =====================================================
app = FastAPI(
    title="Sprint-2 CNN Image Classification API",
    version="1.0"
)

# =====================================================
# DEVICE
# =====================================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# =====================================================
# CNN ARCH
# =====================================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*28*28,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,num_classes)
        )

    def forward(self,x):
        return self.classifier(self.features(x))

# =====================================================
# PREPROCESS
# =====================================================
class ResizeWithPadding:
    def __call__(self, img):
        w, h = img.size
        m = max(w, h)
        pl = (m-w)//2
        pr = m-w-pl
        pt = (m-h)//2
        pb = m-h-pt
        return F.pad(img,(pl,pt,pr,pb),fill=0)

infer_transform = transforms.Compose([
    ResizeWithPadding(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# =====================================================
# LOAD MODELS
# =====================================================
seat_classes = ["Elongated","Round"]
toilet_classes = ["Aquapro","Montecarlo","Smart"]

seat_model = SimpleCNN(2).to(device)
seat_model.load_state_dict(torch.load("seat_no_aug_final.pth", map_location=device))
seat_model.eval()

toilet_model = SimpleCNN(3).to(device)
toilet_model.load_state_dict(torch.load("toilet_no_aug_final.pth", map_location=device))
toilet_model.eval()

# =====================================================
# LOGGING
# =====================================================
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR,"predictions.csv")
os.makedirs(LOG_DIR, exist_ok=True)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE,"w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp","dataset","prediction","confidence"])

# =====================================================
# RESPONSE MODEL
# =====================================================
class PredictionResponse(BaseModel):
    dataset: str
    prediction: str
    confidence: float
    probabilities: dict

# =====================================================
# INFERENCE
# =====================================================
def run_inference(img, model, classes):
    img = infer_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(img),dim=1).cpu().numpy()[0]

    prob_dict = {classes[i]:round(float(probs[i]),4) for i in range(len(classes))}
    idx = probs.argmax()

    return classes[idx], float(probs[idx]), prob_dict

# =====================================================
# HEALTH
# =====================================================
@app.get("/health")
def health():
    return {"status":"ok","device":str(device),"models_loaded":True}

# =====================================================
# PREDICT
# =====================================================
@app.post("/predict", response_model=PredictionResponse)
async def predict_api(
    file: UploadFile = File(...),
    dataset: str = Query(..., enum=["seat","toilet"])
):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="Invalid image")

    if dataset=="seat":
        label, conf, probs = run_inference(img, seat_model, seat_classes)
    else:
        label, conf, probs = run_inference(img, toilet_model, toilet_classes)

    # log
    with open(LOG_FILE,"a",newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(),dataset,label,round(conf,4)])

    return PredictionResponse(
        dataset=dataset,
        prediction=label,
        confidence=round(conf,4),
        probabilities=probs
    )
