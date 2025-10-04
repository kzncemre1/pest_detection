from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
import torch
import random, time
import torchvision.models as models
import torchvision.transforms as transforms
from train_save import load_datasets, config
from sklearn.metrics import confusion_matrix, classification_report
import os
from contextlib import asynccontextmanager
import asyncio
import json
import timm

ROOT_DIR = "/deneme_projesi/archive/goruntuler/test"
CLASSES_PATH = "/deneme_projesi/classes.txt"
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        await asyncio.to_thread(calculate_and_cache_metrics) 
        print("Cache initialization completed!")
    except Exception as e:
        print(f"Cache initialization error: {e}")
    
    yield  
    print("🛑 Server is shutting down...")
    
app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Hello World"}

def load_classes():
    classes = {}
    with open(CLASSES_PATH, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                classes[parts[0]] = " ".join(parts[1:])
    return classes

classes = load_classes()



@app.get("/random-image-info")
def random_image_info():
    random.seed(time.time())  
    all_images = []
    for subdir, dirs, files in os.walk(ROOT_DIR):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(subdir, file)
                all_images.append(full_path)

    if not all_images:
        return JSONResponse(status_code=404, content={"error": "No images found"})

    image_path = random.choice(all_images)
    class_id = os.path.basename(os.path.dirname(image_path))
    class_name = classes.get(class_id, "Unknown")
    relative_path = os.path.relpath(image_path, ROOT_DIR).replace("\\", "/")
    file_name = os.path.basename(image_path)

    return {
        "image_path": relative_path,
        "file_name": file_name,
        "class_id": class_id,
        "class_name": class_name,
    }

@app.get("/image/{image_path:path}")
def serve_image(image_path: str):
    full_path = os.path.join(ROOT_DIR, image_path)
    if not os.path.isfile(full_path):
        return JSONResponse(status_code=404, content={"error": "Image not found"})
    return FileResponse(full_path)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_DIR = "metrics_cache"

def calculate_and_cache_metrics():
    """Calculate the metrics and save to cache"""
    models = {"tiny": model_tiny, "small": model_small, "base":model_base, "convm":model_convmixer_768_32, "swin":model_swin_small, "bigT":model_resnetv2_101x1_bitm}
    test_loader = load_datasets()[2]
    
    for model_name, loaded_model in models.items():
        cache_file = f"{CACHE_DIR}/{model_name}_metrics.json"
        
        
        if os.path.exists(cache_file):
            print(f"{model_name} metrics are available in the cache")
            continue
            
        print(f"Calculating metrics for {model_name}...")
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = loaded_model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
       
        report_dict = classification_report(
            all_labels, all_preds,
            target_names=config['class_names'],
            output_dict=True
        )
        
        cm = confusion_matrix(all_labels, all_preds)
        
        
        metrics_data = {
            "classification_report": report_dict,
            "confusion_matrix": cm.tolist()
        }
        
        with open(cache_file, 'w') as f:
            json.dump(metrics_data, f)
        
        print(f"{model_name} metrics are cached!")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model_tiny = models.convnext_tiny(weights=None)
model_tiny.classifier[2] = torch.nn.Linear(in_features=768, out_features=102)
state_dict = torch.load("saved models/convnext_tiny_model_tam.pth", map_location=device)
model_tiny.load_state_dict(state_dict)
model_tiny.to(device).eval()

model_small = models.convnext_small(weights=None)
model_small.classifier[2] = torch.nn.Linear(in_features=768, out_features=102)  
checkpoint = torch.load("saved models/convnext_small_model_tam.pth", map_location=device)
model_small.load_state_dict(checkpoint['model_state_dict'])
model_small.to(device)
model_small.eval()

model_base = models.convnext_base()
model_base.classifier[2] = torch.nn.Linear(in_features=1024, out_features=102)  
checkpoint = torch.load("saved models/convnext_base_model_tam.pth", map_location=device)
model_base.load_state_dict(checkpoint['model_state_dict'])
model_base.to(device)
model_base.eval()

model_swin_small = timm.create_model('swin_small_patch4_window7_224', pretrained=False, num_classes=102)
checkpoint = torch.load("saved models/swinT_small_model_tam.pth", map_location=device)
model_swin_small.load_state_dict(checkpoint['model_state_dict'])
model_swin_small.to(device)
model_swin_small.eval()

model_convmixer_768_32 = timm.create_model('convmixer_768_32', pretrained=False, num_classes=102)
checkpoint = torch.load("saved models/convMixer_model_tam.pth", map_location=device)
model_convmixer_768_32.load_state_dict(checkpoint['model_state_dict'])
model_convmixer_768_32.to(device)
model_convmixer_768_32.eval()

model_resnetv2_101x1_bitm = timm.create_model('resnetv2_101x1_bitm', pretrained=False, num_classes=102)
checkpoint = torch.load("saved models/bigT_model_tam.pth", map_location=device)
model_resnetv2_101x1_bitm.load_state_dict(checkpoint['model_state_dict'])
model_resnetv2_101x1_bitm.to(device)
model_resnetv2_101x1_bitm.eval()

@app.post("/predict")
async def predict(image: UploadFile = File(...),
                  model: str = Form(...)):
    
    img = Image.open(image.file).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        if model == "tiny":    
            loaded_model = model_tiny                 
        elif model == "small":
            loaded_model = model_small
        elif model == "base":
            loaded_model = model_base
        elif model == "convm":
            loaded_model = model_convmixer_768_32
        elif model == "swin":
            loaded_model = model_swin_small
        else:
            loaded_model = model_resnetv2_101x1_bitm
        
        output = loaded_model(img_tensor)
        _, pred = torch.max(output, 1)
        pred = int(pred)
        pred += 1  

    d = {}
    with open("classes.txt") as f:
        for line in f:
            key, val = line.strip().split(maxsplit=1)
            d[int(key)] = val

    class_id = pred
    class_name = d.get(pred, "Unknown")
       
    return {
        "class_id": class_id,
        "class_name": class_name
    }



@app.get("/metrics")
async def show_metrics(model: str = Query(...)):
    print(model)
    if model not in ["tiny", "small", "base", "convm", "swin", "bigT"]:
        raise HTTPException(status_code=400, detail="Geçersiz model adı")
    
    cache_file = f"{CACHE_DIR}/{model}_metrics.json"
    
    if not os.path.exists(cache_file):
        raise HTTPException(status_code=404, detail="Metrics not yet ready")
    
    with open(cache_file, 'r') as f:
        metrics_data = json.load(f)
    
    return metrics_data
    