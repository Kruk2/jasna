import cv2
import torch
import numpy as np
from pathlib import Path
from jasna.mosaic.rfdetr import RfDetrMosaicDetectionModel

IMAGE_PATH = r"000002062.jpg"
ONNX_PATH = Path("model_weights/rfdetr.onnx")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
RESOLUTION = 768
SCORE_THRESHOLD = 0.3

img = cv2.imread(IMAGE_PATH)
assert img is not None, f"Failed to load image from {IMAGE_PATH}"

h, w = img.shape[:2]

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_bchw = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.uint8)
img_bchw = torch.nn.functional.pad(img_bchw, (0, 0, 0, 0, 0, 0, 0, 3), mode='constant', value=0)

stream = torch.cuda.Stream(device=DEVICE)
with torch.cuda.stream(stream):
    model = RfDetrMosaicDetectionModel(
        onnx_path=ONNX_PATH,
        stream=stream,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        resolution=RESOLUTION,
        score_threshold=SCORE_THRESHOLD,
    )
    detections = model(img_bchw, target_hw=(h, w))

    boxes_list = detections.boxes_xyxy
    masks_list = detections.masks

result = img.copy().astype(np.float32)

if len(masks_list[0]) > 0:
    masks = masks_list[0]
    
    for i, mask in enumerate(masks):
        mask_uint8 = (mask.cpu().numpy() * 255).astype(np.uint8)
        mask_upscaled = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_LINEAR) > 127
        color = np.random.randint(0, 256, 3).tolist()
        result[mask_upscaled] = result[mask_upscaled] * 0.5 + np.array(color) * 0.5

result = result.astype(np.uint8)

boxes = boxes_list[0]
for box in boxes:
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Detections", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
