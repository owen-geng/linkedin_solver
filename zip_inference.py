import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from zip_digits_train import ZipDigitClassifier

IMG_SIZE = 28
CENTROID_MARGIN = 0.10  # blobs whose centroid is within the outer 10% on any side are discarded

_model = None
_idx_to_label = None
_device = None

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

def _load_model():
    global _model, _idx_to_label, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load('zip.pt', map_location=_device)
    _model = ZipDigitClassifier(num_classes=checkpoint['num_classes']).to(_device)
    _model.load_state_dict(checkpoint['model_state'])
    _model.eval()
    _idx_to_label = checkpoint['idx_to_label']

def _infer_single(gray):
    """Run inference on a single-digit grayscale numpy image."""
    pil_img = Image.fromarray(gray)
    tensor = _transform(pil_img).unsqueeze(0).to(_device)
    with torch.no_grad():
        output = _model(tensor)
        probs = torch.softmax(output, dim=1)
        idx = probs.argmax(dim=1).item()
        confidence = probs[0, idx].item()
    return _idx_to_label[idx], confidence

def _find_digit_blobs(gray):
    """
    Returns bboxes (x, y, w, h) of significant digit blobs, sorted left to right.
    Filters out blobs whose centroid falls in the outer CENTROID_MARGIN of the image,
    which removes circular-border artifacts without penalising partially-cropped digits.
    """
    h_img, w_img = gray.shape
    margin_x = w_img * CENTROID_MARGIN
    margin_y = h_img * CENTROID_MARGIN

    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    min_area = h_img * w_img * 0.01
    bboxes = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        cx, cy = centroids[i]
        if cx < margin_x or cx > w_img - margin_x:
            continue
        if cy < margin_y or cy > h_img - margin_y:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        bboxes.append((x, y, w, h))

    bboxes.sort(key=lambda b: b[0])
    return bboxes

def _blob_to_square(img, bbox, pad=8):
    """Crop a blob from img and pad to a square canvas, preserving aspect ratio."""
    x, y, w, h = bbox
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(img.shape[1], x + w + pad)
    y1 = min(img.shape[0], y + h + pad)
    crop = img[y0:y1, x0:x1]
    ch, cw = crop.shape
    side = max(ch, cw)
    canvas = np.zeros((side, side), dtype=np.uint8)
    canvas[(side - ch) // 2:(side - ch) // 2 + ch,
           (side - cw) // 2:(side - cw) // 2 + cw] = crop
    return canvas

def predict_digit(img):
    """
    Takes a grayscale or BGR OpenCV image.
    Auto-detects 1 or 2 digit numbers.
    Returns (digit_value: int, confidence: float).
    For 2-digit numbers, confidence is the minimum of both digits' confidences.
    """
    if _model is None:
        _load_model()

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bboxes = _find_digit_blobs(img)

    if len(bboxes) >= 2:
        tens, conf_tens = _infer_single(_blob_to_square(img, bboxes[0]))
        ones, conf_ones = _infer_single(_blob_to_square(img, bboxes[1]))
        return tens * 10 + ones, min(conf_tens, conf_ones)

    # Single digit — run inference on the full image
    pil_img = Image.fromarray(img)
    tensor = _transform(pil_img).unsqueeze(0).to(_device)
    with torch.no_grad():
        output = _model(tensor)
        probs = torch.softmax(output, dim=1)
        idx = probs.argmax(dim=1).item()
        confidence = probs[0, idx].item()
    return _idx_to_label[idx], confidence
