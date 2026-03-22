import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from zip_digits_train import ZipDigitDataset, ZipDigitClassifier

IMG_SIZE = 28
BATCH_SIZE = 32

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset = ZipDigitDataset('digits_training', transform=transform, split='val')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    checkpoint = torch.load('zip.pt', map_location=device)
    model = ZipDigitClassifier(num_classes=checkpoint['num_classes']).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    idx_to_label = checkpoint['idx_to_label']

    correct = 0
    total = 0
    per_class_correct = {}
    per_class_total = {}

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for pred, label in zip(preds.tolist(), labels.tolist()):
                digit = idx_to_label[label]
                per_class_total[digit] = per_class_total.get(digit, 0) + 1
                if pred == label:
                    per_class_correct[digit] = per_class_correct.get(digit, 0) + 1

    print(f"\nOverall accuracy: {correct}/{total} ({100 * correct / total:.1f}%)\n")
    print("Per-class accuracy:")
    for digit in sorted(per_class_total):
        c = per_class_correct.get(digit, 0)
        t = per_class_total[digit]
        print(f"  {digit:>2}: {c}/{t} ({100 * c / t:.1f}%)")
