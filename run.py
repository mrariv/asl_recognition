def main():
    import cv2
    import torch
    import numpy as np
    import math
    import yaml
    import os
    from cvzone.HandTrackingModule import HandDetector
    from collections import deque, Counter
    from torchvision import transforms
    from model.architecture import HandGestureCNN

    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)

    device = 'cpu'

    with open(os.path.join('model', 'metadata.yaml'), 'r') as f:
        metadata = yaml.safe_load(f)
        labels = metadata['classes']
        input_size = metadata['input_size']

    model = torch.jit.load(os.path.join('model', 'architecture_initial.pt'), map_location=device)
    model.eval()

    transform = torch.load(os.path.join('model', 'transform.pth'), map_location=device, weights_only=False)

    offset = 20
    imgSize = 300

    prediction_history = deque(maxlen=10)

    while True:
        success, img = cap.read()
        if not success:
            continue

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            aspectRatio = h / w

            try:
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                imgInput = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB)
                imgInput = transform(imgInput).unsqueeze(0).to(device)

                # Уверенность (гиперпараметр)
                confidence_threshold = 0.2

                # Предсказание
                with torch.no_grad():
                    outputs = model(imgInput)
                    _, predicted = torch.max(outputs.data, 1)
                    predicted_class = predicted.item()

                confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted_class].item()

                if confidence >= confidence_threshold:
                    prediction_history.append(predicted_class)

                if prediction_history:
                    most_common_prediction = Counter(prediction_history).most_common(1)[0][0]
                    label_text = labels[most_common_prediction]
                else:
                    label_text = "..."

                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (255, 0, 0), 2)
                cv2.putText(imgOutput, label_text,
                            (x, y - offset - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                cv2.putText(imgOutput, f"{confidence:.2f}",
                            (x, y + h + offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            except Exception as e:
                print(f"Ошибка обработки: {e}")

        cv2.imshow("ASL Recognition", imgOutput)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
