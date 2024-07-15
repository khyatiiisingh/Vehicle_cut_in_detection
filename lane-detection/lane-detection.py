import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines):
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 10)

def process_frame(frame):
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    vertices = np.array([[(0, height), (width / 2, height / 2), (width, height)]], dtype=np.int32)
    roi = region_of_interest(edges, vertices)

    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=150)
    line_image = np.zeros_like(frame)
    draw_lines(line_image, lines)

    combined = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return combined

cap = cv2.VideoCapture('lanes_clip.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = process_frame(frame)
    cv2.imshow('Lane Detection', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
