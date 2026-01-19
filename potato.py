import cv2
import numpy as np
import matplotlib.pyplot as plt


def classify_potato(image_path):

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, fg_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((7,7), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("No potato found")
        return
    c = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [c], -1, 255, -1)
    potato_pixels = cv2.bitwise_and(gray, gray, mask=mask)
    _, rotten_mask = cv2.threshold(potato_pixels, 90, 255, cv2.THRESH_BINARY_INV)

    rotten_mask = cv2.morphologyEx(rotten_mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

    rotten_area = cv2.countNonZero(rotten_mask)
    potato_area = cv2.countNonZero(mask)

    rotten_percent = (rotten_area / potato_area) * 100

    # --- decision rule ---
    if rotten_percent > 20:   # rotten patch >5% → BAD
        result = "REJECTED"
        color = (255, 0, 0)
    else:
        result = "ACCEPTED"
        color = (0, 255, 0)

    # draw contour
    output_img = img_rgb.copy()
    cv2.drawContours(output_img, [c], -1, color, 4)

    # show results
    plt.figure(figsize=(6,6))
    plt.imshow(output_img)
    plt.title(f"{result}\nRotten Area = {rotten_percent:.2f}%")
    plt.axis("off")
    plt.show()

    print("Result =", result)


classify_potato("/home/rguktrkvalley/Downloads/rotten_fresh/rotten2.jpeg")
classify_potato("/home/rguktrkvalley/Downloads/rotten_fresh/rotten3.jpeg")
classify_potato("/home/rguktrkvalley/Downloads/rotten_fresh/good_potato2.jpeg")


