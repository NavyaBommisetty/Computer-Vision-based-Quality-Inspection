import cv2
import numpy as np
import matplotlib.pyplot as plt

def classify_banana(image_path, show_steps=False):
    # --------------- Tunable thresholds ---------------
    # mean V (brightness) below which banana is considered too dark
    MIN_MEAN_V = 90
    # mean saturation below which banana is considered desaturated/brown
    MIN_MEAN_S = 45
    # fraction of banana area that are dark spots (V < SPOT_V_TH) to consider bad
    MAX_SPOT_RATIO = 0.12
    SPOT_V_TH = 80
    # edge ratio threshold inside banana
    MAX_EDGE_RATIO = 0.12

    # --------------- Load image ---------------
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found!")
        return
    original = img.copy()
    h_img, w_img = img.shape[:2]

    # --------------- Convert to HSV and isolate banana area ---------------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # A broad yellow/brown mask to capture bananas at different ripeness
    # these ranges are conservative; tweak if needed
    lower = np.array([5, 40, 40])    # includes yellow -> brownish
    upper = np.array([40, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find largest contour (assumed banana)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # If no yellow/brown region found, fallback — treat as bad
        cv2.putText(original, "BAD / REJECTED (no banana region)", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        plt.figure(figsize=(8, 6)); plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB)); plt.axis('off'); plt.show()
        print("RESULT: BAD / REJECTED (no banana region detected)")
        return

    # Get largest contour by area
    largest = max(contours, key=cv2.contourArea)
    banana_mask = np.zeros_like(mask)
    cv2.drawContours(banana_mask, [largest], -1, 255, -1)
    banana_area = np.sum(banana_mask == 255)
    if banana_area == 0:
        print("Zero banana area found, rejecting.")
        return

    # --------------- Compute statistics but only inside banana_mask ---------------
    # mean brightness and saturation inside banana
    mean_v = cv2.mean(v, mask=banana_mask)[0]
    mean_s = cv2.mean(s, mask=banana_mask)[0]

    # spot ratio: dark pixels inside banana (low V)
    _, dark_pixels = cv2.threshold(v, SPOT_V_TH, 255, cv2.THRESH_BINARY_INV)
    dark_spots_mask = cv2.bitwise_and(dark_pixels, banana_mask)
    spot_ratio = np.sum(dark_spots_mask == 255) / banana_area

    # edge ratio computed only inside banana bounding box for speed
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 200)
    edges_in_banana = cv2.bitwise_and(edges, edges, mask=banana_mask)
    edge_ratio = np.sum(edges_in_banana == 255) / banana_area

    # --------------- Decision rules ---------------
    bad = False
    reasons = []

    if mean_v < MIN_MEAN_V:
        bad = True
        reasons.append(f"Too dark (mean V={mean_v:.1f} < {MIN_MEAN_V})")

    if mean_s < MIN_MEAN_S:
        bad = True
        reasons.append(f"Low saturation (brown/desaturated) (mean S={mean_s:.1f} < {MIN_MEAN_S})")

    if spot_ratio > MAX_SPOT_RATIO:
        bad = True
        reasons.append(f"Too many dark spots (spot ratio={spot_ratio:.3f} > {MAX_SPOT_RATIO})")

    if edge_ratio > MAX_EDGE_RATIO:
        bad = True
        reasons.append(f"Rough texture (edge ratio={edge_ratio:.3f} > {MAX_EDGE_RATIO})")


    if bad:
        result_text = "BAD / REJECTED"
        color = (0, 0, 255)  # red
    else:
        result_text = "GOOD & ACCEPTED"
        color = (0, 255, 0)  # green

    # draw banana contour and bounding box for visualization
    x, y, w, h_box = cv2.boundingRect(largest)
    cv2.rectangle(original, (x, y), (x + w, y + h_box), color, 2)
    cv2.drawContours(original, [largest], -1, color, 2)

    cv2.putText(original, result_text, (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
    cv2.putText(original, f"Mean V: {mean_v:.1f}", (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(original, f"Mean S: {mean_s:.1f}", (30, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(original, f"Spots: {spot_ratio:.3f}", (30, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(original, f"Edges: {edge_ratio:.3f}", (30, 215),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

   
    if show_steps:
        plt.figure(figsize=(12, 8))
        plt.subplot(231); plt.title("Original"); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.axis('off')
        plt.subplot(232); plt.title("HSV - V channel"); plt.imshow(v, cmap='gray'); plt.axis('off')
        plt.subplot(233); plt.title("Initial color mask"); plt.imshow(mask, cmap='gray'); plt.axis('off')
        plt.subplot(234); plt.title("Banana mask (largest)"); plt.imshow(banana_mask, cmap='gray'); plt.axis('off')
        plt.subplot(235); plt.title("Dark spots mask (inside banana)"); plt.imshow(dark_spots_mask, cmap='gray'); plt.axis('off')
        plt.subplot(236); plt.title("Edges in banana"); plt.imshow(edges_in_banana, cmap='gray'); plt.axis('off')
        plt.show()

    # --------------- Final output ---------------
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    print("RESULT:", result_text)
    print(f"Mean V     : {mean_v:.2f}")
    print(f"Mean S     : {mean_s:.2f}")
    print(f"Spot Ratio : {spot_ratio:.4f}")
    print(f"Edge Ratio : {edge_ratio:.4f}")
    if bad:
        print("Reasons:", reasons)

# Set show_steps=True to visualize masks and debug thresholds
#classify_banana("/home/rguktrkvalley/Downloads/rotten_banana1.jpeg", show_steps=True)
#classify_banana("/home/rguktrkvalley/Downloads/good_banana1.jpeg", show_steps=True)
# Captured Images(Original images )
classify_banana("/home/rguktrkvalley/Downloads/rotten_fresh/WhatsApp Image 2025-11-21 at 10.29.59 PM.jpeg" ,show_steps=True)
classify_banana("/home/rguktrkvalley/Downloads/rotten_fresh/WhatsApp Image 2025-11-21 at 10.24.39 PM.jpeg" ,show_steps=True)
#3
#classify_banana("/home/rguktrkvalley/Downloads/WhatsApp Image 2025-11-21 at 10.29.59 PM.jpeg" ,show_steps=True)
classify_banana("/home/rguktrkvalley/Downloads/rotten_fresh/WhatsApp Image 2025-11-21 at 8.31.55 PM.jpeg",show_steps=True)

