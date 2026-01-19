import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def classify_lemon_with_thermal(image_path,
                               save_outputs=False,
                               colormap=cv2.COLORMAP_JET,
                               equalize_thermal=True,
                               blend_alpha=0.6):
    """
    Classify lemon and produce a simulated thermal (false-color) visualization.

    Args:
      image_path (str): input RGB image path.
      save_outputs (bool): whether to save thermal and blended images to disk.
      colormap (int): OpenCV colormap constant, e.g. cv2.COLORMAP_JET or cv2.COLORMAP_INFERNO.
      equalize_thermal (bool): apply histogram equalization to V channel before color mapping.
      blend_alpha (float): blending weight for overlay (0-1). 0 -> original, 1 -> thermal fully.
    """

    # ---------------- load image ----------------
    img = cv2.imread(image_path)
    if img is None:
        print("Could not load image!")
        return

    original = img.copy()
    h, w = img.shape[:2]

    # ---------------- convert to HSV ----------------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # ---------------- FEATURE 1: yellow color ratio ----------------
    lower_yellow = np.array([20, 70, 100])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_ratio = np.sum(yellow_mask == 255) / (h * w)

    # ---------------- FEATURE 2: dark spot ratio ----------------
    dark_mask = cv2.inRange(V, 0, 70)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    spot_ratio = np.sum(dark_mask == 255) / (h * w)

    # ---------------- FEATURE 3: edge ratio ----------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 150)
    edge_ratio = np.sum(edges == 255) / (h * w)

    # ---------------- decision thresholds ----------------
    bad = False
    reasons = []

    if yellow_ratio < 0.40:
        bad = True
        reasons.append("Low yellow color")

    if spot_ratio > 0.10:
        bad = True
        reasons.append("High black spot ratio")

    if edge_ratio > 0.10:
        bad = True
        reasons.append("High roughness")

    if not bad:
        label = "GOOD LEMON"
        color = (0, 255, 0)
    else:
        label = "BAD LEMON"
        color = (0, 0, 255)

    # ---------------- annotate the original ----------------
    cv2.putText(original, label, (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
    cv2.putText(original, f"Yellow Ratio: {yellow_ratio:.3f}", (25, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(original, f"Spot Ratio: {spot_ratio:.3f}", (25, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(original, f"Edge Ratio: {edge_ratio:.3f}", (25, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # ---------------- create pseudo-thermal image ----------------
    # Use the V (brightness) channel as the "temperature" proxy
    thermal_source = V.copy()

    # Optional histogram equalization to increase contrast in thermal view
    if equalize_thermal:
        thermal_source = cv2.equalizeHist(thermal_source)

    # Normalize to 0..255 (should already be)
    thermal_norm = cv2.normalize(thermal_source, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply colormap to make it look like thermal (JET, INFERNO, etc.)
    thermal_color = cv2.applyColorMap(thermal_norm, colormap)

    # Optionally blend thermal with the original (convert thermal_color to BGR if not)
    blended = cv2.addWeighted(cv2.cvtColor(thermal_color, cv2.COLOR_BGR2RGB if False else cv2.COLOR_BGR2RGB), 0, cv2.cvtColor(original, cv2.COLOR_BGR2RGB), 1.0, 0)
    # Above line kept for compatibility with matplotlib; we will compute a proper blend below in BGR space
    # Create a proper BGR blend for saving/displaying with cv2 then convert to RGB for matplotlib
    blended_bgr = cv2.addWeighted(thermal_color, blend_alpha, original, 1 - blend_alpha, 0)

    # ---------------- display results side-by-side ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Annotated Original")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(thermal_color, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Simulated Thermal (false-color)")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"Blended (alpha={blend_alpha:.2f})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    # ---------------- print terminal output ----------------
    print("RESULT:", label)
    print("Yellow Ratio:", yellow_ratio)
    print("Spot Ratio:", spot_ratio)
    print("Edge Ratio:", edge_ratio)
    if bad:
        print("Reasons:", reasons)

    # ---------------- optional save ----------------
    if save_outputs:
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_dir = os.path.join(os.path.dirname(image_path), f"{base}_thermal_outputs")
        os.makedirs(out_dir, exist_ok=True)
        thermal_path = os.path.join(out_dir, f"{base}_thermal.png")
        blended_path = os.path.join(out_dir, f"{base}_blended.png")
        # thermal_color and blended_bgr are BGR images (thermal_color from applyColorMap returns BGR)
        cv2.imwrite(thermal_path, thermal_color)
        cv2.imwrite(blended_path, blended_bgr)
        print("Saved thermal to:", thermal_path)
        print("Saved blended to:", blended_path)



classify_lemon_with_thermal("/home/rguktrkvalley/Downloads/rotten_fresh/WhatsApp Image 2025-11-20 at 7.11.57 PM.jpeg",
                           save_outputs=False,
                           colormap=cv2.COLORMAP_JET,
                           equalize_thermal=True,
                           blend_alpha=0.5)

