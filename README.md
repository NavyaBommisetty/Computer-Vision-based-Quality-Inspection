Fruit & VegetableQuality Inspection using Classical Computer Vision (OpenCV)

Project Description

This project presents an end-to-end classical computer vision system for inspecting the quality of fruits and vegetables and classifying them as acceptable or rotten based on visual defects.
The system is implemented using OpenCV and relies on image processing, color segmentation, contour analysis, and rule-based decision logic, without using machine learning or deep learning models.
The project focuses on interpretability, explainability, and engineering clarity, similar to traditional vision systems used in low-cost or edge-device industrial inspection setups.
Currently supported items:
Banana
Lemon
Potato
Each item is handled by a dedicated pipeline tuned to its visual properties

Why This Project
Automated quality inspection is a common problem in agriculture, food processing, and supply chain automation.
While modern solutions often rely on deep learning, many real-world systems still use classical vision techniques due to constraints such as:
Limited compute resources
Lack of large labeled datasets
Requirement for transparent decision logic
This project demonstrates how far classical computer vision alone can be pushed before learning-based methods become necessary.

System Pipeline (End-to-End)
1.Input Image Acquisition
  -RGB images captured under controlled conditions
2.Preprocessing
  -Image resizing
  -Noise reduction
  -Conversion from BGR to HSV color space
3.Object Segmentation
  -Color-based masking to isolate the fruit or vegetable
  -Morphological operations to clean the mask
  -Largest contour extraction to remove background artifacts
4.Defect Analysis
  -Detection of dark spots and discolored regions
  -Edge detection within the segmented object
  -Calculation of defect ratios relative to total object area
5.Decision Logic
  -Rule-based thresholds on:
    -Color ratio
    -Spot ratio
    -Edge density
  -Final classification as ACCEPTED or REJECTED
6.Visualization
  -Segmentation masks
  -Edge maps
  -Annotated original image with inspection results
  
Project Structure

├── Banana.py
├── lemon2.py
├── potato.py
├── sample_images/
│   ├── banana/
│   ├── lemon/
│   └── potato/
├── outputs/
│   ├── banana_results/
│   ├── lemon_results/
│   └── potato_results/
└── README.md
Each script is independent and tuned for the color, texture, and surface characteristics of the corresponding item.

Output Examples
The system generates:
Binary segmentation masks
Detected defect regions
Edge maps highlighting surface irregularities
Final quality decision with visual overlays
Sample outputs demonstrate both good and rotten cases with clear visual justification for each decision.

Technologies Used:
Python
OpenCV
NumPy
Matplotlib

Design Choices
-HSV color space for robust color segmentation
-Contour-based filtering to isolate the primary object
-Rule-based thresholds for transparent decision making
-Visual explainability over black-box prediction
  

This project uses rule-based computer vision, which has known limitations:
-Sensitive to lighting variations
-Thresholds require manual tuning
-Limited generalization across datasets
-Not robust to complex backgrounds

How to Run
Install dependencies:
pip install opencv-python numpy matplotlib

Run the desired inspection script:
python Banana.py
python lemon.py
python potato2.py


