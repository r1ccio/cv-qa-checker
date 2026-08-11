# CV Annotation QA Checker

This repository contains a Python script for automated Quality Assurance (QA) of Computer Vision datasets. It compares human-annotated data against Ground Truth (GT) references and generates a structured JSON report detailing missing objects, false positives, attribute mismatches, and individual object scores.

##  Core Algorithm

Since object IDs (`nm`) between the Ground Truth and the Annotated files do not necessarily match, the script cannot rely on simple key comparisons. Instead, it performs **Spatial Matching**:

1. **Data Normalization:** Merges object metadata (types, attributes) with their geometric coordinates from the JSON files.
2. **Global Attributes Check:** Validates frame-level metadata (e.g., weather conditions).
3. **Spatial Matching (IoU):** 
   - Generates raster masks for polygons and bounding boxes using `OpenCV`.
   - *Note on binarization:* Applies a threshold of `1` to GT masks to prevent dark colors (like red road barriers) from being zeroed out.
   - Calculates the **Intersection over Union (IoU)** metric between GT masks and Annotated masks.
   - Pairs objects that have an IoU >= `0.5`.
4. **Attribute Validation:** Compares classes and nested attributes (e.g., `oncoming`) for the matched pairs.
5. **False Positives & Negatives Detection:** Identifies extra objects drawn by the annotator (FP) and objects they missed (FN). It calculates dynamic reasons for these errors (e.g., distinguishing between "Not annotated at all" and "Poor localization with IoU < 0.5").
6. **Report Generation:** Outputs all findings into a structured `qa_report.json`.

##  Development Plan & Algorithm Description

### 1. Development Plan
1. **Requirements & Data Analysis:** Review the input JSON format to understand how metadata (classes, attributes) and geometric data (polygons, bounding boxes) are structured.
2. **Architecture Design:** Decide against ID-based matching (since IDs might differ) and design a spatial matching approach using Computer Vision metrics.
3. **Core Logic Implementation:** 
   - Parse JSON and merge metadata with coordinates.
   - Implement OpenCV rasterization to draw masks.
   - Implement Intersection over Union (IoU) calculation logic.
4. **QA Metrics Implementation:** Categorize errors into False Negatives (FN), False Positives (FP), and Attribute Mismatches. Implement dynamic reasoning for FN/FP (e.g., distinguishing "missed entirely" from "poorly localized").
5. **Scoring System:** Implement a 0-100% scoring logic based on the IoU of matched objects and penalties for hallucinated objects.

### 2. Algorithm & Scoring Logic
- **Top Level:** The script extracts archives, parses JSONs into dictionaries, spatially matches objects, validates attributes, calculates a global file score, and dumps a structured JSON report.
- **Geometric Comparison:** OpenCV is used to render polygons/bboxes on a blank canvas. `cv2.bitwise_and` and `cv2.bitwise_or` are used to calculate the Intersection and Union pixel areas, generating an IoU score (0.0 to 1.0).
- **Score Combination:** 
  - Each Ground Truth object receives a score based on its best IoU with any annotated mask (*IoU * 100*). These are listed individually in `object_scores`.
  - Pure hallucinations (annotated objects that didn't overlap with any GT) receive a score of `0.0`.
  - The **Overall File Score** is calculated as the average score of all expected GT objects, penalized by the number of unique hallucinated objects (False Positives).

##  Installation & Setup

1. Clone the repository:

        git clone https://github.com/r1ccio/cv-qa-checker.git
        cd cv-qa-checker

2. Install the required dependencies:

        pip install opencv-python numpy

##  Usage

**Data Preparation:**
The script features **automatic extraction**. You do not need to unzip the files manually. Ensure your project structure looks like this before running:

         cv-qa-checker/
        ├──  annotated/           # Folder containing road.jpg.zip
        ├──  groubd_truth/        # Folder containing road.jpg.zip
        └──  main.py              # Main analysis script
      

**Run the analysis:**

        python main.py

## Output

The script will unpack the archives and generate a `qa_report.json` file in the root directory. It provides a clear summary of matched objects, global errors, matching errors, individual `object_scores`, and detailed lists of missed or extra objects with smart reasons for their classification.
