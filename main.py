import json
import cv2
import numpy as np
import os
import zipfile

GT_JSON_PATH = "groubd_truth/road.jpg.json"
ANN_JSON_PATH = "annotated/road.jpg.json"
GT_MASKS_DIR = "groubd_truth/road.jpg.images/00001"
REPORT_PATH = "qa_report.json"
IOU_THRESHOLD = 0.5

def extract_archives():
    gt_zip = "groubd_truth/road.jpg.zip"
    ann_zip = "annotated/road.jpg.zip"

    if os.path.exists(gt_zip):
        print("Extracting ground truth archive (groubd_truth)...")
        zip_ref = zipfile.ZipFile(gt_zip, 'r')
        zip_ref.extractall("groubd_truth/")
        zip_ref.close()
            
    if os.path.exists(ann_zip):
        print("Extracting annotator archive (annotated)...")
        zip_ref = zipfile.ZipFile(ann_zip, 'r')
        zip_ref.extractall("annotated/")
        zip_ref.close()

def load_json(filepath):
    f = open(filepath, 'r', encoding='utf-8')
    text = f.read()
    f.close()
    data = json.loads(text)
    return data

def merge_object_data(json_data):
    meta_list = json_data[0]['objects']
    geo_list = json_data[1]['objects']

    merged = {}

    for i in range(len(meta_list)):
        obj = meta_list[i]
        obj_id = obj['nm']
        merged[obj_id] = {}
        for key in obj.keys():
            merged[obj_id][key] = obj[key]

    for i in range(len(geo_list)):
        obj = geo_list[i]
        obj_id = obj['nm']
        
        if obj_id in merged.keys():
            for key in obj.keys():
                merged[obj_id][key] = obj[key]

    return merged

def get_annotated_mask(obj, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)

    if 'path' in obj.keys():
        if len(obj['path']) > 0:
            pts = np.array(obj['path'][0], np.int32)
            cv2.fillPoly(mask, [pts], 255)
    else:
        if 'x1' in obj.keys():
            x1 = int(obj['x1'])
            y1 = int(obj['y1'])
            x2 = int(obj['x2'])
            y2 = int(obj['y2'])
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    return mask

def get_gt_mask(obj, mask_dir, width, height):
    nm = obj['nm']
    mask_path = mask_dir + "/" + nm + ".png"

    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        ret, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        return mask
    else:
        mask = np.zeros((height, width), dtype=np.uint8)
        if 'x1' in obj.keys():
            x1 = int(obj['x1'])
            y1 = int(obj['y1'])
            x2 = int(obj['x2'])
            y2 = int(obj['y2'])
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        return mask

def calculate_iou(mask1, mask2):
    intersection = cv2.bitwise_and(mask1, mask2)
    union = cv2.bitwise_or(mask1, mask2)

    int_sum = np.sum(intersection)
    un_sum = np.sum(union)

    if un_sum == 0:
        return 0.0

    iou = float(int_sum) / float(un_sum)
    return iou

def check_global_attributes(gt_raw, ann_raw):
    errors = []

    gt_attrs = {}
    if 'attributes' in gt_raw[0].keys():
        gt_attrs = gt_raw[0]['attributes']

    ann_attrs = {}
    if 'attributes' in ann_raw[0].keys():
        ann_attrs = ann_raw[0]['attributes']

    for key in gt_attrs.keys():
        if key not in ann_attrs.keys():
            errors.append("Missing global attribute: '" + key + "'")
        else:
            if gt_attrs[key] != ann_attrs[key]:
                errors.append("Global attribute mismatch for '" + key + "': expected '" + str(gt_attrs[key]) + "', got '" + str(ann_attrs[key]) + "'")

    for key in ann_attrs.keys():
        if key not in gt_attrs.keys():
            errors.append("Extra global attribute found: '" + key + "'")

    return errors

def find_best_match(gt_mask, ann_objs, matched_ann_list, img_width, img_height, ann_best_ious):
    best_iou = 0.0
    best_ann_id = ""

    for ann_id in ann_objs.keys():
        is_used = False
        for used_id in matched_ann_list:
            if ann_id == used_id:
                is_used = True

        if is_used == False:
            ann_data = ann_objs[ann_id]
            ann_mask = get_annotated_mask(ann_data, img_width, img_height)
            iou = calculate_iou(gt_mask, ann_mask)

            if iou > ann_best_ious[ann_id]:
                ann_best_ious[ann_id] = iou

            if iou > best_iou:
                best_iou = iou
                best_ann_id = ann_id

    return best_ann_id, best_iou

def check_match_errors(gt_data, ann_data):
    errors = []

    gt_type = gt_data.get('type', '')
    ann_type = ann_data.get('type', '')

    if gt_type != ann_type:
        errors.append("Type mismatch: expected '" + gt_type + "', got '" + ann_type + "'")

    gt_attrs = {}
    if 'attributes' in gt_data.keys():
        gt_attrs = gt_data['attributes']

    ann_attrs = {}
    if 'attributes' in ann_data.keys():
        ann_attrs = ann_data['attributes']

    if gt_attrs != ann_attrs:
        errors.append("Attributes mismatch: expected " + str(gt_attrs) + ", got " + str(ann_attrs))

    return errors

def get_missing_objects(gt_objs, matched_gt_list, gt_best_ious):
    missed = []
    for gt_id in gt_objs.keys():
        is_matched = False
        for matched_id in matched_gt_list:
            if gt_id == matched_id:
                is_matched = True

        if is_matched == False:
            gt_data = gt_objs[gt_id]
            best_iou = gt_best_ious[gt_id]

            reason = ""
            if best_iou == 0.0:
                reason = "Not annotated at all (IoU is 0.0)"
            else:
                reason = "Annotated badly, IoU too low (" + str(round(best_iou, 4)) + " < 0.5)"

            missed_obj = {
                "gt_id": gt_id,
                "expected_type": gt_data.get('type', ''),
                "max_iou_found": round(best_iou, 4),
                "reason": reason
            }
            missed.append(missed_obj)
            
    return missed

def get_extra_objects(ann_objs, matched_ann_list, ann_best_ious):
    extra = []
    for ann_id in ann_objs.keys():
        is_matched = False
        for matched_id in matched_ann_list:
            if ann_id == matched_id:
                is_matched = True

        if is_matched == False:
            ann_data = ann_objs[ann_id]
            best_iou = ann_best_ious[ann_id]
            
            reason = ""
            if best_iou == 0.0:
                reason = "Hallucinated object (Drawn on empty space, IoU is 0.0)"
            else:
                reason = "Poor localization (IoU " + str(round(best_iou, 4)) + " < 0.5)"

            extra_obj = {
                "ann_id": ann_id,
                "drawn_type": ann_data.get('type', ''),
                "max_iou_found": round(best_iou, 4),
                "reason": reason
            }
            extra.append(extra_obj)
            
    return extra

def run_qa_analysis():
    print("Starting the analysis...")

    extract_archives()

    gt_raw = load_json(GT_JSON_PATH)
    ann_raw = load_json(ANN_JSON_PATH)

    gt_objs = merge_object_data(gt_raw)
    ann_objs = merge_object_data(ann_raw)

    img_width = 1000
    if 'width' in gt_raw[0].keys():
        img_width = gt_raw[0]['width']

    img_height = 667
    if 'height' in gt_raw[0].keys():
        img_height = gt_raw[0]['height']

    report = {
        "status": "PASSED",
        "summary": {
            "total_gt_objects": len(gt_objs.keys()),
            "total_ann_objects": len(ann_objs.keys()),
            "matched_objects": 0
        },
        "global_errors": [],
        "false_negatives_missed": [],
        "false_positives_extra": [],
        "matching_errors": []
    }

    report["global_errors"] = check_global_attributes(gt_raw, ann_raw)

    matched_gt_list = []
    matched_ann_list = []
    
    gt_best_ious = {}
    ann_best_ious = {}
    for ann_id in ann_objs.keys():
        ann_best_ious[ann_id] = 0.0

    for gt_id in gt_objs.keys():
        gt_data = gt_objs[gt_id]
        gt_mask = get_gt_mask(gt_data, GT_MASKS_DIR, img_width, img_height)

        best_ann_id, best_iou = find_best_match(gt_mask, ann_objs, matched_ann_list, img_width, img_height, ann_best_ious)
        
        gt_best_ious[gt_id] = best_iou

        if best_iou >= IOU_THRESHOLD:
            matched_gt_list.append(gt_id)
            matched_ann_list.append(best_ann_id)
            report["summary"]["matched_objects"] = report["summary"]["matched_objects"] + 1

            ann_matched_data = ann_objs[best_ann_id]
            match_errors = check_match_errors(gt_data, ann_matched_data)

            if len(match_errors) > 0:
                error_data = {
                    "gt_id": gt_id,
                    "ann_id": best_ann_id,
                    "iou": round(best_iou, 4),
                    "errors": match_errors
                }
                report["matching_errors"].append(error_data)

    report["false_negatives_missed"] = get_missing_objects(gt_objs, matched_gt_list, gt_best_ious)
    report["false_positives_extra"] = get_extra_objects(ann_objs, matched_ann_list, ann_best_ious)

    if (len(report["global_errors"]) > 0 or 
        len(report["false_negatives_missed"]) > 0 or 
        len(report["false_positives_extra"]) > 0 or 
        len(report["matching_errors"]) > 0):
        report["status"] = "FAILED"

    f = open(REPORT_PATH, 'w', encoding='utf-8')
    f.write(json.dumps(report, indent=4, ensure_ascii=False))
    f.close()

    print("The report saved to file", REPORT_PATH)

if __name__ == "__main__":
    run_qa_analysis()