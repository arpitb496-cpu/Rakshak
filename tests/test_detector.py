"""
tests/test_detector.py - Unit tests for OpenCV HSV Segmentation and Pinhole Estimation.
"""

import math
import numpy as np
import cv2
import pytest
from detector import VisionDetector


def test_detector_survivor_pinhole_range_and_bearing():
    detector = VisionDetector(focal_length=125.57, cx=80.0, cy=45.0)

    # Create a synthetic 160x90 BGR camera image with neutral gray background
    frame = np.full((90, 160, 3), 50, dtype=np.uint8)

    # Draw an orange survivor box at center x=80, y=45
    # Width = 25 px, Height = 36 px
    # Color in BGR: (15, 120, 255)
    w_box = 25
    h_box = 36
    x_box = int(80 - w_box // 2)
    y_box = int(45 - h_box // 2)
    frame[y_box : y_box + h_box, x_box : x_box + w_box] = [15, 120, 255]

    detections = detector.detect(frame, force_update=True)
    assert len(detections) >= 1

    surv = next((d for d in detections if d.class_label == "SURVIVOR"), None)
    assert surv is not None

    # Expected Range: Z = f * W_real / w = 125.57 * 24.0 / 25 ~ 120.5 px
    expected_range = (125.57 * 24.0) / float(w_box)
    assert abs(surv.estimated_range - expected_range) < 5.0

    # Since it is centered at cx=80, bearing should be ~0.0 radians
    assert abs(surv.estimated_bearing) < 0.05

    # Check mask area ratio (solid box should have high ratio > 0.8)
    assert surv.mask_area_ratio > 0.8


def test_detector_rescuer_detection():
    detector = VisionDetector(focal_length=125.57, cx=80.0, cy=45.0)
    frame = np.full((90, 160, 3), 50, dtype=np.uint8)

    # Blue rescuer box in BGR: (235, 120, 30)
    w_box = 20
    h_box = 30
    x_box = 40
    y_box = 30
    frame[y_box : y_box + h_box, x_box : x_box + w_box] = [235, 120, 30]

    detections = detector.detect(frame, force_update=True)
    rescuer = next((d for d in detections if d.class_label == "RESCUER"), None)
    assert rescuer is not None
    # Off-center to left (x=40), bearing should be negative
    assert rescuer.estimated_bearing < -0.1


def test_detector_frame_skipping():
    detector = VisionDetector(focal_length=125.57, cx=80.0, cy=45.0)
    frame = np.full((90, 160, 3), 50, dtype=np.uint8)
    frame[30:60, 70:90] = [15, 120, 255]

    # Frame 1: odd -> runs or sets up
    d1 = detector.detect(frame)
    # Frame 2: even -> recomputes
    d2 = detector.detect(frame)
    # Frame 3: odd -> reuses cached results
    d3 = detector.detect(frame)
    assert d2 is d3 or d2 == d3


def test_detector_triage_classification():
    detector = VisionDetector(focal_length=125.57, cx=80.0, cy=45.0)

    # 1. Test CRITICAL (Red tag)
    frame_crit = np.full((90, 160, 3), 50, dtype=np.uint8)
    frame_crit[25:65, 65:95] = [15, 120, 255]  # Orange body
    frame_crit[40:55, 75:85] = [20, 20, 235]   # Red tag in BGR
    dets = detector.detect(frame_crit, force_update=True)
    surv_crit = next((d for d in dets if d.class_label == "SURVIVOR"), None)
    assert surv_crit is not None
    assert surv_crit.triage_level == "CRITICAL"

    # 2. Test INJURED (Yellow tag)
    frame_inj = np.full((90, 160, 3), 50, dtype=np.uint8)
    frame_inj[25:65, 65:95] = [15, 120, 255]   # Orange body
    frame_inj[40:55, 75:85] = [20, 230, 240]   # Yellow tag in BGR
    dets = detector.detect(frame_inj, force_update=True)
    surv_inj = next((d for d in dets if d.class_label == "SURVIVOR"), None)
    assert surv_inj is not None
    assert surv_inj.triage_level == "INJURED"

    # 3. Test STABLE (Green tag)
    frame_stbl = np.full((90, 160, 3), 50, dtype=np.uint8)
    frame_stbl[25:65, 65:95] = [15, 120, 255]  # Orange body
    frame_stbl[40:55, 75:85] = [30, 220, 40]   # Green tag in BGR
    dets = detector.detect(frame_stbl, force_update=True)
    surv_stbl = next((d for d in dets if d.class_label == "SURVIVOR"), None)
    assert surv_stbl is not None
    assert surv_stbl.triage_level == "STABLE"
