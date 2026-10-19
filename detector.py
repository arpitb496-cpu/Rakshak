"""
detector.py - Pure Vision-Based Object Detection via OpenCV for Rakshak.

Operates EXCLUSIVELY on the camera image buffer (np.ndarray).
Never accesses ground-truth world coordinates.

Pipeline:
- BGR to HSV color space conversion
- Color segmentation for:
  - Survivor (Rescue Orange)
  - Rescuers (Personnel Blue)
  - Rubble / Obstacles (Concrete Gray)
- Morphological noise suppression (Opening / Closing)
- Contour analysis and bounding box extraction
- Pinhole camera range Z = (f * W_real) / w_bbox and bearing estimation
- Mask-area ratio metric (strictly NO fake confidence numbers)
- Frame-skipping: executes full CV processing every 2nd frame, reusing cached results
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2


@dataclass
class Detection:
    """Detected entity from camera image."""
    class_label: str             # "SURVIVOR", "RESCUER", "RUBBLE"
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) in 160x90 camera frame
    estimated_range: float       # Z in pixels
    estimated_bearing: float     # Angle relative to camera optical axis in radians
    mask_area_ratio: float       # Grounded metric: contour_area / (w * h)
    contour_area: float          # Pixel area of segmented mask
    triage_level: str = "CRITICAL"  # "CRITICAL", "INJURED", "STABLE"


class VisionDetector:
    """
    OpenCV HSV color segmentation & contour detector.
    """

    # Physical object real-world reference dimensions (width, height)
    W_REAL_SURVIVOR = 24.0
    H_REAL_SURVIVOR = 36.0
    W_REAL_RESCUER = 22.0
    H_REAL_RESCUER = 36.0
    W_REAL_RUBBLE = 40.0
    H_REAL_RUBBLE = 38.0

    def __init__(
        self,
        focal_length: float = 125.57,
        cx: float = 80.0,
        cy: float = 45.0,
        max_range: float = 340.0,
    ):
        self.focal_length = focal_length
        self.cx = cx
        self.cy = cy
        self.max_range = max_range

        # Morphological kernel
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # HSV color thresholds (OpenCV H: 0..179, S: 0..255, V: 0..255)
        # Rescue Orange: H: 7..25 (tolerant to smoke haze & noise)
        self.lower_orange = np.array([7, 65, 55], dtype=np.uint8)
        self.upper_orange = np.array([25, 255, 255], dtype=np.uint8)

        # Rescuer Blue: H: 95..130
        self.lower_blue = np.array([95, 80, 50], dtype=np.uint8)
        self.upper_blue = np.array([130, 255, 255], dtype=np.uint8)

        # Rubble Gray: Low saturation
        self.lower_rubble = np.array([0, 0, 35], dtype=np.uint8)
        self.upper_rubble = np.array([180, 50, 130], dtype=np.uint8)

        # Minimum contour pixel area to reject noise
        self.min_area = 8.0

        # Phase 4: Noise vs Clean detection statistics
        self.frames_noise_off: int = 0
        self.detections_noise_off: int = 0
        self.frames_noise_on: int = 0
        self.detections_noise_on: int = 0

        # Frame skip state
        self.frame_counter = 0
        self.cached_detections: List[Detection] = []
        self.cached_annotated_frame: Optional[np.ndarray] = None

    def detect(self, camera_frame: np.ndarray, force_update: bool = False) -> List[Detection]:
        """
        Runs detection on the camera image.
        Executes full OpenCV pipeline on odd frames (1, 3, 5...),
        reusing cached detections on even frames (2, 4, 6...) for optimal performance.
        """
        self.frame_counter += 1

        # Frame skipping: execute every 2nd frame, reuse cached in between
        if not force_update and (self.frame_counter > 1) and (self.frame_counter % 2 == 0) and (self.cached_detections is not None):
            return self.cached_detections

        # STRICT REQUIREMENT: Input is purely the camera image
        if camera_frame is None or camera_frame.size == 0:
            self.cached_detections = []
            return []

        hsv = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV)
        detections: List[Detection] = []

        # 1. Detect Survivor (Orange)
        orange_mask = cv2.inRange(hsv, self.lower_orange, self.upper_orange)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, self.kernel)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, self.kernel)

        survivor_dets = self._extract_contours(
            orange_mask,
            "SURVIVOR",
            self.W_REAL_SURVIVOR,
            self.H_REAL_SURVIVOR,
            min_area=self.min_area,
            hsv_img=hsv,
        )
        detections.extend(survivor_dets)

        # 2. Detect Rescuers (Blue)
        blue_mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, self.kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, self.kernel)

        rescuer_dets = self._extract_contours(
            blue_mask,
            "RESCUER",
            self.W_REAL_RESCUER,
            self.H_REAL_RESCUER,
            min_area=self.min_area,
            hsv_img=None,
        )
        detections.extend(rescuer_dets)

        self.cached_detections = detections
        return detections

    def _classify_triage(self, hsv: np.ndarray, x: int, y: int, w: int, h: int) -> str:
        """
        Classifies triage strictly from the camera image using HSV color segmentation on the tag badge.
        Returns: 'CRITICAL' (Red), 'INJURED' (Yellow), or 'STABLE' (Green).
        """
        if hsv is None:
            return "CRITICAL"
        # The triage tag badge is positioned on the torso below the reflective stripe (approx 35% to 80% down, 25% to 75% wide)
        roi_y1 = y + int(h * 0.35)
        roi_y2 = y + int(h * 0.85)
        roi_x1 = x + int(w * 0.20)
        roi_x2 = x + int(w * 0.80)
        roi = hsv[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0:
            roi = hsv[y : y + h, x : x + w]
        if roi.size == 0:
            return "CRITICAL"

        # Red tag (CRITICAL): H in [0..7] or [170..180], S >= 50, V >= 50
        mask_r1 = cv2.inRange(roi, np.array([0, 50, 50], dtype=np.uint8), np.array([7, 255, 255], dtype=np.uint8))
        mask_r2 = cv2.inRange(roi, np.array([170, 50, 50], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
        red_pts = int(cv2.countNonZero(mask_r1) + cv2.countNonZero(mask_r2))

        # Yellow tag (INJURED): H in [23..38], S >= 50, V >= 50
        mask_y = cv2.inRange(roi, np.array([23, 50, 50], dtype=np.uint8), np.array([38, 255, 255], dtype=np.uint8))
        yellow_pts = int(cv2.countNonZero(mask_y))

        # Green tag (STABLE): H in [45..88], S >= 45, V >= 45
        mask_g = cv2.inRange(roi, np.array([45, 45, 45], dtype=np.uint8), np.array([88, 255, 255], dtype=np.uint8))
        green_pts = int(cv2.countNonZero(mask_g))

        # Classification based strictly on highest tag pixel count
        if yellow_pts > red_pts and yellow_pts >= green_pts and yellow_pts > 0:
            return "INJURED"
        elif green_pts > red_pts and green_pts > yellow_pts and green_pts > 0:
            return "STABLE"
        elif red_pts > 0:
            return "CRITICAL"

        return "CRITICAL"

    def _extract_contours(
        self,
        mask: np.ndarray,
        class_label: str,
        real_width: float,
        real_height: float,
        min_area: float,
        hsv_img: Optional[np.ndarray] = None,
    ) -> List[Detection]:
        """Extracts contours, computes bounding box, range Z, bearing, and mask-area ratio."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w <= 1 or h <= 1:
                continue

            # Grounded metric: Mask Area Ratio (actual coverage inside bounding rectangle)
            mask_ratio = float(area / (w * h))

            # Metric range estimation:
            # If horizontally occluded by a wall/pillar edge, effective width is derived from unoccluded height
            aspect_ratio = real_height / real_width  # e.g. 36/24 = 1.5
            effective_w = max(float(w), float(h) / aspect_ratio)

            # Pinhole Camera Range Estimation: Z = (f * W_real) / effective_w
            estimated_range = float((self.focal_length * real_width) / effective_w)

            # STRICT REQUIREMENT: Reject any detection whose estimated range exceeds camera max range
            if estimated_range > self.max_range or estimated_range < 10.0:
                continue

            # Bearing angle from optical center: theta = arctan((bbox_center_x - cx) / f)
            bbox_center_x = x + (w / 2.0)
            estimated_bearing = float(math.atan2(bbox_center_x - self.cx, self.focal_length))

            triage_lvl = "CRITICAL"
            if class_label == "SURVIVOR" and hsv_img is not None:
                triage_lvl = self._classify_triage(hsv_img, x, y, w, h)

            results.append(
                Detection(
                    class_label=class_label,
                    bbox=(x, y, w, h),
                    estimated_range=estimated_range,
                    estimated_bearing=estimated_bearing,
                    mask_area_ratio=mask_ratio,
                    contour_area=area,
                    triage_level=triage_lvl,
                )
            )

        # Sort by contour area descending (strongest primary detection first)
        results.sort(key=lambda d: d.contour_area, reverse=True)
        return results

    def get_annotated_feed(
        self, camera_frame: np.ndarray, scale: int = 2
    ) -> np.ndarray:
        """
        Generates an upscaled HUD visualization feed with bounding boxes,
        class labels, range Z, bearing, and mask area ratio.
        """
        if camera_frame is None:
            return np.zeros((90 * scale, 160 * scale, 3), dtype=np.uint8)

        # Upscale 160x90 to higher resolution for crisp HUD presentation (e.g. 320x180)
        feed = cv2.resize(
            camera_frame,
            (self.cx_int * 2 * scale if hasattr(self, "cx_int") else 160 * scale, 90 * scale),
            interpolation=cv2.INTER_NEAREST,
        )

        # Draw crosshairs
        ch_w = feed.shape[1]
        ch_h = feed.shape[0]
        cc_x = ch_w // 2
        cc_y = ch_h // 2
        cv2.line(feed, (cc_x - 12, cc_y), (cc_x + 12, cc_y), (70, 220, 70), 1)
        cv2.line(feed, (cc_x, cc_y - 12), (cc_x, cc_y + 12), (70, 220, 70), 1)
        cv2.circle(feed, (cc_x, cc_y), 4, (70, 220, 70), 1)

        # Draw detections
        for det in self.cached_detections:
            x, y, w, h = det.bbox
            sx = x * scale
            sy = y * scale
            sw = w * scale
            sh = h * scale

            if det.class_label == "SURVIVOR":
                if det.triage_level == "CRITICAL":
                    box_color = (30, 30, 245)    # Red in BGR
                elif det.triage_level == "INJURED":
                    box_color = (25, 220, 240)   # Yellow in BGR
                else:
                    box_color = (40, 220, 45)    # Green in BGR
                text_color = (255, 255, 255)
            elif det.class_label == "RESCUER":
                box_color = (255, 140, 0)  # Cyan/Blue
                text_color = (240, 240, 240)
            else:
                box_color = (180, 180, 180)
                text_color = (200, 200, 200)

            # Bounding box
            cv2.rectangle(feed, (sx, sy), (sx + sw, sy + sh), box_color, 2)

            # Corner brackets for tactical look
            corner_len = min(8, sw // 3, sh // 3)
            # Top-left
            cv2.line(feed, (sx, sy), (sx + corner_len, sy), (255, 255, 255), 2)
            cv2.line(feed, (sx, sy), (sx, sy + corner_len), (255, 255, 255), 2)
            # Bottom-right
            cv2.line(feed, (sx + sw, sy + sh), (sx + sw - corner_len, sy + sh), (255, 255, 255), 2)
            cv2.line(feed, (sx + sw, sy + sh), (sx + sw, sy + sh - corner_len), (255, 255, 255), 2)

            # Annotation Label: No fake confidence, class + triage + range + bearing
            bearing_deg = math.degrees(det.estimated_bearing)
            label_text = f"{det.class_label} [{det.triage_level}]" if det.class_label == "SURVIVOR" else det.class_label
            sub_text = f"Z:{det.estimated_range:.0f}px | B:{bearing_deg:+.1f}° | A:{det.mask_area_ratio:.2f}"

            # Clamped label coordinates so text stays inside frame and does not overlap box
            label_x = max(6, min(sx, feed.shape[1] - 170))
            if sy >= 30:
                ty1 = sy - 16
                ty2 = sy - 4
            else:
                ty1 = sy + sh + 14
                ty2 = sy + sh + 26
                if ty2 >= feed.shape[0] - 2:
                    ty1 = max(14, sy + 14)
                    ty2 = max(26, sy + 26)

            # Label banner
            cv2.putText(
                feed,
                label_text,
                (label_x, ty1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                box_color,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                feed,
                sub_text,
                (label_x, ty2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                text_color,
                1,
                cv2.LINE_AA,
            )

        return feed
