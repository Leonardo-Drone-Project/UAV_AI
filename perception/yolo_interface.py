from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union

import cv2
import numpy as np


@dataclass
class DetectionResult:
    target_detected: bool
    target_conf: float
    class_id: int
    class_name: str
    bbox_xyxy: Tuple[int, int, int, int]
    bbox_cxcywh: Tuple[int, int, int, int]
    target_offset_px: Tuple[int, int]
    image_size: Tuple[int, int]
    timestamp: float


class YoloDetector:
    """
    Ultralytics YOLO wrapper.
    No camera logic. Frame in, structured result out.
    """

    def __init__(
        self,
        weights_path: Union[str, Path],
        device: str = "0",
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        prefer_classes: Optional[List[str]] = None,
        min_box_px: int = 20,
    ):
        self.weights_path = str(Path(weights_path).expanduser().resolve())
        self.device = device
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.prefer_classes = prefer_classes or []
        self.min_box_px = int(min_box_px)

        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError(
                "ultralytics is not installed.\n"
                "Install with:\n"
                "  python -m pip install ultralytics\n"
            ) from e

        self.model = YOLO(self.weights_path)

        names = getattr(self.model.model, "names", None)
        if isinstance(names, dict):
            self.names = names
        elif isinstance(names, list):
            self.names = {i: n for i, n in enumerate(names)}
        else:
            self.names = {}

    def infer_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp: Optional[float] = None,
        save_dir: Optional[Union[str, Path]] = None,
        save_annotated: bool = False,
        save_json: bool = False,
    ) -> Tuple[DetectionResult, List[Dict[str, Any]]]:
        if timestamp is None:
            timestamp = time.time()

        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
            raise ValueError("infer_frame expects a valid np.ndarray (BGR frame).")

        h, w = frame_bgr.shape[:2]
        cx_img, cy_img = w // 2, h // 2

        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            verbose=False,
        )

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            out = DetectionResult(
                target_detected=False,
                target_conf=0.0,
                class_id=-1,
                class_name="",
                bbox_xyxy=(0, 0, 0, 0),
                bbox_cxcywh=(0, 0, 0, 0),
                target_offset_px=(0, 0),
                image_size=(w, h),
                timestamp=float(timestamp),
            )
            dets: List[Dict[str, Any]] = []
            if save_dir is not None:
                save_dir = Path(save_dir).expanduser().resolve()
                save_dir.mkdir(parents=True, exist_ok=True)
                if save_json:
                    self._write_json(save_dir / "latest_detection.json", out, extra={"all_detections": dets})
                if save_annotated:
                    cv2.imwrite(str(save_dir / "annotated.jpg"), frame_bgr)
            return out, dets

        boxes = results[0].boxes

        dets: List[Dict[str, Any]] = []
        for b in boxes:
            xyxy = b.xyxy[0].cpu().numpy().astype(float)
            conf = float(b.conf[0].cpu().numpy())
            cls_id = int(b.cls[0].cpu().numpy())
            name = self.names.get(cls_id, str(cls_id))

            if conf < self.conf_thres:
                continue

            x1, y1, x2, y2 = xyxy
            bw = int(x2 - x1)
            bh = int(y2 - y1)

            if bw < self.min_box_px or bh < self.min_box_px:
                continue

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            dets.append(
                {
                    "class_id": cls_id,
                    "class_name": name,
                    "conf": conf,
                    "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                    "bbox_cxcywh": [cx, cy, bw, bh],
                    "offset_px": [cx - cx_img, cy - cy_img],
                }
            )

        if len(dets) == 0:
            out = DetectionResult(
                target_detected=False,
                target_conf=0.0,
                class_id=-1,
                class_name="",
                bbox_xyxy=(0, 0, 0, 0),
                bbox_cxcywh=(0, 0, 0, 0),
                target_offset_px=(0, 0),
                image_size=(w, h),
                timestamp=float(timestamp),
            )
            if save_dir is not None:
                save_dir = Path(save_dir).expanduser().resolve()
                save_dir.mkdir(parents=True, exist_ok=True)
                if save_json:
                    self._write_json(save_dir / "latest_detection.json", out, extra={"all_detections": dets})
                if save_annotated:
                    cv2.imwrite(str(save_dir / "annotated.jpg"), frame_bgr)
            return out, dets

        preferred = [d for d in dets if d["class_name"] in self.prefer_classes] if self.prefer_classes else []
        chosen = max(preferred, key=lambda d: d["conf"]) if preferred else max(dets, key=lambda d: d["conf"])

        x1, y1, x2, y2 = chosen["bbox_xyxy"]
        cx, cy, bw, bh = chosen["bbox_cxcywh"]
        dx, dy = chosen["offset_px"]

        out = DetectionResult(
            target_detected=True,
            target_conf=float(chosen["conf"]),
            class_id=int(chosen["class_id"]),
            class_name=str(chosen["class_name"]),
            bbox_xyxy=(x1, y1, x2, y2),
            bbox_cxcywh=(cx, cy, bw, bh),
            target_offset_px=(dx, dy),
            image_size=(w, h),
            timestamp=float(timestamp),
        )

        if save_dir is not None:
            save_dir = Path(save_dir).expanduser().resolve()
            save_dir.mkdir(parents=True, exist_ok=True)

            if save_json:
                self._write_json(save_dir / "latest_detection.json", out, extra={"all_detections": dets})

            if save_annotated:
                annotated = frame_bgr.copy()
                cv2.circle(annotated, (cx_img, cy_img), 6, (0, 255, 255), -1)
                for d in dets:
                    x1_, y1_, x2_, y2_ = d["bbox_xyxy"]
                    cv2.rectangle(annotated, (x1_, y1_), (x2_, y2_), (255, 255, 0), 1)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label = f"{out.class_name} {out.target_conf:.2f} dx={dx} dy={dy}"
                cv2.putText(
                    annotated,
                    label,
                    (max(0, x1), max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.imwrite(str(save_dir / "annotated.jpg"), annotated)

        return out, dets

    @staticmethod
    def _write_json(path: Path, out: DetectionResult, extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {"result": asdict(out)}
        if extra:
            payload.update(extra)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


_DETECTOR: Optional[YoloDetector] = None


def init_detector(
    weights_path: Union[str, Path],
    device: str = "0",
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    prefer_classes: Optional[List[str]] = None,
    min_box_px: int = 20,
) -> None:
    global _DETECTOR
    _DETECTOR = YoloDetector(
        weights_path=weights_path,
        device=device,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        prefer_classes=prefer_classes,
        min_box_px=min_box_px,
    )


def detect(
    frame: np.ndarray,
    *,
    timestamp: Optional[float] = None,
    save_dir: Optional[Union[str, Path]] = None,
    save_annotated: bool = False,
    save_json: bool = False,
) -> Dict[str, Any]:
    if _DETECTOR is None:
        raise RuntimeError("YOLO detector not initialised. Call init_detector(...) first.")

    best, _all = _DETECTOR.infer_frame(
        frame_bgr=frame,
        timestamp=timestamp,
        save_dir=save_dir,
        save_annotated=save_annotated,
        save_json=save_json,
    )

    img_w, img_h = best.image_size
    dx_px, dy_px = best.target_offset_px

    dx_norm = 0.0
    dy_norm = 0.0
    if img_w > 0 and img_h > 0:
        dx_norm = float(dx_px) / (float(img_w) / 2.0)
        dy_norm = float(dy_px) / (float(img_h) / 2.0)
        dx_norm = max(-1.0, min(1.0, dx_norm))
        dy_norm = max(-1.0, min(1.0, dy_norm))

    return {
        "detected": bool(best.target_detected),
        "class": best.class_name if best.target_detected else None,
        "confidence": float(best.target_conf),
        "bbox": tuple(best.bbox_xyxy),
        "offset_px": (int(dx_px), int(dy_px)),
        "offset_norm": (dx_norm, dy_norm),
        "timestamp": float(best.timestamp),
        "image_size": (int(img_w), int(img_h)),
    }

