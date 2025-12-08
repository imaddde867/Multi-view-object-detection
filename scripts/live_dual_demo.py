import argparse
import os

import cv2
from ultralytics import YOLO

CLASS_NAMES = {
    0: "person",
    1: "car"
}


def parse_video_source(source: str):
    """Interpret numeric device IDs vs. file paths."""
    if source.isdigit() and not os.path.exists(source):
        return int(source)
    return source


def clip_bbox(bbox, width, height):
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def shrink_bbox(bbox, shrink_factor):
    if shrink_factor <= 0:
        return bbox
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    delta_w = int(width * shrink_factor / 2)
    delta_h = int(height * shrink_factor / 2)
    new_bbox = (x1 + delta_w, y1 + delta_h, x2 - delta_w, y2 - delta_h)
    return new_bbox if new_bbox[0] < new_bbox[2] and new_bbox[1] < new_bbox[3] else bbox


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def apply_nms(detections, iou_threshold):
    if len(detections) <= 1:
        return detections
    ordered = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept = []
    while ordered:
        current = ordered.pop(0)
        kept.append(current)
        ordered = [det for det in ordered
                   if det["class_id"] != current["class_id"]
                   or compute_iou(det["bbox"], current["bbox"]) < iou_threshold]
    return kept


def main():
    parser = argparse.ArgumentParser(description="Live dual-camera demo with YOLOv8 overlays.")
    parser.add_argument("--model", type=str, default="demo_material/yolov8m_best.pt", help="Path to YOLO weights.")
    parser.add_argument("--source1", type=str, default="0", help="First video source (int camera ID or file path).")
    parser.add_argument("--source2", type=str, default="1", help="Second video source (int camera ID or file path).")
    parser.add_argument("--imgsz", type=int, default=896, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="Model NMS IoU.")
    parser.add_argument("--box_shrink", type=float, default=0.15, help="Fraction to shrink boxes for visualization.")
    parser.add_argument("--nms_iou", type=float, default=0.6, help="Per-camera NMS IoU.")
    parser.add_argument("--device", type=str, default=None, help="Force device (cuda:0, mps, cpu).")
    parser.add_argument("--half", action="store_true", help="Use half precision when supported (CUDA only).")
    parser.add_argument("--record_out", type=str, default=None, help="Optional path to save the combined feed.")
    parser.add_argument("--window_title", type=str, default="Dual-Cam Demo", help="OpenCV window title.")
    args = parser.parse_args()

    source1 = parse_video_source(args.source1)
    source2 = parse_video_source(args.source2)

    model = YOLO(args.model)
    if args.device:
        try:
            model.to(args.device)
        except Exception as exc:  # pragma: no cover
            print(f"⚠️ Could not switch to {args.device} ({exc}). Falling back to default device.")
            args.device = None
    if args.half:
        try:
            model.model.half()
        except AttributeError:
            print("⚠️ Half precision not supported; continuing in full precision.")

    cap1 = cv2.VideoCapture(source1)
    cap2 = cv2.VideoCapture(source2)

    if not cap1.isOpened() or not cap2.isOpened():
        print("❌ Error: could not open one or both sources.")
        return

    fps = cap1.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 20
    writer = None

    print("🎬 Starting live dual-camera demo. Press 'q' to exit.")
    if args.record_out:
        print(f"💾 Recording enabled: {args.record_out}")

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            print("⚠️ End of stream or camera disconnected.")
            break

        dets1 = run_inference(model, frame1, args.imgsz, args.conf, args.iou, args.box_shrink, args.nms_iou)
        dets2 = run_inference(model, frame2, args.imgsz, args.conf, args.iou, args.box_shrink, args.nms_iou)

        overlay_detections(frame1, dets1)
        overlay_detections(frame2, dets2)

        combined = stack_frames(frame1, frame2)
        cv2.imshow(args.window_title, combined)

        if args.record_out:
            if writer is None:
                height, width = combined.shape[:2]
                writer = cv2.VideoWriter(
                    args.record_out,
                    cv2.VideoWriter_fourcc(*"XVID"),
                    fps,
                    (width, height)
                )
            writer.write(combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if writer:
        writer.release()
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    print("✅ Demo finished.")


def run_inference(model, frame, imgsz, conf, iou, box_shrink, nms_iou):
    height, width = frame.shape[:2]
    results = model(frame, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
    detections = []
    for bbox, score, cls in zip(results.boxes.xyxy.tolist(),
                                results.boxes.conf.tolist(),
                                results.boxes.cls.tolist()):
        class_id = int(cls)
        class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
        clipped = clip_bbox(bbox, width, height)
        if not clipped:
            continue
        shrunk = shrink_bbox(clipped, box_shrink)
        detections.append({
            "bbox": list(map(int, shrunk)),
            "confidence": float(score),
            "class_id": class_id,
            "class_name": class_name
        })
    return apply_nms(detections, nms_iou)


def overlay_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["class_name"]
        conf = det["confidence"]
        color = (0, 255, 0) if label == "person" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def stack_frames(frame1, frame2):
    target_h = min(frame1.shape[0], frame2.shape[0])
    resized1 = cv2.resize(frame1, (int(frame1.shape[1] * target_h / frame1.shape[0]), target_h))
    resized2 = cv2.resize(frame2, (int(frame2.shape[1] * target_h / frame2.shape[0]), target_h))
    return cv2.hconcat([resized1, resized2])


if __name__ == "__main__":
    main()
