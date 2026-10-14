# Video Classifier — Design and Deployment

## Goal
Predict the dominant video format so an indexing pipeline can select a suitable chunking strategy. The classifier routes videos; it does not itself create the final chunks.

## Taxonomy and routing
| Class | Chunking strategy |
|---|---|
| gameplay | `state_change_and_event` |
| egocentric | `activity_and_temporal` |
| podcast | `speech_turn_and_topic` |
| wildlife | `scene_and_activity` |
| sports | `event_and_play` |
| news_lecture | `topic_and_slide` |
| music_performance | `music_segment_and_performance` |
| film_animation | `scene_and_shot` |
| surveillance | `activity_and_time_window` |
| general | `adaptive_scene_and_semantic` |

A mixed video receives one dominant label in this baseline. Window-level routing can be added later.

## Architecture
1. Uniformly sample eight frames per video.
2. Extract visual embeddings with pretrained MobileNetV3-Small.
3. Aggregate frame embeddings using their mean and standard deviation.
4. Add inexpensive frame-difference, coarse scene-cut, duration, FPS, resolution, aspect-ratio and sampling-coverage features.
5. Standardize the fixed-length feature vector and train class-balanced Logistic Regression.
6. Return predicted class, confidence and chunking strategy. Low-confidence predictions route to `general` and are marked for review.

The backbone is frozen, so only the lightweight classifier is fitted. The first run downloads pretrained torchvision weights.

## Feature engineering
- **Visual:** MobileNetV3-Small embeddings capture scene, object and texture cues. Mean represents typical content; standard deviation captures visual variation.
- **Temporal:** normalized mean absolute differences between sampled grayscale frames estimate visual change. Their mean, standard deviation and thresholded cut rate are cheap proxies, not full motion analysis.
- **Metadata:** log duration, FPS, width and height, plus aspect ratio. Metadata should supplement rather than replace visual evidence.
- **Data hygiene:** use labels that represent dominant video type. Split by creator/source where possible to prevent near-duplicate leakage. Report macro-F1, per-class precision/recall, confusion matrix, latency and model size. Accuracy alone is insufficient for imbalanced classes.

## Dataset format
Create a CSV with `video_path,label`, using real paths and manually verified labels. Use varied sources, lengths, resolutions and recording conditions. Start with several dozen to hundreds of examples per class where possible. No actual dataset or performance result is bundled with this starter.

## Local commands
```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/main.py --help
python src/main.py train --csv src/labels.csv --model artifacts/video_classifier.joblib
python src/main.py predict --model artifacts/video_classifier.joblib data/unseen_video.mp4 --output predictions.json
```
The unseen video should not appear in the training manifest. The CSV paths are resolved relative to the current working directory.

## Deployment plan
**Offline:** store labelled data in object storage; cache extracted features by video checksum and feature version; train and validate; version model, label map, feature schema and chunking policy together.

**Online:** validate uploads, store video in object storage, enqueue a job (Redis/SQS), then let a bounded CPU worker sample frames and run inference. Return class, confidence, chunking strategy and model version to the indexing service. Use `general` for low confidence or decode failures. Cache by content hash and apply timeouts/retries.

**Scale and monitoring:** start with CPU workers and batch frame inference where useful. Consider ONNX/quantization only after validating parity. Monitor p50/p95 latency, decode failures, confidence drift, class distribution and downstream chunking quality.

## Limitations
Visual-only features can confuse related formats such as podcasts and lectures or gameplay and sports. Audio speech/music cues, OCR and ASR can improve routing at added cost. The confidence threshold is a configurable heuristic and should be calibrated on validation data. Do not claim performance until measured on a held-out dataset.
