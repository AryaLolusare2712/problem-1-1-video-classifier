# Video Classifier: Design, Feature Engineering and Deployment

## 1. Objective

Predict a **video type that determines how the downstream index should chunk the video**. This is not intended to classify every action or produce captions. It routes each video to a suitable chunking policy.

## 2. Taxonomy and chunking policy

| Class | Typical cues | Recommended chunking |
|---|---|---|
| `gameplay` | game UI, screen capture, HUD, rapid state changes | Split at scene/state changes and meaningful events; retain short context overlap |
| `egocentric` | first-person camera, hands, continuous real-world movement | Activity boundaries and temporal windows; preserve action continuity |
| `podcast` | one or more people speaking, studio/interview framing | Speech turns and topic boundaries; align to transcript timestamps |
| `wildlife` | animals in natural environments | Scene plus animal/activity changes; longer windows when activity is continuous |
| `sports` | field/court, athletes, play and scoreboard | Plays, possession changes, goals/points and replays |
| `news_lecture` | presenter, slides, classroom/news set | Topic, slide and speaker transitions |
| `music_performance` | instruments, stage, musical performance | Musical sections and performance changes |
| `film_animation` | cinematic shots, animation, narrative scenes | Shot/scene boundaries with narrative context |
| `surveillance` | fixed camera, repeated view, monitoring | Time windows and detected activity intervals |
| `general` | unknown, mixed, or low-confidence content | Adaptive scene boundaries plus semantic overlap |

A video may contain multiple types. This first version assigns one dominant routing label to the whole video. A future extension can classify short windows independently and switch chunkers within a video.

## 3. Lightweight architecture

1. Decode a small number of uniformly sampled frames (default: 8), not every frame.
2. Run each frame through pretrained **MobileNetV3-Small** and global-average-pool its feature map.
3. Aggregate frame embeddings using their mean and standard deviation. The mean represents overall visual content; the standard deviation captures variation over time.
4. Append inexpensive temporal and metadata features: average frame difference, variability of frame difference, coarse scene-cut rate, duration, frame rate, dimensions, aspect ratio and sample coverage.
5. Standardize features and train a balanced, regularized multinomial Logistic Regression classifier.
6. Return class, confidence and the corresponding chunking strategy. Low-confidence predictions are routed to `general` and marked for review.

The CNN backbone is pretrained and frozen, so only the small classifier is fitted on the target corpus. This reduces training cost and avoids training a large video transformer from scratch. The first run downloads the torchvision MobileNetV3-Small pretrained weights.

## 4. Feature engineering

### Visual embedding
For each sampled frame, MobileNetV3-Small produces a compact representation of objects, textures and scene appearance. Use `mean(embeddings)` and `std(embeddings)` across frames instead of concatenating all frame embeddings. This gives a fixed-size vector regardless of video duration.

### Temporal features
Resize frames to 64×64 grayscale and calculate normalized mean absolute differences between consecutive sampled frames. Their mean approximates visual change; standard deviation measures irregularity; the fraction above a threshold estimates abrupt cuts. These are low-cost signals, not a replacement for optical flow or a learned temporal model.

### Metadata
Use log-scaled duration, frame rate, width and height, plus aspect ratio. Metadata helps identify screen captures, long fixed-camera footage and common format patterns, but should never dominate visual evidence.

### Data and leakage prevention
Create a labelled manifest with `video_path,label`. Labels should describe the dominant content format, not the chunking output. Split by source video/channel/creator where possible, so near-duplicate clips from the same source cannot leak between train and test. Use a held-out set and report macro-F1, per-class precision/recall, confusion matrix, latency and model size. Accuracy alone can hide rare-class failures.

## 5. Training data

No labelled corpus was included in the repository statement visible to this implementation. Supply a CSV with columns `video_path,label` and labelled examples from the provided corpus. Include diverse lengths, resolutions, camera styles, creators and class balance. Start with a few hundred labelled examples per class if available, then use active learning to label low-confidence and misclassified cases. Do not report performance until measured on a held-out dataset.

Example:
```csv
video_path,label
data/gameplay_001.mp4,gameplay
data/podcast_001.mp4,podcast
data/wildlife_001.mp4,wildlife
data/sports_001.mp4,sports
```

## 6. Run locally

```bash
pip install -r requirements.txt
python src/main.py --help
python src/main.py train --csv src/labels.csv --model artifacts/video_classifier.joblib
python src/main.py predict --model artifacts/video_classifier.joblib --threshold 0.45 data/incoming.mp4
```

The CSV paths must be accessible from the working directory. OpenCV must be able to decode the video formats. The first run may download pretrained weights from torchvision.

## 7. Deployment plan

**Offline training**
- Store labelled manifest and source videos in object storage or a versioned dataset.
- Run feature extraction in a batch job; cache embeddings by video checksum and extractor version.
- Train Logistic Regression, evaluate per-class metrics and calibrate the confidence threshold on validation data.
- Version the classifier, feature schema, label mapping and chunking policy together.

**Online inference**
- Upload/API gateway validates file type and size and stores the video in object storage.
- A queue (e.g. Redis/SQS) sends a job to CPU inference workers. Decode only 8 frames and batch frames across videos when throughput permits.
- Worker returns `{class, confidence, chunking_strategy, model_version}` to the indexing pipeline.
- If confidence is below the calibrated threshold, route to `general` or human review.
- Cache by content hash; use bounded worker concurrency, timeouts and retries. Keep video decoding separate from model serving.

**Scaling**
- Use CPU workers initially; use GPU only if measured throughput justifies it.
- Export/quantize the backbone to ONNX or a supported inference runtime only after validating accuracy parity.
- Monitor latency percentiles, decode failures, class distribution, confidence drift and chunking quality. Retrain when new sources or new video types shift the distribution.

## 8. Limitations and next steps

- A single label can oversimplify mixed-format videos.
- Visual-only features may confuse podcast vs lecture or sports vs gameplay. Add lightweight audio speech/music features and OCR/ASR-derived signals if the corpus and runtime budget justify them.
- Logistic Regression probabilities may be miscalibrated; calibrate with held-out data before treating 0.45 as a production threshold.
- The frame-difference scene-cut signal is coarse.
- No measured performance is claimed until training and evaluation are run on the actual corpus.
