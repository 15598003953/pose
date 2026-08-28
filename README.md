# Pose estimation from MediaPipe Pose

This model estimates 33 pose keypoints and person segmentation mask per detected person from [person detector](../person_detection_mediapipe). (The image below is referenced from [MediaPipe Pose Keypoints](https://github.com/tensorflow/tfjs-models/tree/master/pose-detection#blazepose-keypoints-used-in-mediapipe-blazepose))
 
![MediaPipe Pose Landmark](examples/pose_landmarks.png)

This model is converted from TFlite to ONNX using following tools:
- TFLite model to ONNX: https://github.com/onnx/tensorflow-onnx
- simplified by [onnx-simplifier](https://github.com/daquexian/onnx-simplifier)

**Note**:
- Visit https://github.com/google/mediapipe/blob/master/docs/solutions/models.md#pose for models of larger scale.
- `pose_estimation_mediapipe_2023mar_int8bq.onnx` represents the block-quantized version in int8 precision and is generated using [block_quantize.py](../../tools/quantize/block_quantize.py) with `block_size=64`.

## Demo

### python

Run the following commands to try the demo:
```bash
# detect on camera input
python demo.py
# detect on an image
python demo.py -i /path/to/image -v
```
### C++

Install latest OpenCV and CMake >= 3.24.0 to get started with:

```shell
# A typical and default installation path of OpenCV is /usr/local
cmake -B build -D OPENCV_INSTALLATION_PATH=/path/to/opencv/installation .
cmake --build build

# detect on camera input
./build/opencv_zoo_pose_estimation_mediapipe
# detect on an image
./build/opencv_zoo_pose_estimation_mediapipe -m=/path/to/model -i=/path/to/image -v
# get help messages
./build/opencv_zoo_pose_estimation_mediapipe -h
```

### Example outputs

![webcam demo](./example_outputs/mpposeest_demo.webp)

## PyQt video app

Run the local video analysis app:

```bash
python pose_app.py
```

The app loads the local person-detection and pose-estimation ONNX models,
detects people, estimates pose landmarks, and runs a lightweight put-luggage
clip classifier in real time. The classifier uses a 4-second sliding window of
upper-body keypoint features and overlays `旅客正在放置行李` when the model score
crosses the threshold. Events are appended to:

```text
result/<video_name>_put_luggage_events.jsonl
```

The bundled action model is:

```text
models/put_luggage_centroid_model.json
```

It is a small `nearest_centroid_numpy` baseline trained from locally labeled
video clips. It does not detect luggage as an object and should be treated as a
first deployable baseline, not a final high-accuracy production model.

### Put-luggage labeling and training workflow

1. Label fixed-length clips:

```bash
python label_video_clips.py --video data/2026-08-12_102107_553.mp4
```

Default clip settings:

```text
clip_seconds: 4
stride_seconds: 2
labels: put_luggage / not_put_luggage / skip
output: data/clip_labels/labels.csv
```

Keyboard shortcuts:

```text
p = put_luggage
n = not_put_luggage
s = skip
b = previous clip
r = replay current clip
q = save and quit
```

2. Extract upper-body keypoint features:

```bash
python extract_clip_pose_features.py --stride_frames 10
```

Outputs:

```text
data/clip_features/features.jsonl
data/clip_features/quality_report.json
```

3. Train the lightweight baseline model:

```bash
python train_clip_classifier.py
```

Outputs:

```text
data/clip_classifier_baseline/model.json
data/clip_classifier_baseline/report.json
data/clip_classifier_baseline/predictions.csv
```

Copy the trained model into `models/put_luggage_centroid_model.json` to make
`pose_app.py` load it at startup. If the JSON model is missing or invalid, the
app falls back to the older rule-based recognizer.

### MQTT event publishing

The PyQt app can publish put-luggage events to MQTT. Click `MQTT配置` in
`pose_app.py`, fill in the broker information, enable MQTT, and save. The app
stores the setting in:

```text
mqtt_config.json
```

The config is loaded automatically on the next startup. MQTT publishing is
optional; install the extra dependency only when this integration is needed:

```bash
pip install paho-mqtt
```

Published messages use JSON and include the event time, action text, action
code, ROI number, confidence, frame index, video timestamp, and source. Example:

```json
{
  "time": "2026-08-27T10:30:12.123+08:00",
  "action": "旅客正在放置行李",
  "action_code": "put_luggage",
  "roi_id": 1,
  "confidence": 0.82,
  "frame_idx": 123,
  "timestamp_ms": 4560,
  "source": "rtsp://camera/main"
}
```

When no ROI is configured, `roi_id` is `FULL`. When multiple ROIs are
configured, each ROI runs an independent action recognizer and publishes its
own ROI number.

### RTSP / RTMP input

Click `数据来源` in `pose_app.py` to configure the input source. The app supports:

```text
file: local video file
rtsp: rtsp://...
rtmp: rtmp://...
```

The source setting is stored in:

```text
source_config.json
```

On the next startup, the saved source is loaded automatically. For RTSP/RTMP,
click `Open Video` to open the configured stream. For local files, `Open Video`
still opens the file picker and saves the last selected file path.

The source dialog also includes low-latency settings:

```text
启用低延迟缓冲: asks OpenCV to keep a smaller capture buffer
推理间隔帧数: runs detection / pose / action inference every N frames
```

For smoother RTSP viewing, keep low-latency buffering enabled and increase
`推理间隔帧数` gradually, for example 3 to 5. The app still displays every read
frame, and reuses the most recent pose/action overlay on non-inference frames
to avoid visible frame skipping.

### Current baseline diagnostics

The bundled model was trained on one labeled local video with 41 usable clips:

```text
not_put_luggage: 27
put_luggage: 14
feature_count: 45
training accuracy: 0.8049
put_luggage precision: 0.6875
put_luggage recall: 0.7857
```

These numbers are training-set diagnostics on a small, overlapping dataset.
Add more independent positive and negative clips before treating the model as
reliable.

### Tests

Run the focused Python tests:

```bash
python -m unittest \
  test_label_video_clips.py \
  test_extract_clip_pose_features.py \
  test_train_clip_classifier.py \
  test_pose_model_action_recognizer.py \
  test_pose_action_recognizer.py
```

## License

All files in this directory are licensed under [Apache 2.0 License](LICENSE).

## Reference
- MediaPipe Pose: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
- MediaPipe pose model and model card: https://github.com/google/mediapipe/blob/master/docs/solutions/models.md#pose
- BlazePose TFJS: https://github.com/tensorflow/tfjs-models/tree/master/pose-detection/src/blazepose_tfjs
