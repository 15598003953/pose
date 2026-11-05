import numpy as np
import cv2 as cv
import os

class MPPose:
    def __init__(self, modelPath, confThreshold=0.5, backendId=0, targetId=0):
        self.model_path = modelPath
        self.conf_threshold = confThreshold
        self.backend_id = backendId
        self.target_id = targetId

        self.input_size = np.array([256, 256])  # wh
        # RoI will be larger so the performance will be better, but preprocess will be slower. Default to 1.
        self.PERSON_BOX_PRE_ENLARGE_FACTOR = 1
        self.PERSON_BOX_ENLARGE_FACTOR = 1.25

        self.ort_session = None
        self.ort_input_name = None
        self.ort_output_names = None
        self.allow_opencv_fallback = os.environ.get(
            "MPPOSE_ALLOW_OPENCV_DNN", "").strip() == "1"
        try:
            import onnxruntime as ort
            self.ort_session = ort.InferenceSession(
                self.model_path,
                providers=["CPUExecutionProvider"],
            )
            self.ort_input_name = self.ort_session.get_inputs()[0].name
            self.ort_output_names = [
                output.name for output in self.ort_session.get_outputs()
            ]
            input_shape = list(self.ort_session.get_inputs()[0].shape)
            output_count = len(self.ort_session.get_outputs())
            if input_shape != [1, 256, 256, 3] or output_count != 5:
                raise ValueError(
                    "not a MediaPipe Pose ONNX model. "
                    "Expected input [1, 256, 256, 3] and 5 outputs, "
                    "got input {} and {} outputs.".format(
                        input_shape, output_count)
                )
            print("[mp_pose] using ONNX Runtime for pose:", self.model_path)
        except ValueError:
            raise
        except Exception as exc:
            if not self.allow_opencv_fallback:
                raise RuntimeError(
                    "ONNX Runtime is required for stable MediaPipe Pose "
                    "inference in this environment. Install onnxruntime or "
                    "set MPPOSE_ALLOW_OPENCV_DNN=1 to force OpenCV DNN."
                ) from exc
            print("[mp_pose] ONNX Runtime unavailable, using OpenCV DNN:", exc)

        self.model = None
        if self.allow_opencv_fallback:
            self.model = cv.dnn.readNet(self.model_path)
            self.model.setPreferableBackend(self.backend_id)
            self.model.setPreferableTarget(self.target_id)
        self._bad_output_warned = False
        self._ort_failed_warned = False

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backendId, targetId):
        self.backend_id = backendId
        self.target_id = targetId
        if self.model is not None:
            self.model.setPreferableBackend(self.backend_id)
            self.model.setPreferableTarget(self.target_id)

    def _reload_model(self):
        if not self.allow_opencv_fallback:
            return
        self.model = cv.dnn.readNet(self.model_path)
        self.model.setPreferableBackend(self.backend_id)
        self.model.setPreferableTarget(self.target_id)

    @staticmethod
    def _output_shapes(output_blob):
        shapes = []
        for idx, blob in enumerate(output_blob):
            if blob is None:
                shapes.append((idx, None))
            else:
                shapes.append((idx, np.asarray(blob).shape))
        return shapes

    @staticmethod
    def _looks_like_pose_outputs(output_blob):
        sizes = sorted(int(np.asarray(blob).size) for blob in output_blob if blob is not None)
        return sizes == sorted([195, 1, 256 * 256, 64 * 64 * 39, 117])

    def _forward_pose(self, input_blob):
        if self.ort_session is not None:
            try:
                input_blob = np.ascontiguousarray(
                    input_blob.astype(np.float32, copy=False))
                output_blob = self.ort_session.run(
                    self.ort_output_names,
                    {self.ort_input_name: input_blob},
                )
                if self._looks_like_pose_outputs(output_blob):
                    return output_blob
                if not self._bad_output_warned:
                    self._bad_output_warned = True
                    print(
                        "[mp_pose] bad ONNX Runtime pose outputs. "
                        "model:{} output_names:{} shapes:{}".format(
                            self.model_path,
                            self.ort_output_names,
                            self._output_shapes(output_blob),
                        )
                    )
            except Exception as exc:
                if not self._ort_failed_warned:
                    self._ort_failed_warned = True
                    print("[mp_pose] ONNX Runtime pose forward failed:", exc)
        if not self.allow_opencv_fallback or self.model is None:
            return None

        self.model.setInput(input_blob)
        output_names = self.model.getUnconnectedOutLayersNames()
        output_blob = self.model.forward(output_names)
        if self._looks_like_pose_outputs(output_blob):
            return output_blob

        print(
            "[mp_pose] bad pose outputs, retrying with reloaded model. "
            "model:{} output_names:{} shapes:{}".format(
                self.model_path,
                output_names,
                self._output_shapes(output_blob),
            )
        )
        self._reload_model()
        self.model.setInput(input_blob)
        output_names = self.model.getUnconnectedOutLayersNames()
        output_blob = self.model.forward(output_names)
        if (not self._looks_like_pose_outputs(output_blob)) and not self._bad_output_warned:
            self._bad_output_warned = True
            print(
                "[mp_pose] bad pose outputs after reload. "
                "model:{} output_names:{} shapes:{}".format(
                    self.model_path,
                    output_names,
                    self._output_shapes(output_blob),
                )
            )
        return output_blob

    def _preprocess(self, image, person):
        '''
        Rotate input for inference.
        Parameters:
          image - input image of BGR channel order
          face_bbox - human face bounding box found in image of format [[x1, y1], [x2, y2]] (top-left and bottom-right points)
          person_landmarks - 4 landmarks (2 full body points, 2 upper body points) of shape [4, 2]
        Returns:
          rotated_person - rotated person image for inference
          rotate_person_bbox - person box of interest range
          angle - rotate angle for person
          rotation_matrix - matrix for rotation and de-rotation
          pad_bias - pad pixels of interest range
        '''
        # crop and pad image to interest range
        pad_bias = np.array([0, 0], dtype=np.int32)  # left, top
        person = np.array(person, dtype=np.float32, copy=True).reshape(-1)
        if person.size < 12 or not np.all(np.isfinite(person[:12])):
            return None
        person_keypoints = person[4: 12].reshape(-1, 2)
        mid_hip_point = person_keypoints[0]
        full_body_point = person_keypoints[1]
        # get RoI
        full_dist = np.linalg.norm(mid_hip_point - full_body_point)
        if (not np.isfinite(full_dist)) or full_dist < 2:
            return None
        full_bbox = np.array([mid_hip_point - full_dist, mid_hip_point + full_dist], np.int32)
        # enlarge to make sure full body can be cover
        center_bbox = np.sum(full_bbox, axis=0) / 2
        wh_bbox = full_bbox[1] - full_bbox[0]
        new_half_size = wh_bbox * self.PERSON_BOX_PRE_ENLARGE_FACTOR / 2
        full_bbox = np.array([
            center_bbox - new_half_size,
            center_bbox + new_half_size], np.int32)

        person_bbox = full_bbox.copy()
        # refine person bbox
        person_bbox[:, 0] = np.clip(person_bbox[:, 0], 0, image.shape[1])
        person_bbox[:, 1] = np.clip(person_bbox[:, 1], 0, image.shape[0])
        if np.any(person_bbox[1] <= person_bbox[0]):
            return None
        # crop to the size of interest
        image = image[person_bbox[0][1]:person_bbox[1][1], person_bbox[0][0]:person_bbox[1][0], :]
        if image.size == 0:
            return None
        # pad to square
        left, top = person_bbox[0] - full_bbox[0]
        right, bottom = full_bbox[1] - person_bbox[1]
        image = cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT, None, (0, 0, 0))
        pad_bias += person_bbox[0] - [left, top]
        # compute rotation
        mid_hip_point -= pad_bias
        full_body_point -= pad_bias
        radians = np.pi / 2 - np.arctan2(-(full_body_point[1] - mid_hip_point[1]), full_body_point[0] - mid_hip_point[0])
        radians = radians - 2 * np.pi * np.floor((radians + np.pi) / (2 * np.pi))
        angle = np.rad2deg(radians)
        #  get rotation matrix
        rotation_matrix = cv.getRotationMatrix2D(mid_hip_point, angle, 1.0)
        #  get rotated image
        rotated_image = cv.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        #  get landmark bounding box
        blob = cv.resize(rotated_image, dsize=self.input_size, interpolation=cv.INTER_AREA).astype(np.float32)
        rotated_person_bbox = np.array([[0, 0], [image.shape[1], image.shape[0]]], dtype=np.int32)
        blob = cv.cvtColor(blob, cv.COLOR_BGR2RGB)
        blob = blob / 255. # [0, 1]
        return blob[np.newaxis, :, :, :], rotated_person_bbox, angle, rotation_matrix, pad_bias

    def infer(self, image, person):
        h, w, _ = image.shape
        # Preprocess
        preprocessed = self._preprocess(image, person)
        if preprocessed is None:
            return None
        input_blob, rotated_person_bbox, angle, rotation_matrix, pad_bias = preprocessed

        # Forward
        output_blob = self._forward_pose(input_blob)
        if output_blob is None:
            return None

        # Postprocess
        results = self._postprocess(output_blob, rotated_person_bbox, angle, rotation_matrix, pad_bias, np.array([w, h]))
        return results # [bbox_coords, landmarks_coords, conf]

    def _postprocess(self, blob, rotated_person_bbox, angle, rotation_matrix, pad_bias, img_size):
        # Be robust against OpenCV DNN returning outputs in a different order.
        # Expected outputs (any order):
        #   - landmarks      : (1, 195)  -> reshape (39, 5)
        #   - landmarks_word : (1, 117) -> reshape (39, 3)
        #   - conf           : (1, 1)   scalar
        #   - mask           : (1, 256, 256, 1)
        #   - heatmap        : (1, 64, 64, 39)
        landmarks = conf = mask = heatmap = landmarks_word = None
        for b in blob:
            if b is None:
                continue
            a = np.asarray(b)
            total = a.size
            if a.ndim == 2 and a.shape == (1, 1):
                if conf is None:
                    conf = a
            elif a.ndim >= 2 and total == 195:
                if landmarks is None:
                    landmarks = a
            elif a.ndim >= 2 and total == 117:
                if landmarks_word is None:
                    landmarks_word = a
            elif a.ndim == 4 and a.shape[-1] == 1 and a.shape[-2] == 256:
                if mask is None:
                    mask = a
            elif a.ndim == 4 and a.shape[-1] == 39 and a.shape[-2] == 64:
                if heatmap is None:
                    heatmap = a

        if landmarks is None or conf is None or mask is None or heatmap is None or landmarks_word is None:
            # Fallback to original order.
            if len(blob) != 5:
                shapes = []
                for idx, b in enumerate(blob):
                    if b is None:
                        shapes.append((idx, None))
                    else:
                        shapes.append((idx, np.asarray(b).shape))
                print("[mp_pose] unexpected output count:", len(blob), "shapes:", shapes)
                return None
            landmarks, conf, mask, heatmap, landmarks_word = blob

        conf = conf[0][0]
        if conf < self.conf_threshold:
            return None

        landmarks = landmarks[0].reshape(-1, 5)  # shape: (1, 195) -> (39, 5)
        landmarks_word = landmarks_word[0].reshape(-1, 3) # shape: (1, 117) -> (39, 3)

        # recover sigmoid score
        landmarks[:, 3:] = 1 / (1 + np.exp(-landmarks[:, 3:]))
        # TODO: refine landmarks with heatmap. reference: https://github.com/tensorflow/tfjs-models/blob/master/pose-detection/src/blazepose_tfjs/detector.ts#L577-L582
        heatmap = heatmap[0]

        # transform coords back to the input coords
        wh_rotated_person_bbox = rotated_person_bbox[1] - rotated_person_bbox[0]
        scale_factor = wh_rotated_person_bbox / self.input_size
        landmarks[:, :2] = (landmarks[:, :2] - self.input_size / 2) * scale_factor
        landmarks[:, 2] = landmarks[:, 2] * max(scale_factor) # depth scaling
        coords_rotation_matrix = cv.getRotationMatrix2D((0, 0), angle, 1.0)
        rotated_landmarks = np.dot(landmarks[:, :2], coords_rotation_matrix[:, :2])
        rotated_landmarks = np.c_[rotated_landmarks, landmarks[:, 2:]]
        rotated_landmarks_world = np.dot(landmarks_word[:, :2], coords_rotation_matrix[:, :2])
        rotated_landmarks_world = np.c_[rotated_landmarks_world, landmarks_word[:, 2]]
        # invert rotation
        rotation_component = np.array([
            [rotation_matrix[0][0], rotation_matrix[1][0]],
            [rotation_matrix[0][1], rotation_matrix[1][1]]])
        translation_component = np.array([
            rotation_matrix[0][2], rotation_matrix[1][2]])
        inverted_translation = np.array([
            -np.dot(rotation_component[0], translation_component),
            -np.dot(rotation_component[1], translation_component)])
        inverse_rotation_matrix = np.c_[rotation_component, inverted_translation]
        #  get box center
        center = np.append(np.sum(rotated_person_bbox, axis=0) / 2, 1)
        original_center = np.array([
            np.dot(center, inverse_rotation_matrix[0]),
            np.dot(center, inverse_rotation_matrix[1])])
        landmarks[:, :2] = rotated_landmarks[:, :2] + original_center + pad_bias

        # get bounding box from rotated_landmarks
        bbox = np.array([
            np.amin(landmarks[:, :2], axis=0),
            np.amax(landmarks[:, :2], axis=0)])  # [top-left, bottom-right]
        center_bbox = np.sum(bbox, axis=0) / 2
        wh_bbox = bbox[1] - bbox[0]
        new_half_size = wh_bbox * self.PERSON_BOX_ENLARGE_FACTOR / 2
        bbox = np.array([
            center_bbox - new_half_size,
            center_bbox + new_half_size])

        # invert rotation for mask
        mask = mask[0].reshape(256, 256) # shape: (1, 256, 256, 1) -> (256, 256)
        invert_rotation_matrix = cv.getRotationMatrix2D((mask.shape[1]/2, mask.shape[0]/2), -angle, 1.0)
        invert_rotation_mask = cv.warpAffine(mask, invert_rotation_matrix, (mask.shape[1], mask.shape[0]))
        # enlarge mask
        invert_rotation_mask = cv.resize(invert_rotation_mask, wh_rotated_person_bbox)
        # crop and pad mask
        min_w, min_h = -np.minimum(pad_bias, 0)
        left, top = np.maximum(pad_bias, 0)
        pad_over = img_size - [invert_rotation_mask.shape[1], invert_rotation_mask.shape[0]] - pad_bias
        max_w, max_h = np.minimum(pad_over, 0) + [invert_rotation_mask.shape[1], invert_rotation_mask.shape[0]]
        right, bottom = np.maximum(pad_over, 0)
        invert_rotation_mask = invert_rotation_mask[min_h:max_h, min_w:max_w]
        invert_rotation_mask = cv.copyMakeBorder(invert_rotation_mask, top, bottom, left, right, cv.BORDER_CONSTANT, None, 0)
        # binarize mask
        invert_rotation_mask = np.where(invert_rotation_mask > 0, 255, 0).astype(np.uint8)

        # 2*2 person bbox: [[x1, y1], [x2, y2]]
        # 39*5 screen landmarks: 33 keypoints and 6 auxiliary points with [x, y, z, visibility, presence], z value is relative to HIP
        # Visibility is probability that a keypoint is located within the frame and not occluded by another bigger body part or another object
        # Presence is probability that a keypoint is located within the frame
        # 39*3 world landmarks: 33 keypoints and 6 auxiliary points with [x, y, z] 3D metric x, y, z coordinate
        # img_height*img_width mask: gray mask, where 255 indicates the full body of a person and 0 means background
        # 64*64*39 heatmap: currently only used for refining landmarks, requires sigmod processing before use
        # conf: confidence of prediction
        return [bbox, landmarks, rotated_landmarks_world, invert_rotation_mask, heatmap, conf]
