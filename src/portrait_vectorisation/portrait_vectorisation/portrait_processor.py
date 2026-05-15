import cv2
import numpy as np
import mediapipe as mp
import os
from typing import List, Tuple
from ament_index_python.packages import get_package_share_directory

import portrait_vectorisation.sig_gen as sig_gen

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

# A stroke is a nav_msgs/Path where each PoseStamped encodes one (x, y) pixel
# coordinate as pose.position.x / .y, with z=0 and identity orientation.
Stroke = Path

# Internal working type — plain (x, y) tuples used during chaining / sorting
# before the final conversion to Path messages.
_Point = Tuple[int, int]
_RawStroke = List[_Point]


class PortraitProcessor:
    def __init__(
        self,
        line_thickness: int = 4,
        sort_strokes: bool = True,
        min_stroke_length: float = 20.0,
        signature_scale: float = 0.40,
    ):
        """
        Parameters
        ----------
        line_thickness : int
            Maximum pixel distance between stroke endpoints for chaining.
            Smaller values -> fewer merges, more pen lifts.
            Larger values  -> more merges, but may incorrectly join unrelated strokes.
        sort_strokes : bool
            Whether to reorder strokes using nearest-neighbour TSP so the arm
            travels the shortest path between strokes.
        min_stroke_length : float
            Minimum contour length (in pixels) required to keep a stroke.
        signature_scale : float
            Scale factor for the signature strokes relative to the image size.
        """
        mp_selfie = mp.solutions.selfie_segmentation
        self.segmenter = mp_selfie.SelfieSegmentation(model_selection=1)
        self.line_thickness = line_thickness
        self.sort_strokes = sort_strokes
        self.min_stroke_length = min_stroke_length
        self.signature_scale = signature_scale

        mp_face = mp.solutions.face_mesh
        self.face_mesh = mp_face.FaceMesh(static_image_mode=False)

        try:
            base = get_package_share_directory("portrait_vectorisation")
        except Exception:
            base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        self.masks = {
            "moustache": cv2.imread(os.path.join(base, "masks", "moustache.png"), cv2.IMREAD_UNCHANGED),
            "hat": cv2.imread(os.path.join(base, "masks", "hat.png"), cv2.IMREAD_UNCHANGED),
            "glasses": cv2.imread(os.path.join(base, "masks", "glasses.png"), cv2.IMREAD_UNCHANGED),
            "nose": cv2.imread(os.path.join(base, "masks", "nose.png"), cv2.IMREAD_UNCHANGED),
        }
        self.signature_strokes = self._load_signature_svg(
            os.path.join(base, "signature", "signature.svg")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        image: np.ndarray,
        frame_id: str = 'camera_frame',
        mask_type: str = "none",
    ) -> Tuple[np.ndarray, List[Stroke]]:
        """
        Full portrait processing pipeline.

        Parameters
        ----------
        image : np.ndarray
            BGR input frame from the camera.
        frame_id : str
            frame_id written into every Path and PoseStamped header.
            Should match whatever frame_id the rest of your pipeline expects.

        Returns
        -------
        canvas : np.ndarray
            Preview image (black lines on white), same resolution as input.
        strokes : List[nav_msgs/Path]
            Ordered list of strokes ready for the UR3.  Each Path contains an
            ordered sequence of PoseStamped waypoints.  pose.position.x/y hold
            pixel coordinates; z is 0 and orientation is identity quaternion.
            The list is ordered to minimise total travel distance between
            consecutive strokes.
        """
        img = self.remove_background(image)
        if mask_type != "none":
            img = self.apply_mask(img, mask_type)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 9, 120, 120)
        edges = cv2.Canny(filtered, 20, 50)

        raw_contours = self._extract_contours(edges)
        raw_strokes = self._contours_to_raw(raw_contours)
        raw_strokes = self._chain_strokes(raw_strokes)

        raw_strokes = self._add_signature_strokes(raw_strokes, img.shape)

        if self.sort_strokes:
            raw_strokes = self._sort_strokes(raw_strokes)

        canvas = self._render(edges, raw_strokes)
        strokes = [self._raw_to_path(s, frame_id) for s in raw_strokes]
        return canvas, strokes

    def close(self):
        self.segmenter.close()
        self.face_mesh.close()

    # ------------------------------------------------------------------
    # Background removal (unchanged from original)
    # ------------------------------------------------------------------

    def remove_background(self, image: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.segmenter.process(rgb)
        mask = results.segmentation_mask
        mask = (mask > 0.4).astype(np.uint8)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 0)
        mask = mask > 0.3
        output = image.copy()
        output[~mask] = 255
        return output

    # ------------------------------------------------------------------
    # Mask application (dynamic scale + rotation)
    # ------------------------------------------------------------------

    def apply_mask(self, image: np.ndarray, mask_type: str) -> np.ndarray:
        if mask_type not in self.masks:
            return image

        mask_img = self.masks[mask_type]
        if mask_img is None:
            return image

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return image

        lm = res.multi_face_landmarks[0].landmark
        h, w, _ = image.shape

        def pt(i: int) -> np.ndarray:
            return np.array([lm[i].x * w, lm[i].y * h])

        left_eye = pt(33)
        right_eye = pt(263)
        nose = pt(1)

        eye_dist = np.linalg.norm(right_eye - left_eye)

        dx, dy = right_eye - left_eye
        angle = -np.degrees(np.arctan2(dy, dx))

        if mask_type == "moustache":
            width = int(eye_dist * 1.2)
            height = int(width * 0.4)
            anchor_x = width / 2
            anchor_y = 0
            anchor_target_x = float(nose[0])
            anchor_target_y = float(nose[1])

        elif mask_type == "glasses":
            width = int(eye_dist * 1.8)
            height = int(width * 0.35)
            center = (left_eye + right_eye) / 2
            anchor_x = width / 2
            anchor_y = height / 2
            anchor_target_x = float(center[0])
            anchor_target_y = float(center[1])

        elif mask_type == "hat":
            width = int(eye_dist * 2.7)
            hat_aspect = mask_img.shape[0] / mask_img.shape[1]
            height = int(width * hat_aspect)
            center = (left_eye + right_eye) / 2
            anchor_x = width / 2
            anchor_y = height * 0.90
            up = np.array([dy, -dx], dtype=np.float32)
            up_norm = np.linalg.norm(up)
            if up_norm > 1e-6:
                up /= up_norm
            else:
                up = np.array([0.0, -1.0], dtype=np.float32)
            anchor_target = center - (up * (eye_dist * 0.40))
            anchor_target_x = float(anchor_target[0])
            anchor_target_y = float(anchor_target[1])

        elif mask_type == "nose":
            width = int(eye_dist * 2.0)
            height = int(width * 0.70)
            anchor_x = width / 2
            anchor_y = height * 0.60
            anchor_target_x = float(nose[0])
            anchor_target_y = float(nose[1])

        else:
            return image

        mask_resized = cv2.resize(mask_img, (width, height))

        mask_rot, anchor_rot = self._rotate_image_around_anchor(
            mask_resized,
            angle,
            (anchor_x, anchor_y),
        )
        x = int(anchor_target_x - anchor_rot[0])
        y = int(anchor_target_y - anchor_rot[1])

        return self._overlay(image, mask_rot, x, y)

    def _rotate_image_around_anchor(
        self,
        image: np.ndarray,
        angle_deg: float,
        anchor: Tuple[float, float],
    ) -> Tuple[np.ndarray, Tuple[float, float]]:
        h, w = image.shape[:2]
        anchor_x, anchor_y = anchor

        M = cv2.getRotationMatrix2D((anchor_x, anchor_y), angle_deg, 1.0)

        corners = np.array(
            [[0, 0], [w, 0], [w, h], [0, h]],
            dtype=np.float32,
        )
        ones = np.ones((corners.shape[0], 1), dtype=np.float32)
        corners_h = np.hstack([corners, ones])
        rotated = corners_h @ M.T

        min_x = float(np.min(rotated[:, 0]))
        min_y = float(np.min(rotated[:, 1]))
        max_x = float(np.max(rotated[:, 0]))
        max_y = float(np.max(rotated[:, 1]))

        new_w = int(np.ceil(max_x - min_x))
        new_h = int(np.ceil(max_y - min_y))

        M[0, 2] -= min_x
        M[1, 2] -= min_y

        rotated_img = cv2.warpAffine(
            image,
            M,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        anchor_rot = (anchor_x - min_x, anchor_y - min_y)
        return rotated_img, anchor_rot

    def _overlay(self, bg: np.ndarray, overlay: np.ndarray, x: int, y: int) -> np.ndarray:
        h, w = overlay.shape[:2]

        x0 = max(x, 0)
        y0 = max(y, 0)
        x1 = min(x + w, bg.shape[1])
        y1 = min(y + h, bg.shape[0])

        if x0 >= x1 or y0 >= y1:
            return bg

        ox0 = x0 - x
        oy0 = y0 - y
        ox1 = ox0 + (x1 - x0)
        oy1 = oy0 + (y1 - y0)

        overlay_crop = overlay[oy0:oy1, ox0:ox1]
        if overlay_crop.shape[2] == 4:
            alpha = overlay_crop[:, :, 3] / 255.0
            overlay_rgb = overlay_crop[:, :, :3]
        else:
            alpha = np.ones((overlay_crop.shape[0], overlay_crop.shape[1]), dtype=np.float32)
            overlay_rgb = overlay_crop

        for c in range(3):
            bg[y0:y1, x0:x1, c] = (
                alpha * overlay_rgb[:, :, c] +
                (1 - alpha) * bg[y0:y1, x0:x1, c]
            )

        return bg

    def _load_signature_svg(self, path: str) -> List[_RawStroke] | None:
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                svg_text = handle.read()
            try:
                svg_text = sig_gen.extract_svg_text(svg_text)
            except ValueError:
                pass
            strokes = sig_gen.svg_to_strokes(svg_text)
            if not isinstance(strokes, list):
                return None
            return [
                [(float(x), float(y)) for x, y in stroke]
                for stroke in strokes
                if isinstance(stroke, list) and len(stroke) >= 2
            ]
        except Exception:
            return None

    def _add_signature_strokes(
        self,
        strokes: List[_RawStroke],
        image_shape: Tuple[int, int, int],
    ) -> List[_RawStroke]:
        if not self.signature_strokes:
            return strokes

        img_h, img_w = image_shape[:2]

        sig_points = [p for stroke in self.signature_strokes for p in stroke]
        xs = [p[0] for p in sig_points]
        ys = [p[1] for p in sig_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        sig_w = max_x - min_x
        sig_h = max_y - min_y
        if sig_w <= 0 or sig_h <= 0:
            return strokes

        max_w = int(img_w * self.signature_scale)
        max_h = int(img_h * self.signature_scale)
        scale = min(1.0, max_w / sig_w, max_h / sig_h)

        pad = max(5, int(min(img_w, img_h) * 0.01))
        scaled_w = sig_w * scale
        scaled_h = sig_h * scale
        candidates = [
            (pad, pad),
            (img_w - scaled_w - pad, pad),
            (pad, img_h - scaled_h - pad),
            (img_w - scaled_w - pad, img_h - scaled_h - pad),
        ]

        points = [p for stroke in strokes for p in stroke]

        def count_hits(x: float, y: float) -> int:
            x1 = x + scaled_w
            y1 = y + scaled_h
            return sum(1 for px, py in points if x <= px <= x1 and y <= py <= y1)

        best_x, best_y = min(candidates, key=lambda pos: count_hits(pos[0], pos[1]))

        placed: List[_RawStroke] = []
        for stroke in self.signature_strokes:
            placed.append([
                (
                    int(round((x - min_x) * scale + best_x)),
                    int(round((y - min_y) * scale + best_y)),
                )
                for x, y in stroke
            ])

        return strokes + placed

    # ------------------------------------------------------------------
    # Stage 1 -- extract and simplify contours
    # ------------------------------------------------------------------

    def _extract_contours(self, edges: np.ndarray) -> List[np.ndarray]:
        """Return simplified contours, filtering out short noise."""
        contours, _ = cv2.findContours(
            edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )
        result = []
        for c in contours:
            if cv2.arcLength(c, False) < self.min_stroke_length:
                continue
            smooth = cv2.approxPolyDP(c, 2.0, False)
            result.append(smooth)
        return result

    # ------------------------------------------------------------------
    # Stage 2 -- convert OpenCV contour arrays -> internal (x, y) lists
    # ------------------------------------------------------------------

    def _contours_to_raw(self, contours: List[np.ndarray]) -> List[_RawStroke]:
        """
        OpenCV stores contour points as shape (N, 1, 2).
        Flatten each one to a plain list of (x, y) integer tuples for use
        during chaining and sorting before the final Path conversion.
        """
        strokes = []
        for c in contours:
            pts = [(int(p[0][0]), int(p[0][1])) for p in c]
            if len(pts) >= 2:
                strokes.append(pts)
        return strokes

    # ------------------------------------------------------------------
    # Stage 3 -- convert internal (x, y) lists -> nav_msgs/Path
    # ------------------------------------------------------------------

    def _raw_to_path(self, raw: _RawStroke, frame_id: str) -> Path:
        """
        Convert a list of (x, y) pixel tuples to a nav_msgs/Path message.

        Each waypoint becomes a PoseStamped with:
          pose.position.x  = pixel x
          pose.position.y  = pixel y
          pose.position.z  = 0.0
          pose.orientation = identity quaternion (w=1)

        The Path header carries the same frame_id as each pose; timestamps
        are left at zero -- stamp them in the ROS2 node using node.get_clock()
        if your downstream consumer requires it.
        """
        path = Path()
        path.header.frame_id = frame_id

        for (x, y) in raw:
            pose = PoseStamped()
            pose.header.frame_id = frame_id
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

        return path

    # ------------------------------------------------------------------
    # Feature 1 -- Chain nearby strokes into longer continuous strokes
    # ------------------------------------------------------------------

    def _chain_strokes(self, strokes: List[_RawStroke]) -> List[_RawStroke]:
        """
        Merge strokes whose endpoints are within `chain_threshold` pixels of
        each other into a single continuous stroke.

        For each candidate pair the algorithm checks all four endpoint
        combinations (end->start, end->end, start->start, start->end) and picks
        the join that requires the shortest travel, reversing a stroke's
        direction when necessary so the merged result always reads start->end.

        This is a greedy single-pass algorithm:
          - Start with the first unmerged stroke.
          - Repeatedly scan remaining strokes for the closest connectable
            endpoint within the threshold.
          - Append (or prepend) it and continue growing the current stroke.
          - When no neighbour is within the threshold, emit the stroke and
            start a new one.
        """
        if not strokes:
            return strokes

        remaining = [list(s) for s in strokes]
        chained: List[_RawStroke] = []

        while remaining:
            current = remaining.pop(0)

            merged = True
            while merged:
                merged = False
                best_dist = self.line_thickness * 2  # max distance to consider for chaining
                best_idx = -1
                best_mode = None   # ('append'|'prepend', flip: bool)

                for i, candidate in enumerate(remaining):
                    # current end -> candidate start
                    d = _dist(current[-1], candidate[0])
                    if d < best_dist:
                        best_dist, best_idx, best_mode = d, i, ('append', False)

                    # current end -> candidate end  (flip candidate)
                    d = _dist(current[-1], candidate[-1])
                    if d < best_dist:
                        best_dist, best_idx, best_mode = d, i, ('append', True)

                    # current start -> candidate end
                    d = _dist(current[0], candidate[-1])
                    if d < best_dist:
                        best_dist, best_idx, best_mode = d, i, ('prepend', False)

                    # current start -> candidate start  (flip candidate)
                    d = _dist(current[0], candidate[0])
                    if d < best_dist:
                        best_dist, best_idx, best_mode = d, i, ('prepend', True)

                if best_idx != -1:
                    neighbour = remaining.pop(best_idx)
                    action, flip = best_mode
                    if flip:
                        neighbour = neighbour[::-1]
                    current = current + neighbour if action == 'append' else neighbour + current
                    merged = True

            chained.append(current)

        return chained

    # ------------------------------------------------------------------
    # Feature 2 -- Sort strokes to minimise pen-travel (nearest neighbour)
    # Feature 3 -- Direction flipping per stroke based on pen position
    # ------------------------------------------------------------------

    def _sort_strokes(self, strokes: List[_RawStroke]) -> List[_RawStroke]:
        """
        Reorder strokes using a nearest-neighbour greedy heuristic so the arm
        travels the shortest path between consecutive strokes.

        Direction flipping is applied here: before committing to the next
        stroke, both its start and end are considered as the entry point, and
        the stroke is reversed if that reduces travel from the current pen
        position.
        """
        if not strokes:
            return strokes

        remaining = list(strokes)
        sorted_strokes: List[_RawStroke] = []

        # Start from the stroke whose start point is closest to the image
        # origin (top-left), which is a natural first pen-down position.
        first_idx = min(
            range(len(remaining)),
            key=lambda i: _dist(remaining[i][0], (0, 0))
        )
        current_stroke = remaining.pop(first_idx)
        sorted_strokes.append(current_stroke)
        pen_pos = current_stroke[-1]

        while remaining:
            best_dist = float('inf')
            best_idx = -1
            best_flip = False

            for i, candidate in enumerate(remaining):
                d_start = _dist(pen_pos, candidate[0])
                d_end = _dist(pen_pos, candidate[-1])

                if d_start < best_dist:
                    best_dist, best_idx, best_flip = d_start, i, False

                if d_end < best_dist:
                    best_dist, best_idx, best_flip = d_end, i, True

            next_stroke = remaining.pop(best_idx)

            if best_flip:
                next_stroke = next_stroke[::-1]

            sorted_strokes.append(next_stroke)
            pen_pos = next_stroke[-1]

        return sorted_strokes

    # ------------------------------------------------------------------
    # Rendering -- draw sorted raw strokes onto a preview canvas
    # ------------------------------------------------------------------

    def _render(self, edges: np.ndarray, strokes: List[_RawStroke]) -> np.ndarray:
        """
        Render strokes onto a white BGR canvas.
        Each stroke is drawn in its own colour, currently all black (0, 0, 0).
        The colour per stroke is determined by _stroke_colour(), making it
        straightforward to introduce per-stroke colours for multicolour support.
        """
        canvas = np.ones((*edges.shape, 3), dtype=np.uint8) * 255
        for idx, stroke in enumerate(strokes):
            pts = np.array(stroke, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], isClosed=False, color=self._stroke_colour(idx), thickness=self.line_thickness)
        return canvas

    def _stroke_colour(self, stroke_idx: int) -> Tuple[int, int, int]:
        """
        Return the BGR colour for a given stroke index.
        Currently returns black for all strokes.
        Override or extend this method to add multicolour support in the future.
        """
        return (0, 0, 0)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _dist(a: _Point, b: _Point) -> float:
    """Euclidean distance between two (x, y) points."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5