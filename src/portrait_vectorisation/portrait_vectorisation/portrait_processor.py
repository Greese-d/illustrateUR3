import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple

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
        chain_threshold: float = 10.0,
        sort_strokes: bool = True,
        line_thickness: int = 6,
    ):
        """
        Parameters
        ----------
        chain_threshold : float
            Maximum pixel distance between the endpoint of one contour and the
            startpoint of another for them to be merged into a single stroke.
            Smaller values -> fewer merges, more pen lifts.
            Larger values  -> more merges, but may incorrectly join unrelated strokes.
        sort_strokes : bool
            Whether to reorder strokes using nearest-neighbour TSP so the arm
            travels the shortest path between strokes.
        """
        mp_selfie = mp.solutions.selfie_segmentation
        self.segmenter = mp_selfie.SelfieSegmentation(model_selection=1)
        self.chain_threshold = chain_threshold
        self.sort_strokes = sort_strokes
        self.line_thickness = line_thickness

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        image: np.ndarray,
        frame_id: str = 'camera_frame',
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
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 9, 120, 120)
        edges = cv2.Canny(filtered, 25, 60)

        raw_contours = self._extract_contours(edges)
        raw_strokes = self._contours_to_raw(raw_contours)
        raw_strokes = self._chain_strokes(raw_strokes)

        if self.sort_strokes:
            raw_strokes = self._sort_strokes(raw_strokes)

        canvas = self._render(edges, raw_strokes)
        strokes = [self._raw_to_path(s, frame_id) for s in raw_strokes]
        return canvas, strokes

    def close(self):
        self.segmenter.close()

    # ------------------------------------------------------------------
    # Background removal (unchanged from original)
    # ------------------------------------------------------------------

    def remove_background(self, image: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.segmenter.process(rgb)
        mask = results.segmentation_mask
        mask = (mask > 0.7).astype(np.uint8)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 0)
        mask = mask > 0.3
        output = image.copy()
        output[~mask] = 255
        return output

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
            if cv2.arcLength(c, False) < 20:
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
                best_dist = self.line_thickness  # Max distance to consider for chaining
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
        canvas = np.ones_like(edges) * 255
        for stroke in strokes:
            pts = np.array(stroke, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], isClosed=False, color=0, thickness=self.line_thickness)
        return canvas


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _dist(a: _Point, b: _Point) -> float:
    """Euclidean distance between two (x, y) points."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5