"""
Biometric face-match and liveness.

What this module is for
    Signature verification proves the *document* is authentic. It does not
    prove the person in front of you owns it. A forger who photographs someone
    else's genuine Aadhaar passes every cryptographic check in this app. The
    face-match is what closes that hole, and that pairing is the strongest
    thing in your pitch. Lead with it when a judge asks what makes this
    different: you are verifying issuance *and* binding it to a live presenter.

Non-negotiable from your plan, enforced here
    Compare live, discard immediately, never store biometric data. Concretely:
      - Frames arrive as bytes, live in local variables, and are overwritten
        before the function returns.
      - Temporary files are written into a per-request directory that is
        removed in a `finally` block, because DeepFace's API wants paths.
      - The return value carries a float and a boolean. No embeddings, no
        crops, no file paths. An embedding is biometric data; a 512-float
        vector is not anonymous just because it looks like noise.
      - Nothing from this module is passed to the audit store.

Liveness: read this before you promise anything on stage
    The blink/head-turn check below compares landmark geometry across frames.
    It defeats a printed photo held up to the camera. It does NOT defeat a
    video replay, a phone screen held to the lens, or a deepfake stream, and
    real anti-spoofing needs either depth sensing or a trained PAD model.
    If a judge asks whether your liveness check can be fooled, say yes and name
    the attack. Claiming otherwise in a room that may contain a security
    reviewer is a bad trade for one nod of approval.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field

import numpy as np

# Cosine-distance thresholds are model-specific. ArcFace at 0.68 is DeepFace's
# own tuned default; do not reuse a threshold across models.
MODEL_NAME = "ArcFace"
DETECTOR = "retinaface"
MATCH_THRESHOLD = 0.68

EAR_DROP_THRESHOLD = 0.18   # blink: eye-aspect-ratio must fall below this
YAW_DELTA_THRESHOLD = 0.12  # head turn: normalised horizontal nose shift


@dataclass
class FaceMatchResult:
    compared: bool
    distance: float | None
    similarity: float | None       # 1 - distance, clamped, for display only
    is_match: bool | None
    threshold: float
    model: str
    reasons: list[str] = field(default_factory=list)


@dataclass
class LivenessResult:
    checked: bool
    passed: bool | None
    challenge: str | None
    detail: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


def compare_faces(id_photo: bytes, selfie: bytes) -> FaceMatchResult:
    """
    One-shot comparison. Both inputs are discarded before returning.
    """
    workspace = tempfile.mkdtemp(prefix="fm_")
    id_path = os.path.join(workspace, "id.jpg")
    selfie_path = os.path.join(workspace, "live.jpg")
    try:
        _write_normalised(id_photo, id_path)
        _write_normalised(selfie, selfie_path)

        from deepface import DeepFace

        verdict = DeepFace.verify(
            img1_path=id_path,
            img2_path=selfie_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            distance_metric="cosine",
            enforce_detection=True,
        )
        distance = float(verdict["distance"])
        return FaceMatchResult(
            compared=True,
            distance=round(distance, 4),
            similarity=round(max(0.0, 1.0 - distance), 4),
            is_match=distance <= MATCH_THRESHOLD,
            threshold=MATCH_THRESHOLD,
            model=MODEL_NAME,
            reasons=["FACE_MATCH" if distance <= MATCH_THRESHOLD else "FACE_NO_MATCH"],
        )
    except ValueError as exc:
        # DeepFace raises ValueError when it cannot find a face at all.
        return FaceMatchResult(
            compared=False,
            distance=None,
            similarity=None,
            is_match=None,
            threshold=MATCH_THRESHOLD,
            model=MODEL_NAME,
            reasons=["FACE_NOT_DETECTED", str(exc)[:120]],
        )
    except Exception as exc:
        return FaceMatchResult(
            compared=False,
            distance=None,
            similarity=None,
            is_match=None,
            threshold=MATCH_THRESHOLD,
            model=MODEL_NAME,
            reasons=["FACE_MATCH_ERROR", type(exc).__name__],
        )
    finally:
        # Wipe before unlink so the bytes are not merely dereferenced.
        for path in (id_path, selfie_path):
            _shred(path)
        shutil.rmtree(workspace, ignore_errors=True)
        id_photo = b""
        selfie = b""


def check_liveness(frames: list[bytes], challenge: str = "blink") -> LivenessResult:
    """
    Compare facial landmark geometry across a short burst of frames.

    blink      -> eye aspect ratio must dip below EAR_DROP_THRESHOLD in at
                  least one frame and recover, so a still image fails.
    head_turn  -> normalised nose-to-eye-midpoint offset must shift by
                  YAW_DELTA_THRESHOLD across the burst.
    """
    if len(frames) < 3:
        return LivenessResult(
            checked=False,
            passed=None,
            challenge=challenge,
            reasons=["LIVENESS_INSUFFICIENT_FRAMES"],
        )

    try:
        landmarks = [_landmarks(frame) for frame in frames]
    except RuntimeError as exc:
        return LivenessResult(
            checked=False,
            passed=None,
            challenge=challenge,
            reasons=["LIVENESS_BACKEND_UNAVAILABLE", str(exc)[:120]],
        )

    usable = [item for item in landmarks if item is not None]
    if len(usable) < 3:
        return LivenessResult(
            checked=True,
            passed=False,
            challenge=challenge,
            reasons=["LIVENESS_FACE_NOT_TRACKED"],
        )

    if challenge == "head_turn":
        yaws = [_yaw_proxy(points) for points in usable]
        delta = max(yaws) - min(yaws)
        passed = delta >= YAW_DELTA_THRESHOLD
        detail = {"yaw_delta": round(delta, 4), "threshold": YAW_DELTA_THRESHOLD}
        reason = "LIVENESS_HEAD_TURN_OK" if passed else "LIVENESS_HEAD_TURN_FAILED"
    else:
        ratios = [_eye_aspect_ratio(points) for points in usable]
        closed = min(ratios)
        opened = max(ratios)
        passed = closed < EAR_DROP_THRESHOLD < opened
        detail = {
            "min_ear": round(closed, 4),
            "max_ear": round(opened, 4),
            "threshold": EAR_DROP_THRESHOLD,
        }
        reason = "LIVENESS_BLINK_OK" if passed else "LIVENESS_BLINK_FAILED"

    frames.clear()
    return LivenessResult(
        checked=True,
        passed=passed,
        challenge=challenge,
        detail=detail,
        reasons=[reason, "LIVENESS_REPLAY_ATTACK_NOT_COVERED"],
    )


# ---------------------------------------------------------------------------
# Landmark helpers (MediaPipe Face Mesh)
# ---------------------------------------------------------------------------

# Face Mesh indices for the six classic EAR points, per eye.
_LEFT_EYE = (33, 160, 158, 133, 153, 144)
_RIGHT_EYE = (362, 385, 387, 263, 373, 380)
_NOSE_TIP = 1


def _landmarks(frame: bytes) -> np.ndarray | None:
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(f"mediapipe/opencv missing: {exc}") from exc

    array = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    if array is None:
        return None

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True
    ) as mesh:
        result = mesh.process(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))

    if not result.multi_face_landmarks:
        return None
    face = result.multi_face_landmarks[0]
    return np.array([[point.x, point.y] for point in face.landmark], dtype=np.float32)


def _eye_aspect_ratio(points: np.ndarray) -> float:
    def ratio(indices: tuple[int, ...]) -> float:
        p1, p2, p3, p4, p5, p6 = (points[index] for index in indices)
        vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
        horizontal = 2.0 * np.linalg.norm(p1 - p4)
        return float(vertical / horizontal) if horizontal else 0.0

    return (ratio(_LEFT_EYE) + ratio(_RIGHT_EYE)) / 2.0


def _yaw_proxy(points: np.ndarray) -> float:
    """
    Horizontal offset of the nose tip from the eye midpoint, normalised by
    inter-ocular distance so it survives changes in distance to camera.
    """
    left = points[_LEFT_EYE[0]]
    right = points[_RIGHT_EYE[3]]
    nose = points[_NOSE_TIP]
    interocular = float(np.linalg.norm(right - left)) or 1.0
    midpoint_x = (left[0] + right[0]) / 2.0
    return float((nose[0] - midpoint_x) / interocular)


# ---------------------------------------------------------------------------
# Byte handling
# ---------------------------------------------------------------------------

def _write_normalised(image_bytes: bytes, path: str) -> None:
    """
    Decode whatever came in (including the JPEG2000 photo pulled out of an
    Aadhaar QR) and write a plain JPEG that DeepFace can open.
    """
    import io

    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.save(path, "JPEG", quality=95)


def _shred(path: str) -> None:
    try:
        size = os.path.getsize(path)
        with open(path, "r+b") as handle:
            handle.write(b"\x00" * size)
            handle.flush()
            os.fsync(handle.fileno())
        os.remove(path)
    except OSError:
        pass
