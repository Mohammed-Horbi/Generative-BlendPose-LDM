import cv2
import torch
import numpy as np
import mediapipe as mp
from PIL import ImageColor, Image, ImageDraw


def get_face_pose_euler_angles(img):
    """
    Extracts yaw, pitch, and roll from a single face image tensor (C, H, W) in range [-1, 1]
    """    
    # Convert torch tensor to uint8 image for MediaPipe
    if isinstance(img, torch.Tensor):
        img_np = (img.permute(1, 2, 0).cpu().numpy() + 1) / 2  # [-1,1] -> [0,1]
        img_np = (img_np * 255).astype(np.uint8)
    else:
        raise ValueError("Expected image as torch.Tensor in shape [C, H, W]")

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]

        # if len(face_landmarks.landmark) != 468:
        #     return None  # reject bad detection

        # Get selected 2D image points
        image_points = np.array([
            [face_landmarks.landmark[1].x * img_np.shape[1],  face_landmarks.landmark[1].y * img_np.shape[0]],    # Nose tip
            [face_landmarks.landmark[33].x * img_np.shape[1], face_landmarks.landmark[33].y * img_np.shape[0]],   # Left eye
            [face_landmarks.landmark[263].x * img_np.shape[1], face_landmarks.landmark[263].y * img_np.shape[0]], # Right eye
            [face_landmarks.landmark[61].x * img_np.shape[1], face_landmarks.landmark[61].y * img_np.shape[0]],   # Mouth left
            [face_landmarks.landmark[291].x * img_np.shape[1], face_landmarks.landmark[291].y * img_np.shape[0]], # Mouth right
            [face_landmarks.landmark[199].x * img_np.shape[1], face_landmarks.landmark[199].y * img_np.shape[0]], # Forehead
            [face_landmarks.landmark[152].x * img_np.shape[1], face_landmarks.landmark[152].y * img_np.shape[0]], # Chin
            ], dtype='double')

        # 3D model points (approximate values in mm)
        model_points = np.array([
            [0.0, 0.0, 0.0],        # Nose tip
            [-30.0, -30.0, -30.0],  # Left eye
            [30.0, -30.0, -30.0],   # Right eye
            [-30.0, 30.0, -30.0],   # Mouth left
            [30.0, 30.0, -30.0],    # Mouth right
            [0.0, 50.0, -30.0],     # Forehead
            [0.0, -63.0, -30.0]     # Chin
        ], dtype='double')

        focal_length = img_np.shape[1]
        center = (img_np.shape[1] / 2, img_np.shape[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype='double')

        dist_coeffs = np.zeros((4, 1))  # no distortion

        success, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Convert rotation matrix to Euler angles
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            yaw = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            yaw = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = 0

        # Convert to degrees
        yaw = np.degrees(yaw)
        pitch = np.degrees(pitch)
        roll = np.degrees(roll)

        return [yaw, pitch, roll]