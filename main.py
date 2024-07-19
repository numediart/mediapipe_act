import json
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from websockets.sync.client import connect
from collections import deque


class BlendShapeData:
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def to_dict(self):
        return {
            "Key": self.key,
            "Value": self.value
        }

    def __repr__(self):
        return f'{{Key: "{self.key}", Value: {self.value}}}'


def send_to_server(payload, ws):
    """Send JSON payload to the WebSocket server."""
    ws.send(payload)


def draw_landmarks_on_image(rgb_image, detection_result):
    """Draw face landmarks on the image."""
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for face_landmarks in face_landmarks_list:
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1))
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1))
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=1))

    return annotated_image


def add_blend_shape(blend_shape, value, blendshapes_dict):
    """Add or update the value of a blend shape in the dictionary."""
    blendshapes_dict[blend_shape] = blendshapes_dict.get(blend_shape, 0) + value


def convert_to_blend_shape_data(blendshapes_dict):
    """Convert blend shape dictionary to a list of BlendShapeData objects."""
    return [BlendShapeData(key, value).to_dict() for key, value in blendshapes_dict.items()]


def smooth_blendshapes(blendshapes_dict, buffers, window_size):
    """Apply smoothing to blend shape values using a sliding window."""
    for key, value in blendshapes_dict.items():
        if key not in buffers:
            buffers[key] = deque([value] * window_size, maxlen=window_size)
        else:
            buffers[key].append(value)
        blendshapes_dict[key] = np.mean(buffers[key])
    return blendshapes_dict


# Mapping from Mediapipe blendshapes to custom blendshapes
blendshape_mapping = {
    'browDownLeft': 'browOutVertL',
    'browDownRight': 'browOutVertR',
    'browInnerUp': 'browsMidVert',
    'browOuterUpLeft': 'browSqueezeL',
    'browOuterUpRight': 'browSqueezeR',
    'cheekPuff': 'cheekSneerL',
    'cheekSquintLeft': 'cheekSneerL',
    'cheekSquintRight': 'cheekSneerR',
    'eyeBlinkLeft': 'eyeClosedL',
    'eyeBlinkRight': 'eyeClosedR',
    'eyeLookDownLeft': 'eyesVert',
    'eyeLookDownRight': 'eyesVert',
    'eyeLookInLeft': 'eyesHoriz',
    'eyeLookInRight': 'eyesHoriz',
    'eyeLookOutLeft': 'eyesHoriz',
    'eyeLookOutRight': 'eyesHoriz',
    'eyeLookUpLeft': 'eyesVert',
    'eyeLookUpRight': 'eyesVert',
    'eyeSquintLeft': 'eyeSquintL',
    'eyeSquintRight': 'eyeSquintR',
    'eyeWideLeft': 'pupilsDilatation',
    'eyeWideRight': 'pupilsDilatation',
    'jawForward': 'jawOut',
    'jawLeft': 'jawHoriz',
    'jawOpen': 'mouthOpen',
    'jawRight': 'jawHoriz',
    'mouthClose': 'mouthClosed',
    'mouthDimpleLeft': 'mouthSmileL',
    'mouthDimpleRight': 'mouthSmileR',
    'mouthFrownLeft': 'mouthSmileL',
    'mouthFrownRight': 'mouthSmileR',
    'mouthFunnel': 'mouthOpenO',
    'mouthLeft': 'mouthHoriz',
    'mouthLowerDownLeft': 'mouthLowerOut',
    'mouthLowerDownRight': 'mouthLowerOut',
    'mouthPressLeft': 'mouthBite',
    'mouthPressRight': 'mouthBite',
    'mouthPucker': 'mouthOpenO',
    'mouthRight': 'mouthHoriz',
    'mouthRollLower': 'mouthOpenTeethClosed',
    'mouthRollUpper': 'mouthOpenTeethClosed',
    'mouthShrugLower': 'mouthSmile',
    'mouthShrugUpper': 'mouthSmile',
    'mouthSmileLeft': 'mouthSmileL',
    'mouthSmileRight': 'mouthSmileR',
    'mouthStretchLeft': 'mouthChew',
    'mouthStretchRight': 'mouthChew',
    'mouthUpperUpLeft': 'mouthOpenHalf',
    'mouthUpperUpRight': 'mouthOpenHalf',
    'noseSneerLeft': 'nostrilsExpansion',
    'noseSneerRight': 'nostrilsExpansion'
}


def main():
    model_path = 'face_landmarker_v2_with_blendshapes.task'
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True)

    landmarker = vision.FaceLandmarker.create_from_options(options)

    roomID = input("Enter Room ID: ")
    server_address = f"ws://localhost:9001/mediapipe/blendshapedata/{roomID}"

    blendshape_buffers = {}
    window_size = 5  # Smoothing window size
    blendshape_scaling_factor = 1.3  # Adjust this factor as needed

    try:
        with connect(server_address) as websocket:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    break

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                detection = landmarker.detect(mp_image)

                annotated_image = draw_landmarks_on_image(frame, detection)
                rgb = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2RGBA)
                cv2.imshow("Media pipe result", rgb)
                cv2.setWindowProperty("Media pipe result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                blendshapes_dict = {}
                if detection.face_blendshapes:
                    for blendshape in detection.face_blendshapes[0]:
                        blendshape_name = blendshape.category_name
                        blendshape_value = blendshape.score

                        scaled_value = blendshape_value * blendshape_scaling_factor

                        # Map Mediapipe blendshape name to custom blendshape name
                        if blendshape_name in blendshape_mapping:
                            custom_blendshape_name = blendshape_mapping[blendshape_name]
                            add_blend_shape("Expressions_" + custom_blendshape_name + "_max", scaled_value,
                                            blendshapes_dict)
                            add_blend_shape("Expressions_" + custom_blendshape_name + "_min", 0.0, blendshapes_dict)

                    blendshapes_dict = smooth_blendshapes(blendshapes_dict, blendshape_buffers, window_size)

                    payload = {
                        "RoomId": roomID,
                        "BlendshapeList": convert_to_blend_shape_data(blendshapes_dict),
                    }
                    data = json.dumps({"EventName": "MediapipeBlendshape", "Data": payload})
                    send_to_server(data, websocket)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
