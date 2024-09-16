import time
from typing import Tuple, Union
import math
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from deepface import DeepFace
from emotion_maps import send_midi_notes_to_ableton, SCALE_PATTERNS, generate_scale, scale_to_midi

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode

previous_emotion = None


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes and keypoints on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  annotated_image = image.copy()
  height, width, _ = image.shape

  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

    # Draw keypoints
    for keypoint in detection.keypoints:
      keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                     width, height)
      color, thickness, radius = (0, 255, 0), 2, 2
      cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    category_name = '' if category_name is None else category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)


  return annotated_image


# def get_emote(result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
    
#     print(f"face detector result: {result}")
#     image_copy = np.copy(output_image.numpy_view())
#     print(f"output image: {output_image}")

#     # annotated_image = visualize(image_copy, result)
#     # rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
#     try:
#       cv2.imshow("frame", output_image.numpy_view())
#       cv2.waitKey(1)
#     except Exception as e:
#        print(f"Error processing image: {e}")
#     # cv2.waitKey(10)


def detect():
    global previous_emotion
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_asset_path = os.path.join(dir_path, 'detector.tflite')
    base_options = python.BaseOptions(model_asset_path=model_asset_path)
    # options = vision.FaceDetectorOptions(base_options=base_options, running_mode=VisionRunningMode.LIVE_STREAM, result_callback=get_emote)
    options = vision.FaceDetectorOptions(base_options=base_options)

    detector = vision.FaceDetector.create_from_options(options)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_counter= 0
    start_time = time.time()

    while True:
        # Capture frame-by-frame
        # print("capturing frame...")
        ret, frame = cap.read()
        frame_counter += 1

        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

          
        # Reduce resolution by a factor of 4
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        fps = frame_counter / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.3f}", (30, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"frame_counter: {frame_counter}", (200, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2, cv2.LINE_AA)
        # cv2.imshow("frame", frame)
        # key = cv2.waitKey(1)
        # continue

        if frame_counter % 10 == 0:
          mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=small_frame)
          image_copy = np.copy(mp_image.numpy_view())
          try:
            objs = DeepFace.analyze(
                img_path = image_copy,
                actions = ['emotion'],
                enforce_detection=False
            )
        
            # Extract emotion data
            emotions_obj = objs[0]['emotion'] # the keys are 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral', values are float probabilities
            emotion = objs[0]['dominant_emotion']
            print(f"Got emotion {emotion}")

            # Add text (fps frame_couter, emotion) to the image
            cv2.putText(frame, f"FPS: {fps:.3f}", (30, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"frame_counter: {frame_counter}", (200, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image_copy, f"Emotion: {emotion}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            rgb_annotated_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
            cv2.imshow("frame", rgb_annotated_image)
            cv2.waitKey(1)

            if (previous_emotion == emotion):
              print(f"got same emotion... skipping")
              continue
            previous_emotion = emotion

            pattern = SCALE_PATTERNS[emotion]
            print(f"Got pattern {pattern}")
            scale = generate_scale('C', pattern)
            print (f"Got scale {scale}")
            midi = scale_to_midi(scale, 3)
            print (f"Got midi {midi}")
            result = send_midi_notes_to_ableton(0, 0, midi)
            print(f"AbletonOSC response: {result}")

            print(f"emotions:", objs[0]['emotion'])
            
          except Exception as e:
            print(f"Error in emotion analysis: {e}")
    
          # detector.detect_async(mp_image, timestsamp_int)
          # detection_result = detector.detect(mp_image)
          # image_copy = np.copy(mp_image.numpy_view())
          # annotated_image = visualize(image_copy, detection_result)


    # Release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    detect()
