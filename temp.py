import subprocess
import cv2
import tempfile
import os

def capture_image():
    # Open a connection to the default camera (0)
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not access the camera.")
        return None

    # Capture a single frame
    ret, frame = camera.read()

    if ret:
        # Create a temporary file to save the image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_img:
            temp_filename = temp_img.name
            cv2.imwrite(temp_filename, frame)
            print(f"Image saved as temporary file '{temp_filename}'")

        camera.release()
        return temp_filename
    else:
        print("Error: Could not capture an image.")
        camera.release()
        return None


def run_yolo_detection(model_path="best.pt", show=True):
    img_path = capture_image()
    
    if not img_path:
        print("Error: No valid image to process.")
        return

    # Prepare the YOLO command
    command = [
        "python", "ultralytics/yolo/v8/detect/predict.py",
        f"model={model_path}", f"source={img_path}",
        f"show={str(show).lower()}"
    ]

    try:
        # Run the YOLO detection
        subprocess.run(command, check=True)
        print("YOLO detection completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running YOLO detection: {e}")
    except Exception as ex:
        print(f"Unexpected error: {ex}")
    finally:
        # Clean up the temporary file
        if os.path.exists(img_path):
            os.remove(img_path)
            print(f"Temporary image file '{img_path}' deleted.")

# Example usage
run_yolo_detection()
