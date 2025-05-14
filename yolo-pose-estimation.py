from ultralytics import YOLO
import cv2

# Load a model
model = YOLO("yolo11n-pose.pt")  # pretrained YOLO11n model

image_path = "cropped_birds/bird_0.jpg"
# Run batched inference on a list of images
results = model([image_path])  # return a list of Results objects

# Load image with OpenCV
image = cv2.imread(image_path)

# Draw keypoints
for result in results:
    keypoints = result.keypoints
    if keypoints is None:
        continue

    kp_list = keypoints.xy  # list of Tensors, each of shape (num_keypoints, 2)

    for person in kp_list:
        for (x, y) in person.tolist():
            cv2.circle(image, (int(x), int(y)), radius=5,
                       color=(0, 255, 0), thickness=-1)

# Save or display the image
cv2.imwrite("pose_output.jpg", image)
cv2.imshow("Pose Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
