import boxjoin
import cv2

filename = "people-walking-original.jpg"
save_path = "people-walking-original-grouped.jpg"

# Load the image using OpenCV (preferred to be numpy array)
img = cv2.imread(filename)

# Each box is in the format of [x1, y1, x2, y2]
# They can be extracted from YOLO output
boxes = [
    [143, 91, 174, 118],
    [142, 98, 164, 123],
    [143, 87, 204, 165],
    [127, 118, 225, 181],
    [371, 195, 386, 220],
    [334, 152, 380, 243],
    [293, 193, 335, 301],
    [470, 136, 494, 167],
    [464, 123, 500, 214],
    [565, 234, 586, 260],
    [554, 178, 582, 261],
    [219, 313, 261, 405],
    [182, 297, 223, 387],
    [151, 315, 196, 421]
]

# Using tolerance to define how close two boxes to be considered overlapping
# Tolerance > 0 means two boxes doesn't need to be overlapping to be considered overlapping
# It works by offseting the box by tolerance value before clustering process.
clusters, grouped_boxes, grouped_boxes_with_offset = boxjoin.BoxClustering(boxes=boxes, img=img, save_path=save_path, tolerance=20)

for i, cluster in enumerate(clusters):
    print(f"Cluster {i}: {cluster}")
    print(f"Cluster box: {grouped_boxes[i]}")
    print(f"Cluster box with offset: {grouped_boxes_with_offset[i]}\n")

    # The output should look like this:

    # Cluster 0: [(123, 71, 194, 138), (122, 78, 184, 143), (123, 67, 224, 185), (107, 98, 245, 201)]
    # Cluster box: [107, 67, 245, 201]
    # Cluster box with offset: [107, 67, 245, 201]

    # Cluster 1: [(351, 175, 406, 240), (314, 132, 400, 263), (273, 173, 355, 321), (199, 293, 281, 425), (162, 277, 243, 407), (131, 295, 216, 427)]
    # Cluster box: [131, 132, 406, 427]
    # Cluster box with offset: [131, 132, 406, 427]

    # Cluster 2: [(450, 116, 514, 187), (444, 103, 520, 234)]
    # Cluster box: [444, 103, 520, 234]
    # Cluster box with offset: [444, 103, 520, 234]

    # Cluster 3: [(545, 214, 606, 280), (534, 158, 602, 281)]
    # Cluster box: [534, 158, 606, 281]
    # Cluster box with offset: [534, 158, 606, 281]