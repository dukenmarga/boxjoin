import boxjoin
from PIL import Image

filename = "people-walking-original.jpg"
save_path = "people-walking-original-grouped.jpg"

# It can also be PIL image
img = Image.open(filename)

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

# We can also adjust the offset (in pixel) of the grouping
clusters, grouped_boxes, grouped_boxes_with_offset = boxjoin.BoxClustering(boxes=boxes, img=img, save_path=save_path, offset=30)

for i, cluster in enumerate(clusters):
    print(f"Cluster {i}: {cluster}")
    print(f"Cluster box: {grouped_boxes[i]}")
    print(f"Cluster box with offset: {grouped_boxes_with_offset[i]}\n")

    # The output should look like this:

    # Cluster 0: [[143, 91, 174, 118], [142, 98, 164, 123], [143, 87, 204, 165], [127, 118, 225, 181]]
    # Cluster box: [127, 87, 225, 181]
    # Cluster box with offset: [97, 57, 255, 211]

    # Cluster 1: [[371, 195, 386, 220], [334, 152, 380, 243], [293, 193, 335, 301]]
    # Cluster box: [293, 152, 386, 301]
    # Cluster box with offset: [263, 122, 416, 331]

    # Cluster 2: [[470, 136, 494, 167], [464, 123, 500, 214]]
    # Cluster box: [464, 123, 500, 214]
    # Cluster box with offset: [434, 93, 530, 244]

    # Cluster 3: [[565, 234, 586, 260], [554, 178, 582, 261]]
    # Cluster box: [554, 178, 586, 261]
    # Cluster box with offset: [524, 148, 616, 291]

    # Cluster 4: [[219, 313, 261, 405], [182, 297, 223, 387], [151, 315, 196, 421]]
    # Cluster box: [151, 297, 261, 421]
    # Cluster box with offset: [121, 267, 291, 427]