# Source - https://stackoverflow.com/q
# Posted by Nikita Kit, modified by community. See post 'Timeline' for change history
# Retrieved 2025-12-13, License - CC BY-SA 4.0

import cv2
import numpy 
import boxjoin

img = cv2.imread("medium.png")
img0 = img.copy()
blue, green, red = cv2.split(img)

def medianCanny(img, thresh1, thresh2):
    median = numpy.median(img)
    img = cv2.Canny(img, int(thresh1 * median), int(thresh2 * median))
    return img

blue_edges = medianCanny(blue, 0, 1)
green_edges = medianCanny(green, 0, 1)
red_edges = medianCanny(red, 0, 1)

edges = blue_edges | green_edges | red_edges

contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)

boxes = []

# Save original bounding box
for cnt in contours:
    # Get bounding box
    # x,y is top left corner (not center)
    x,y,w,h = cv2.boundingRect(cnt)

    # Ignore large bounding box
    if w * h > 10000:
        continue

    # Save original bounding box
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
    boxes.append([x,y,w,h])

# Save original before grouping
cv2.imwrite("medium-ungrouped.png", img)

# Grouping and saving
save_path = "medium-grouped.png"
clusters = boxjoin.BoxClustering(boxes=boxes, img=img0, save_path=save_path, tolerance=13, mode="xywh", xy_mode="top_left", thickness=2)

cv2.imshow("Grouped bounding boxes",img0)
cv2.waitKey(0)
cv2.destroyAllWindows()
