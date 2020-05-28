# genel sorular
# fotonun rennkli olamsı mı daha iyi olur siyah beyaz olması mı?
import numpy as np
import math
import cv2
from matplotlib import pyplot as plt

# Slow down program on big images
def rotate_image(image, angle, rotation_center):
    rot_mat = cv2.getRotationMatrix2D(rotation_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1],
                            flags=cv2.INTER_LINEAR)
    return result


# Ismi yeterince acikliyor mk
def getRotatedCoordinates2D(x, y, angle, rotation_center):
    rot_mat = cv2.getRotationMatrix2D(rotation_center, angle, 1.0)
    point = np.array([x, y, 1])
    new = rot_mat.dot(point)
    return (new[0], new[1])

# sait's process eye code
def process_eye(eye_img, *coordinates_of_eye):
    coordinates_of_eye = list(coordinates_of_eye)
    x1, x2, y1, y2 = coordinates_of_eye

    # rotasyon pls
    angle = abs(math.atan((y1 - y2) / (x1 - x2)) * 180 / math.pi)

    if y2 < y1:
        angle = -angle

    center_of_eye = ((x1 + x2) / 2, (y1 + y2) / 2)
    eye_img = rotate_image(eye_img, angle, center_of_eye)

    # Update coordinates for rotation
    point_num = int(len(coordinates_of_eye) / 2)  # 4 points for 8 element tuple
    for i in range(point_num):
        x, y = coordinates_of_eye[i], coordinates_of_eye[i + point_num]
        x, y = getRotatedCoordinates2D(x, y, angle, center_of_eye)
        coordinates_of_eye[i], coordinates_of_eye[i + point_num] = x, y

    #masking
    center_of_eye = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    major_axis = int((x2- x1) / 2)
    minor_axis = int((x2 - x1) / 5)
    mask = np.zeros_like(eye_img)
    mask = cv2.ellipse(mask, center_of_eye, (major_axis, minor_axis), 0, 0, 360, (255, 255, 255), -1)
    plt.imshow(mask)
    plt.show()
    eye_img = np.bitwise_and(eye_img, mask)

    print("last_y1:"+ str(y1) + " last_y2:" + str(y2))
    print("last_x1:" + str(x1) + " last_x2:" + str(x2))
    eye_img = cv2.resize(eye_img, (64,128))
    return eye_img

# what this function do
def createEyeFrame(img, *coordinates):
    # Parsing eye coordinates
    coordinates = list(coordinates)
    (x1, x2, x3, x4, y1, y2, y3, y4) = coordinates

    # Rotate image 1 times according to position of both eye corners
    angle = abs(math.atan((x1 - x4) / (y1 - y4)) * 180 / math.pi) # wrong angle y/x must be replaced !!!
    rotation_center = ((x1 + x4) / 2, (y1 + y4) / 2)
    img = rotate_image(img, angle, rotation_center)

  # Update coordinates for rotation
    point_num = int(len(coordinates) / 2)  # 4 points for 8 element tuple
    for i in range(point_num):
        x, y = coordinates[i], coordinates[i + point_num]
        x, y = getRotatedCoordinates2D(x, y, angle, rotation_center)
        coordinates[i], coordinates[i + point_num] = x, y


    # Calculate error for eye1
    errx_eye_1 = abs(int((coordinates[0] - coordinates[1]) / 10))

    # Height of an eye is tought as 3 of width
    top_eye1 = int(min(coordinates[4], coordinates[5]) - ((coordinates[0] - coordinates[1]) /3))
    bot_eye1 = int(max(coordinates[4], coordinates[5]) + ((coordinates[0] - coordinates[1]) /3))

    # Get roi according to bounds
    roi_x1 = int(coordinates[0] - errx_eye_1)
    roi_x2= int(coordinates[1] + errx_eye_1)
    width_of_eye1 = abs(coordinates[1]-coordinates[0])
    eye1 = img[int(bot_eye1):int(top_eye1),int(roi_x1):int(roi_x2)] #error here

    # update coordinates x1,x2 for roi
    x1 = errx_eye_1
    x2 = errx_eye_1 + width_of_eye1
    if coordinates[5] > coordinates[4]:
        y2 = coordinates[4] - top_eye1      #coordinates[5]- top_eye1 + height_of_eye1
        y1 = coordinates[5]- top_eye1
    elif coordinates[5] < coordinates[4]:
        y2 = abs(coordinates[4] - top_eye1)
        y1 = abs(coordinates[5] - top_eye1)
    else:
        print("they are already equal, pass the process and return eye")

    # Calculate error for eye2
    errx_eye_2 = abs(int((coordinates[2] - coordinates[3]) / 10))
    # Height of an eye is tought as 1/3 of width
    top_eye2 = int(min(coordinates[6], coordinates[7]) - ((coordinates[2] - coordinates[3]) /3)) # error here
    bot_eye2 = int(max(coordinates[6], coordinates[7]) + ((coordinates[2] - coordinates[3]) /3))

    # Get roi according to bounds
    roi_x3 = int(coordinates[2] - errx_eye_2)
    roi_x4= int(coordinates[3] + errx_eye_2)
    width_of_eye2 = coordinates[3]- coordinates[2]
    eye2 = img[int(bot_eye2):int(top_eye2), int(roi_x3):int(roi_x4)]

    # update coordinates x3,x4 for roi
    x3 = errx_eye_2
    x4 = errx_eye_2 + width_of_eye2
    if coordinates[7] > coordinates[6]:
        y4 = coordinates[6] - top_eye2      #coordinates[5]- top_eye1 + height_of_eye1
        y3 = coordinates[7]- top_eye2
    elif coordinates[7] < coordinates[6]:
        y4 = abs(coordinates[6] - top_eye2)
        y3 = abs(coordinates[7] - top_eye2)
    else:
        print("they are already equal, pass the process and return eye")


    print("y1:"+ str(y1) + " y2:" + str(y2) + " y3:" + str(y3) + " y4:" + str(y4))
    print("x1:" + str(x1) + " x2:" + str(x2) + " x3:" + str(x3) + " x4:" + str(x4))

    #WORKS UNTIL HERE
    # see process function
    eye1 = process_eye(eye1, *(x1, x2, y1, y2)) # error in y coordinates
    eye2 = process_eye(eye2, *(x3, x4, y3, y4))

    #concat two images horizontally
    img = np.column_stack((eye1, eye2))

    return img

img = cv2.imread('yamukgoz.jpeg', -1)
plt.imshow(img)
plt.show()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = createEyeFrame(img, *(386, 450, 493, 537, 90, 142, 180, 214))
plt.imshow(img)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

# Said do the same things for eye2
# Do the concatenation
# return the concatenated eye


