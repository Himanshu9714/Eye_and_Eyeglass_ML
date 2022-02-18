from scipy.spatial import distance as dist
from imutils import perspective, contours
from google.colab.patches import cv2_imshow
import imutils
import cv2
import mediapipe as mp
import time
import utils, math
import dlib
import numpy as np
from scipy import ndimage


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def image_processing(image):
    # image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), cv2.BORDER_DEFAULT)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 50, 150)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    return image, cnts


def finding_pixel_per_metric(image, cnts):
    pixelsPerMetric = None
    for c in cnts:
        if cv2.contourArea(c) < 100:
            continue
        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

            cv2.line(
                orig,
                (int(tltrX), int(tltrY)),
                (int(blbrX), int(blbrY)),
                (255, 0, 255),
                2,
            )
            cv2.line(
                orig,
                (int(tlblX), int(tlblY)),
                (int(trbrX), int(trbrY)),
                (255, 0, 255),
                2,
            )

            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            if pixelsPerMetric is None:
                print((tltrX, tltrY), (blbrX, blbrY))
                print((tlblX, tlblY), (trbrX, trbrY))
                pixelsPerMetric = dB / (0.75)

            dimA = dA / pixelsPerMetric
            dimB = dB / pixelsPerMetric

            print(f"Pixels Per Metric: {pixelsPerMetric}\ndA: {dimA}\ndB: {dimB}")

            cv2.putText(
                orig,
                "{:.1f}in".format(dimA),
                (int(tltrX - 15), int(tltrY - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                orig,
                "{:.1f}in".format(dimB),
                (int(trbrX + 10), int(trbrY)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2_imshow(orig)
            cv2.waitKey(0)
            if pixelsPerMetric is not None:
                return pixelsPerMetric
    return pixelsPerMetric


def pixel_per_metric_measurement(image):
    # load the image, convert it to grayscale, and blur it slightly
    image = resize(image, 700)
    image, cnts = image_processing(image)
    pixelsPerMetric = finding_pixel_per_metric(image, cnts)
    return pixelsPerMetric


# Resize an image to a certain width
def resize(img, width):
    r = float(width) / img.shape[1]
    print(f"Width: {width}, R: {r}, img.shape[0]: {img.shape}")
    dim = (width, int(img.shape[0] * r))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img


# Combine an image that has a transparency alpha channel
def blend_transparent(face_img, sunglasses_img):
    overlay_img = sunglasses_img[:, :, :3]
    overlay_mask = sunglasses_img[:, :, 3:]

    background_mask = 255 - overlay_mask

    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))
    # print(f"This is int arr: {type(np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0)))}")

    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


# Find the angle between two points
def angle_between(point_1, point_2):
    angle_1 = np.arctan2(*point_1[::-1])
    angle_2 = np.arctan2(*point_2[::-1])
    return np.rad2deg((angle_1 - angle_2) % (2 * np.pi))


def facial_landmarks_detection(landmarks, img):
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])

        if idx == 0:
            eye_left = pos

        elif idx == 16:
            eye_right = pos

        elif idx == 19:
            right_eyebrow = pos

        elif idx == 24:
            left_eyebrow = pos

        elif idx == 27:
            nose_points = pos
            cv2.circle(
                img,
                (int(right_eyebrow[0] + 10), int(right_eyebrow[1] + 5)),
                5,
                (255, 0, 0),
                -1,
            )
            cv2.circle(
                img,
                (int(left_eyebrow[0] - 10), int(left_eyebrow[1] + 5)),
                5,
                (0, 255, 0),
                -1,
            )
            cv2.circle(
                img, (int(nose_points[0]), int(nose_points[1])), 5, (0, 0, 255), -1
            )
            cv2_imshow(img)

        try:
            # cv2.line(img_copy, eye_left, eye_right, color=(0, 255, 255))
            degree = np.rad2deg(
                np.arctan2(eye_left[0] - eye_right[0], eye_left[1] - eye_right[1])
            )

        except Exception as e:
            pass
    return eye_left, eye_right, right_eyebrow, left_eyebrow, nose_points, degree


def resize_and_rotate_glasses(
    eye_left, eye_right, glasses, degree, x, y, w, h, img, img_copy
):
    # Translate facial object based on input object.
    eye_center = (eye_left[1] + eye_right[1]) / 2

    # Sunglasses translation
    glass_trans = int(0.2 * (eye_center - y))

    # Funny tanslation
    # glass_trans = int(-.3 * (eye_center - y ))

    # Mask translation
    # glass_trans = int(-.8 * (eye_center - y))

    # resize glasses to width of face and blend images
    face_width = w - x

    # resize_glasses
    glasses_resize = resize(glasses, face_width)
    # Rotate glasses based on angle between eyes
    yG, xG, cG = glasses_resize.shape
    glasses_resize_rotated = ndimage.rotate(glasses_resize, (degree + 90))
    glass_rec_rotated = ndimage.rotate(
        img[y + glass_trans : y + yG + glass_trans, x:w], (degree + 90)
    )

    # blending with rotation
    h5, w5, s5 = glass_rec_rotated.shape

    rec_resize = img_copy[y + glass_trans : y + h5 + glass_trans, x : x + w5]
    blend_glass3 = blend_transparent(rec_resize, glasses_resize_rotated)

    img_copy[y + glass_trans : y + h5 + glass_trans, x : x + w5] = blend_glass3
    return img_copy


def glass_selection(img_path, frame_path):
    video_capture = cv2.imread(img_path)
    glasses = cv2.imread(frame_path, -1)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        "/content/sample_data/shape_predictor_68_face_landmarks.dat"
    )

    # Start main program
    img = video_capture
    img = resize(img, 700)
    img_copy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    try:
        # detect faces
        dets = detector(gray, 1)
        # find face box bounding points
        for d in dets:
            x = d.left()
            y = d.top()
            w = d.right()
            h = d.bottom()

        dlib_rect = dlib.rectangle(x, y, w, h)

        ##############   Find facial landmarks   ##############
        detected_landmarks = predictor(gray, dlib_rect).parts()
        landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])
        (
            eye_left,
            eye_right,
            right_eyebrow,
            left_eyebrow,
            nose_points,
            degree,
        ) = facial_landmarks_detection(landmarks, img)
        print(eye_left, eye_right)

        ##############   Resize and rotate glasses   ##############
        img_copy = resize_and_rotate_glasses(
            eye_left, eye_right, glasses, degree, x, y, w, h, img, img_copy
        )
        # img_copy = resize(img_copy, 368)

        # cv2.circle(img, (int(eye_left[0]+5), int(eye_left[1]-2)), 5, (255,0,0), -1)
        cv2_imshow(img_copy)

    except:
        cv2_imshow(img_copy)
    return img_copy


# landmark detection function
def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [
        (int(point.x * img_width), int(point.y * img_height))
        for point in results.multi_face_landmarks[0].landmark
    ]
    if draw:
        [cv2.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks
    return mesh_coord


# Euclidean distance
def euclideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return distance


# Blinking Ratio
def pupilDistance(img, landmarks, right_indices, left_indices, pixelsPerMetric):
    # Right eyes
    # horizontal line
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    print("Right and Left of Right Eye", rh_right, rh_left)
    # draw lines on right eyes
    # cv2.line(img, rh_right, rh_left, utils.GREEN, 2)
    mids = ((rh_right[0] + rh_left[0]) / 2, (rh_right[1] + rh_left[1]) / 2)
    cv2.circle(img, (int(mids[0] - 5), int(mids[1])), 5, (255, 0, 0), -1)

    # LEFT_EYE
    # horizontal line
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    mids1 = ((lh_right[0] + lh_left[0]) / 2, (lh_right[1] + lh_left[1]) / 2)
    cv2.circle(img, (int(mids1[0] + 5), int(mids1[1] - 2)), 5, (255, 0, 0), -1)
    midsDistance = euclideanDistance(mids1, mids)

    rhDistance = euclideanDistance(rh_right, rh_left)
    lhDistance = euclideanDistance(lh_right, lh_left)


def eye_measurement(img_path, pixelsPerMetric):
    FONTS = cv2.FONT_HERSHEY_COMPLEX
    FACE_OVAL = [
        10,
        338,
        297,
        332,
        284,
        251,
        389,
        356,
        454,
        323,
        361,
        288,
        397,
        365,
        379,
        378,
        400,
        377,
        152,
        148,
        176,
        149,
        150,
        136,
        172,
        58,
        132,
        93,
        234,
        127,
        162,
        21,
        54,
        103,
        67,
        109,
    ]
    LIPS = [
        61,
        146,
        91,
        181,
        84,
        17,
        314,
        405,
        321,
        375,
        291,
        308,
        324,
        318,
        402,
        317,
        14,
        87,
        178,
        88,
        95,
        185,
        40,
        39,
        37,
        0,
        267,
        269,
        270,
        409,
        415,
        310,
        311,
        312,
        13,
        82,
        81,
        42,
        183,
        78,
    ]
    LOWER_LIPS = [
        61,
        146,
        91,
        181,
        84,
        17,
        314,
        405,
        321,
        375,
        291,
        308,
        324,
        318,
        402,
        317,
        14,
        87,
        178,
        88,
        95,
    ]
    UPPER_LIPS = [
        185,
        40,
        39,
        37,
        0,
        267,
        269,
        270,
        409,
        415,
        310,
        311,
        312,
        13,
        82,
        81,
        42,
        183,
        78,
    ]
    LEFT_EYE = [
        362,
        382,
        381,
        380,
        374,
        373,
        390,
        249,
        263,
        466,
        388,
        387,
        386,
        385,
        384,
        398,
    ]
    LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
    RIGHT_EYE = [
        33,
        7,
        163,
        144,
        145,
        153,
        154,
        155,
        133,
        173,
        157,
        158,
        159,
        160,
        161,
        246,
    ]
    RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

    map_face_mesh = mp.solutions.face_mesh
    # camera object
    camera = cv2.VideoCapture(img_path)

    with map_face_mesh.FaceMesh(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as face_mesh:
        # starting time here
        start_time = time.time()
        # starting Video loop here.
        while True:
            ret, frame = camera.read()  # getting frame from camera
            if not ret:
                break  # no more frames break
            #  resizing frame
            frame = resize(frame, 700)

            # frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            frame_height, frame_width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                mesh_coords = landmarksDetection(frame, results, False)
                pupilDistance(frame, mesh_coords, RIGHT_EYE, LEFT_EYE, pixelsPerMetric)
                # cv2.polylines(frame,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv2.LINE_AA)
                # cv2.polylines(frame,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv2.LINE_AA)

            # calculating  frame per seconds FPS
            end_time = time.time() - start_time

            cv2_imshow(frame)
            key = cv2.waitKey(0)
            if key == ord("q") or key == ord("Q"):
                break
        cv2.destroyAllWindows()
        camera.release()


def eyeglass_frame_measurement():
    pass


def main():
    img_path = "12.jpg"
    frame_path = "f2.png"
    image = cv2.imread(img_path)
    pixelsPerMetric = pixel_per_metric_measurement(image)
    eye_measurement(img_path, pixelsPerMetric)
    img_copy = glass_selection(img_path, frame_path)


main()
