from scipy.spatial import distance as dist


def eye_aspect_ratio(eye):
    """
    eye = liste de 6 points :
    [p1, p2, p3, p4, p5, p6]
    """

    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear


def eyes_are_open(left_eye, right_eye, threshold=0.22):

    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)

    avg_ear = (left_ear + right_ear) / 2

    return avg_ear > threshold, avg_ear