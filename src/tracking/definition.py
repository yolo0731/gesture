import numpy as np

def get_angle(v1, v2):
    angle = np.dot(v1, v2) / (np.sqrt(np.sum(v1*v1)) * np.sqrt(np.sum(v2*v2)))
    angle = np.arccos(angle) / 3.14 * 180
    return angle


def get_str_guester(up_fingers):
    if len(up_fingers) == 1 and up_fingers[0] == 8:
        number = "1"
    elif len(up_fingers) == 2 and up_fingers[0] == 8 and up_fingers[1] == 12:
        number = "2"
    elif len(up_fingers) == 3 and up_fingers[0] == 8 and up_fingers[1] == 12 and up_fingers[2] == 16:
        number = "6"
    elif len(up_fingers) == 3 and up_fingers[0] == 8 and up_fingers[1] == 16 and up_fingers[2] == 20:
        number = "8"
    elif len(up_fingers) == 3 and up_fingers[0] == 4 and up_fingers[1] == 8 and up_fingers[2] == 12:
        number = "3"
    elif len(up_fingers) == 3 and up_fingers[0] == 8 and up_fingers[1] == 12 and up_fingers[2] == 20:
        number = "7"
    elif len(up_fingers) == 4 and up_fingers[0] == 8 and up_fingers[1] == 12 and up_fingers[2] == 16 and up_fingers[3] == 20:
        number = "4"
    elif len(up_fingers) == 5:
        number = "5"
    elif len(up_fingers) == 0:
        number = "0"
    elif len(up_fingers) == 3 and up_fingers[0] == 12 and up_fingers[1] == 16 and up_fingers[2] == 20:
        number = "9"
    else:
        number = " "
    return number

