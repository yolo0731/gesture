from .Landmark import Landmark

class Finger:
    '''
    Creates a finger object with 4 landmarks
    '''

    def __init__(self):
        self.landmarks = [Landmark(0, 0, 0, 0)] * 4
