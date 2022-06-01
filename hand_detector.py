
""" 
Finger detection utilties
using mediapipe
"""

import mediapipe as mp
import cv2

# Initializations: static code
hands_module = mp.solutions.hands
draw_module = mp.solutions.drawing_utils


class HandDetector:
    """
    Finger detection utility class

    ...


    Methods
    -------
    findHandLandMarks(self, image, hand_number=0, draw=False)
        returns list of finger landmarks detected

    """

    def __init__(self, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # when the mediapipe is first started, it detects the hands. After that it tries to track the hands
        # as detecting is more time consuming than tracking. If the tracking confidence goes down than the
        # specified value then again it switches back to detection
        self.hands = hands_module.Hands(max_num_hands=max_num_hands, min_detection_confidence=min_detection_confidence,
                                        min_tracking_confidence=min_tracking_confidence)

    def findHandLandMarks(self, image, hand_number=0, draw=False):
        """
        Parameters
        ----------
        image : object
            camera frame
        hand_number : int
            number of hands 
        draw : boolean
           landmarks wil be drawn or not

        Returns
        -------
        list
        Detected figer landmark list 
        """

        original_image = image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # mediapipe needs RGB
        results = self.hands.process(image)
        landmark_list = []

        if results.multi_handedness:
            # label gives if hand is left or right
            label = results.multi_handedness[hand_number].classification[0].label
            # account for inversion in webcams
            if label == "Left":
                label = "Right"
            elif label == "Right":
                label = "Left"

        if results.multi_hand_landmarks:  # returns None if hand is not found
            # results.multi_hand_landmarks returns landMarks for all the hands
            hand = results.multi_hand_landmarks[hand_number]

            for id, landMark in enumerate(hand.landmark):
                # landMark holds x,y,z ratios of single landmark
                imgH, imgW, imgC = original_image.shape  # height, width, channel for image
                xPos, yPos = int(landMark.x * imgW), int(landMark.y * imgH)
                landmark_list.append([id, xPos, yPos, label])

            if draw:
                draw_module.draw_landmarks(
                    original_image, hand, hands_module.HAND_CONNECTIONS)

        return landmark_list
