""" 
Hand Cricket game logics and functionaities.
Run this module to play the game

"""

from cv2 import FlannBasedMatcher
from hand_detector import HandDetector
import cv2
import numpy as np
import random
from menu import Menu

# Game display height and width initialization
screen_width = 1080
screen_height = 720


class Game:
    """
    The main game class

    ...


    Methods
    -------
    countFingers(count, thumb, hand_landmarks)
        Returns the number of detected fingers and if thumb was detected
    """

    def countFingers(count, thumb, hand_landmarks):
        """
        Parameters
        ----------
        count : int
            Number of fingers
        thumb : boolean
            Thumb detection
        hand_landmarks : list
           Coordinates of the fingers

        Returns
        -------
        int, boolean
        Number of fingers and thumb detection 
        """

        # we will get y coordinate of finger-tip and check if it lies above middle landmark of that finger
        # details: https://google.github.io/mediapipe/solutions/hands

        # Right Thumb
        if hand_landmarks[4][3] == "Right" and hand_landmarks[4][1] > hand_landmarks[3][1]:
            count = count+1
            thumb = True
        # Left Thumb
        elif hand_landmarks[4][3] == "Left" and hand_landmarks[4][1] < hand_landmarks[3][1]:
            count = count+1
            thumb = True

        if hand_landmarks[8][2] < hand_landmarks[6][2]:  # Index finger
            count = count+1
        if hand_landmarks[12][2] < hand_landmarks[10][2]:  # Middle finger
            count = count+1
        if hand_landmarks[16][2] < hand_landmarks[14][2]:  # Ring finger
            count = count+1
        if hand_landmarks[20][2] < hand_landmarks[18][2]:  # Little finger
            count = count+1

        return count, thumb

    def start(self, bat, capture):
        """
        Parameters
        ----------
        bat : boolean
            Check if the player is batting 
        capture : object
           Camera frames
        """

        user_score = 0
        pc_score = 0
        pc_score_live = 0
        user_score_live = 0
        guess = 0
        prev = 0
        turn = 0    # This will track number of events. e.g: After batting one increments
        frame = 30  # Number of frames to show the game over screen

        result = None

        count_start = False
        game_over = False
        out = False

        detected_hand = HandDetector(min_detection_confidence=0.8)

        while True:
            status, image = capture.read()
            image = cv2.flip(image, 1)
            hand_landmarks = detected_hand.findHandLandMarks(
                image=image, draw=True)
            count = 0
            thumb = False

            if turn <= 1:
                if(len(hand_landmarks) != 0):

                    count, thumb = self.countFingers(
                        count, thumb, hand_landmarks)

                    if(count == 1 and thumb == True):
                        # Thumb counts as 6 runs
                        count = 6
                        thumb = False

                    if(count == 0):
                        # Initializes move
                        count_start = True

                    if(count_start):

                        if count is not prev:
                            # detects finger changes first
                            prev = count
                            count_start = False
                            if count == 0:
                                guess = 0
                            else:
                                guess = random.randint(1, 6)

                            if(guess == count and count != 0):
                                out = True
                                turn += 1

                                if(bat == True):
                                    bat = False
                                else:
                                    bat = True
                            else:
                                out = False
                                if(bat == True):
                                    user_score += count
                                else:
                                    pc_score += guess

                if(bat == True):
                    cv2.putText(image, "Batting ", ((screen_width//2 - 175), 60),
                                cv2.FONT_HERSHEY_TRIPLEX, 2, (50, 50, 50), 4)

                else:
                    cv2.putText(image, "Bowling ", ((screen_width//2 - 175), 60),
                                cv2.FONT_HERSHEY_TRIPLEX, 2, (250, 50, 200), 4)

                user_score_live = count
                pc_score_live = guess

                if turn == 1:
                    if bat == False:
                        if(pc_score > user_score):
                            turn = 2

                    elif bat == True:
                        if user_score > pc_score:
                            turn = 2

            cv2.putText(image, "Your Total: " + str(user_score),
                        (700, 525), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 10), 2)
            cv2.putText(image, "PC Total: " + str(pc_score),
                        (45, 525), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 10), 2)

            cv2.putText(image, "You ", (screen_width//2+250, screen_height//2-130),
                        cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 25), 4)
            cv2.putText(image, str(user_score_live), (screen_width//2+275, (screen_height//2 - 60)),
                        cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 25), 4)

            cv2.putText(image, "PC ", (45, screen_height//2-130),
                        cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 25), 4)
            cv2.putText(image, str(pc_score_live), (50, (screen_height//2 - 60)),
                        cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 25), 4)

            if out:
                cv2.putText(image, "Out", (screen_width//2-160, 175),
                            cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 255), 7)

            cv2.imshow("Hand Cricket", image)

            if turn > 1:

                if(pc_score > user_score):
                    result = 1
                elif(pc_score == user_score and pc_score != 0):
                    result = 0
                else:
                    result = 2

                cv2.putText(image, "Game Over", (screen_width//2 - 350, 300),
                            cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 255), 10)
                cv2.imshow("Hand Cricket", image)

                hand_landmarks = None
                frame -= 1

                if frame == 0:
                    game_over = True
                    break

            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) == 27:
                # Quit game if q is pressed
                quit()
                break

        if game_over:
            go = Menu(2)  # Mode 2 for play again options
            go.setResult(result)
            game_over = False
            cv2.destroyAllWindows()
            go.show(capture)

        cv2.destroyAllWindows()
        capture.release()
        quit()


# Mode 1 for Toss
menu = Menu(1)
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

menu.show(capture)
