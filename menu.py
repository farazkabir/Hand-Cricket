import cv2
import mediapipe
import scipy.spatial
import time


class Menu:
    """
    Game menu class

    ...


    Methods
    -------
    setResult(self, result)
        Sets message according to game output
    show(self, capture):
        shows menu options
    """

    def __init__(self, mode=1):
        """
        Parameters
        ----------
        mode : int
            Mode of the menu(Toss or play again)
            1 for toss
            2 for play again
        """

        self.mode = mode

    def setResult(self, result):
        """
        Parameters
        ----------
        result : int
            Win, Loss or Draw
        """

        if result == 1:
            self.msg = "Pc wins"
        elif result == 2:
            self.msg = "You win"
        else:
            self.msg = "Draw"

        self.msg += ". Play Again?"

    def show(self, capture):
        """
        Parameters
        ----------
        capture : object
            Camera frames
        """

        from hand_cricket import Game

        time.sleep(1)

        option_1 = "Bat"
        option_2 = "Ball"
        if(self.mode == 2):
            option_1 = "Yes"
            option_2 = "No"

        draw_module = mediapipe.solutions.drawing_utils
        hands_module = mediapipe.solutions.hands
        distance_module = scipy.spatial.distance
        start_game = True

        frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

        circleCenter_1 = (round(frame_width/2)-100, round(frame_height/2)-50)
        circleCenter_2 = (round(frame_width/2)+100, round(frame_height/2)-50)

        circle_radius = 30
        bat = True

        with hands_module.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1) as detected_hand:

            while (True):

                ret, frame = capture.read()

                if ret == False:
                    continue

                frame = cv2.flip(frame, 1)

                results = detected_hand.process(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                circleColor = (0, 0, 0)

                if results.multi_hand_landmarks != None:

                    normalizedLandmark = results.multi_hand_landmarks[
                        0].landmark[hands_module.HandLandmark.INDEX_FINGER_TIP]
                    pixelCoordinatesLandmark = draw_module._normalized_to_pixel_coordinates(normalizedLandmark.x,
                                                                                            normalizedLandmark.y,
                                                                                            frame_width,
                                                                                            frame_height)

                    cv2.circle(frame, pixelCoordinatesLandmark,
                               2, (255, 0, 0), -1)

                    # Check if finger tip is in the circle
                    if distance_module.euclidean(pixelCoordinatesLandmark, circleCenter_1) < circle_radius:
                        circleColor = (0, 255, 0)
                        cv2.circle(frame, circleCenter_1,
                                   circle_radius, circleColor, -1)
                        cv2.imshow('Toss', frame)

                        bat = True
                        if(self.mode == 2):
                            menu = Menu(1)
                            cv2.destroyAllWindows()
                            menu.show(capture)

                        break

                    elif distance_module.euclidean(pixelCoordinatesLandmark, circleCenter_2) < circle_radius:
                        circleColor = (0, 255, 0)
                        cv2.circle(frame, circleCenter_2,
                                   circle_radius, circleColor, -1)
                        cv2.imshow('Toss', frame)

                        bat = False
                        if(self.mode == 2):
                            start_game = False
                            break

                        break

                    else:
                        circleColor = (0, 0, 255)

                cv2.circle(frame, circleCenter_1,
                           circle_radius, circleColor, -1)
                cv2.circle(frame, circleCenter_2,
                           circle_radius, circleColor, -1)
                if self.mode == 1:
                    cv2.putText(frame, option_1, (int(round(frame_width/2)-200), int(
                        round(frame_height/2)-100)), cv2.FONT_HERSHEY_TRIPLEX, 3, (255, 0, 0), 7)
                    cv2.putText(frame, option_2, (int(round(frame_width/2)+20), int(
                        round(frame_height/2)-100)), cv2.FONT_HERSHEY_TRIPLEX, 3, (255, 0, 0), 7)
                else:
                    cv2.putText(frame, self.msg, (120, int(
                        round(frame_height/2)-200)), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 0), 2)

                    cv2.putText(frame, option_1, (int(round(frame_width//2)-200), int(
                        round(frame_height/2)-100)), cv2.FONT_HERSHEY_TRIPLEX, 3, (255, 0, 0), 7)
                    cv2.putText(frame, option_2, (int(round(frame_width//2)+30), int(
                        round(frame_height/2)-100)), cv2.FONT_HERSHEY_TRIPLEX, 3, (255, 0, 0), 7)

                cv2.imshow('Toss', frame)

                if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) == 27:
                    start_game = False
                    quit()
                    break

        cv2.destroyAllWindows()

        if not start_game:
            capture.release()

        if start_game:
            Game.start(Game, bat, capture)
