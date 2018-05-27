import numpy as np
import cv2
import cv2.aruco
import datetime
import math
import pickle
import yaml
from socket import *


class Menu(object):
    def __init__(self, options):
        self.option_count = len(options)
        self.grid_size = int(math.ceil(math.sqrt(self.option_count)))
        self.resolution = 200
        self.selected = 0
        self.selected_change = 0
        self.image = None
        self.filled = 0
        self.options = options
        self.chosen = None
        self.chosen_option = None

    # returns the holo-interface frame to be overlayed
    def get_image(self):
        line_size = 10
        self.image = np.zeros([self.grid_size * self.resolution, self.grid_size * self.resolution, 3], dtype="uint8")
        # grid
        for i in range(0, self.grid_size + 1):
            cv2.line(self.image, (i * self.resolution, 0), (i * self.resolution, self.image.shape[0]), (255, 0, 0),
                     line_size)
        for i in range(0, self.grid_size + 1):
            cv2.line(self.image, (0, i * self.resolution), (self.image.shape[1], i * self.resolution), (255, 0, 0),
                     line_size)
        # highlight
        cv2.rectangle(self.image, (
            self.selected % self.grid_size * self.resolution, self.selected // self.grid_size * self.resolution), (
                          (self.selected % self.grid_size + 1) * self.resolution,
                          (self.selected // self.grid_size + 1) * self.resolution), (255, 0, 0), thickness=-1)
        # selection progress bar
        cv2.rectangle(self.image, (
            self.selected % self.grid_size * self.resolution, self.selected // self.grid_size * self.resolution), (
                          (self.selected % self.grid_size) * self.resolution + int(self.resolution * self.filled),
                          (self.selected // self.grid_size + 1) * self.resolution),
                      (0, 255, 0) if self.chosen == self.selected else (255, 255, 0), thickness=-1)
        # titles
        for ind, (option_id, option_text) in enumerate(self.options):
            for n, line in enumerate(reversed(option_text.split(" "))):
                cv2.putText(self.image, line,
                            ((ind % self.grid_size) * self.resolution, (ind // self.grid_size + 1) * self.resolution - (
                                n * int(5.0 / (1 + self.grid_size)) * 50)), 0,
                            int(5.0 / (1 + self.grid_size)),
                            (255, 255, 255), 2)

    def update_menu(self):
        if self.selected_change is not self.selected:
            self.filled = 0.02
            self.selected_change = self.selected
        elif self.selected is not self.chosen:
            self.filled += 0.02
            if self.filled >= 1:
                self.filled = 1
                self.chosen_option = (self.options[self.selected][0])
                self.chosen = self.selected

    def reset(self):
        self.selected = 0
        self.selected_change = -1
        self.chosen = None
        self.chosen_option = None
        self.filled = 0


# stores a collection of menus, handles swapping them on selection
class UI(object):
    def __init__(self, ui_frame, ui_event_handler):
        self.pending_moves = []
        self.frame = ui_frame
        self.ui_overlay = np.zeros([ui_frame.shape[0], ui_frame.shape[1], 3], dtype="uint8")
        self.menu = None
        self.inflated_menus = {}
        self.event_handler = ui_event_handler

    def send_move_to_menu(self):
        for marker_id, marker in frame_manager.markers.items():
            marker_points = marker.rect
            center = np.repeat([np.average(marker_points, axis=0)], 4, axis=0)
            marker_points = center + (marker_points - center) * 4
            if self.pending_moves:
                move = self.pending_moves.pop()
                if move == "LEFT" and self.menu.selected > 0 and self.menu.selected % self.menu.grid_size != 0:
                    self.menu.selected -= 1
                if move == "RIGHT" and self.menu.selected < self.menu.option_count - 1 and (
                            self.menu.selected + 1) % self.menu.grid_size != 0:
                    self.menu.selected += 1
                if move == "UP" and self.menu.selected >= self.menu.grid_size:
                    self.menu.selected -= self.menu.grid_size
                if move == "DOWN" and self.menu.selected + self.menu.grid_size <= self.menu.option_count - 1:
                    self.menu.selected += self.menu.grid_size
            self.show_image(marker_points)

    # When the menu image is ready, we align it to the 3d marker and add it to the ui_overlay
    def show_image(self, marker_points):
        self.menu.get_image()
        (remote_width, remote_height, _) = self.menu.image.shape

        pts1 = np.float32([[0, 0], [remote_height, 0], [remote_height, remote_width], [0, remote_width]])
        pts2 = np.float32([marker_points[0], marker_points[1], marker_points[2], marker_points[3]])
        transform_matrix = cv2.getPerspectiveTransform(pts1, pts2)
        self.ui_overlay += cv2.warpPerspective(self.menu.image, transform_matrix,
                                               (frame.shape[1], frame.shape[0]))

    def get_ui_image(self):
        self.ui_overlay = np.zeros([frame.shape[0], frame.shape[1], 3], dtype="uint8")
        self.send_move_to_menu()
        self.menu.update_menu()
        self.act_on_menu_choice()

    def act_on_menu_choice(self):
        if self.menu.chosen_option is None:
            return
        # whether to send action or go to submenu
        if self.menu.chosen_option[0] == "switchtomenu":
            self.menu = self.inflated_menus[self.menu.chosen_option[1]]
            self.menu.reset()
        elif self.menu.chosen_option[0] == "runaction":
            self.event_handler.act_on_event(self.menu.chosen_option[1])
        self.menu.chosen_option = None

    # prepare menu objects for all submenus, no dynamic loading implemented
    def inflate_menu_structure(self, menu_dict):
        main_options = []
        for room in menu_dict:
            k, v = list(room.items())[0]
            main_options.append((("switchtomenu", k), k))
            room_options = []
            for room_id, action in v.items():
                room_options.append((("switchtomenu", room_id), action))
                on_off_options = [(("runaction", "on " + room_id), "On"), (("runaction", "off " + room_id), "Off"),
                                  (("switchtomenu", k), "Powrot")]
                self.inflated_menus[room_id] = Menu(on_off_options)

            room_options.append((("switchtomenu", "main"), "Powrot"))
            self.inflated_menus[k] = Menu(room_options)
        self.inflated_menus["main"] = Menu(main_options)
        self.menu = self.inflated_menus["main"]


# Main class, handles marker detection, showing ui, and sending events to handler
class FrameManager(object):
    def __init__(self, fm_frame, fm_event_handler):
        self.markers = {}
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        self.frame = fm_frame
        self.ui = UI(fm_frame, fm_event_handler)
        self.event_handler = fm_event_handler
        self.ui.inflate_menu_structure(fm_event_handler.event_structure)
        self.calibration = lambda: None
        # read picked calibration data created using Calibration.py
        with open("calibration_data", "rb") as calibration_data:
            self.calibration.retval, self.calibration.cameraMatrix, self.calibration.distCoeffs, self.calibration.rvecs, self.calibration.tvecs = pickle.load(
                calibration_data)
        self.ui.inflate_menu_structure(fm_event_handler.event_structure)

    # util function to save marker to a jpg file
    def save_marker_from_id(self, marker_id, filename):
        img = cv2.aruco.drawMarker(self.dictionary, marker_id, 1000)
        cv2.imwrite(filename, img)

    def update_from_frame(self, current_frame):
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        points, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary)
        self.update_markers(points, ids)
        self.ui.get_ui_image()
        self.frame = current_frame

    # check whether some markers expired, update all markers 3d transformation, get ui nav move and pass it to the ui
    def update_markers(self, points, ids):
        if ids is None:
            ids = []
        for ([marker_points], [marker_id]) in zip(points, ids):

            if marker_id not in self.markers:
                self.markers[marker_id] = Marker(marker_id, self)
            self.markers[marker_id].set_transform_from_rectangle(marker_points)
            moved = self.markers[marker_id].check_moved()
            if moved is not None:
                self.ui.pending_moves.append(moved)
        self.markers = {k: v for k, v in self.markers.items() if
                        (datetime.datetime.now() - v.last_seen) < datetime.timedelta(seconds=0.1)}
        if len(self.markers) is 0:
            self.ui.menu.reset()

    # Adds the camera recorded frame to the UI image. Could use different blend modes,
    # but simple adding looks like a holo image
    def get_current_display_frame(self):
        return cv2.add(self.frame, self.ui.ui_overlay)


# Stores data about a single marker entity, differentiated by its id
class Marker(object):
    def __init__(self, id, curr_frame_manager):
        self.transform = None
        self.prev_transform = None
        self.transform_change = None
        self.rect = None
        self.id = id
        self.cool_down = datetime.datetime.now()
        self.last_seen = datetime.datetime.now()
        self.frame_manager = curr_frame_manager

    # get 3d from a set of 4 points using the camera calibration data
    def set_transform_from_rectangle(self, marker_points):
        # https://github.com/jerryhouuu/Face-Yaw-Roll-Pitch-from-Pose-Estimation-using-OpenCV/blob/master/pose_estimation.py
        rot, trans, _ = cv2.aruco.estimatePoseSingleMarkers([marker_points], 0.06,
                                                            self.frame_manager.calibration.cameraMatrix,
                                                            self.frame_manager.calibration.distCoeffs)
        rvec_matrix = cv2.Rodrigues(rot)[0]
        proj_matrix = np.hstack((rvec_matrix, np.transpose(trans[0])))
        euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

        pitch, yaw, roll = [math.radians(_) for _ in euler_angles]

        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))
        self.transform = [pitch, roll, yaw]
        self.last_seen = datetime.datetime.now()
        self.rect = marker_points

    def check_moved(self):
        # the bigger angle the faster we move trough the interface
        move, angle = self.get_direction()
        if (datetime.datetime.now() - self.cool_down) > + datetime.timedelta(
                milliseconds=max(2000.0 * math.pow(1.2, (-(angle - 5) * 0.4)), 2)):
            if move is not None:
                self.cool_down = datetime.datetime.now()
                return move

    # euler angle rotation to one of 4 directions
    def get_direction(self):
        point_dir = [self.transform[0], self.transform[2]]
        max_angle = np.max(np.abs(point_dir))
        if max_angle > 15:
            maximum_indice = np.argmax(np.abs(point_dir))

            if maximum_indice == 0 and point_dir[0] > 0:
                return "UP", max_angle
            elif maximum_indice == 0 and point_dir[1] < 0:
                return "DOWN", max_angle
            elif maximum_indice == 1 and point_dir[1] < 0:
                return "RIGHT", max_angle
            elif maximum_indice == 1 and point_dir[1] > 0:
                return "LEFT", max_angle

        return None, 0


# reads events from .yaml file, calls them trough the socket on ui call
class EventHandler(object):
    def __init__(self):
        self.event_structure = None
        self.socket = None

    # parse yaml to dict
    def get_events_from_yaml(self, eh_events):
        for new, org in zip(['a', 'c', 'e', 'l', 'n', 'o', 's', 'z', 'z'],
                            ['ą', 'ć', 'ę', 'ł', 'ń', 'ó', 'ś', 'ż', 'ź']):
            eh_events = eh_events.replace(org, new)

        event_dict = yaml.load(eh_events)
        self.event_structure = event_dict
        pass

    # send action to device
    def act_on_event(self, chosen_option):
        print(chosen_option)
        self.socket = socket(AF_INET, SOCK_DGRAM)
        self.socket.setsockopt(SOL_SOCKET, SO_BROADCAST, 1)
        self.socket.sendto(bytes(chosen_option + '\n', encoding="utf8"), ('<broadcast>', 2018))
        self.socket.close()


if __name__ == "__main__":
    # open camera feed
    capture = cv2.VideoCapture(0)
    frame_manager = None
    event_handler = EventHandler()
    # read config
    with open("events.yaml", "r", encoding="utf8") as events:
        event_handler.get_events_from_yaml(events.read())
    while True: # for every frame
        ret, frame = capture.read()
        if frame_manager is None:
            frame_manager = FrameManager(frame, event_handler)
        frame_manager.update_from_frame(np.flip(frame, 1))
        # we flip it for allowing better coordination of movement, by def cameras are made to make us look good,
        # they flip the image, so it looks like what we see in a normal mirror, but this is bad for controlling the ui
        cv2.imshow('frame', frame_manager.get_current_display_frame())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()
