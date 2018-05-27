# Calibration script
# http://answers.opencv.org/question/98447/camera-calibration-using-charuco-and-python/
import cv2
import pickle

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
board = cv2.aruco.CharucoBoard_create(5, 5, .025, .0125, dictionary)
img = board.draw((200 * 3, 200 * 3))

cv2.imwrite('calibration_board.png', img)

cap = cv2.VideoCapture(0)

allCorners = []
allIds = []
decimator = 0
frame_size = None
for i in range(300):

    ret, frame = cap.read()
    frame_size = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = cv2.aruco.detectMarkers(gray, dictionary)

    if len(res[0]) > 0:
        res2 = cv2.aruco.interpolateCornersCharuco(res[0], res[1], gray, board)
        if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3 and decimator % 3 == 0:
            allCorners.append(res2[1])
            allIds.append(res2[2])

        cv2.aruco.drawDetectedMarkers(gray, res[0], res[1])

    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    decimator += 1

# Calibration fails for lots of reasons. Release the video if we do
try:
    cal = cv2.aruco.calibrateCameraCharuco(allCorners, allIds, board, frame_size, None, None)
    print(cal)
    filename = 'calibration_data'
    outfile = open(filename, 'wb')
    pickle.dump(cal, outfile)
    outfile.close()
except:
    cap.release()

cap.release()
cv2.destroyAllWindows()
