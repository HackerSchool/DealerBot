import cv2
import numpy as np
import math
from enum import Enum


class CalibrationStep(Enum):
	ASKING_FOR_POINTS = 1
	DONE = 2

corners = []
img = None
gray = None
obj_points = np.matrix([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 2.0, 0.0], [1.0, 1.0, 0.0]], np.float32) # list of points in real coords
img_points = [] # list of points in image coords

point_idx_ask = 0 # o indice do ponto que se está a pedir
calibration_step = CalibrationStep.ASKING_FOR_POINTS

def ask_for_point():
	print("Select point " + str(point_idx_ask))
	print(obj_points[point_idx_ask])

def point_selected(point):
	global point_idx_ask, calibration_step, img_points

	img_points.append(point)
	point_idx_ask += 1

	if point_idx_ask == len(obj_points):
		calibrate_camera()
		calibration_step = CalibrationStep.DONE
		print("DONE!")
	else:
		ask_for_point()

def find_closest_corner(x, y):
	closestIndex = -1
	closestDist = -1
	for i, corner in enumerate(corners):
		dist = math.sqrt((corner[0]-x)**2 + (corner[1]-y)**2)
		if closestIndex == -1 or closestDist > dist:
			closestIndex = i
			closestDist = dist
	return closestIndex

def img_window_click(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONUP:
		print(calibration_step)
		if calibration_step == CalibrationStep.ASKING_FOR_POINTS:
			
			clickedCornerIdx = find_closest_corner(x, y)
			print(corners[clickedCornerIdx])

			cv2.circle(img, (corners[clickedCornerIdx][0], corners[clickedCornerIdx][1]), 3, (0,0,255), -1)
			cv2.imshow('image',img)

			point_selected(corners[clickedCornerIdx])

def find_corners():
	global gray
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	#find Harris corners
	gray = np.float32(gray)

	dst = cv2.cornerHarris(gray,2,3,0.04)
	dst = cv2.dilate(dst,None)
	ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
	dst = np.uint8(dst)

	#find centroids
	ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

	#define the criteria to stop and refine the corners
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
	return cv2.cornerSubPix(gray,np.float32(centroids),(7,7),(-1,-1),criteria)	

def calibrate_camera():
	#print(np.array([np.array(obj_points, np.float32)]).asmatrix())
	img_points_np = np.matrix(img_points, np.float32)
	print(img_points_np)

	ret, cameraMatrix, distCoeffs, rvec, tvec = cv2.calibrateCamera([obj_points], [img_points_np], gray.shape[::-1],None,None)
	print(ret)
	print(cameraMatrix)
	print(rvec)
	print(tvec)

	print("\n\n")
	rotMatrix, jacobian = cv2.Rodrigues(rvec[0])

	rtMatrix = np.concatenate((rotMatrix, tvec[0]), axis=1)

	print("Camera matrix:")
	print(cameraMatrix.tolist())
	print("R|t matrix:")
	print(rtMatrix.tolist())


filename = 'a.png'
img = cv2.imread(filename)

corners = find_corners()
print(obj_points)
#here u can get corners
print (corners[0])
print (len(corners))

#Now draw them
#res = np.hstack((centroids,corners)) 
#res = np.int0(res) 
#img[res[:,1],res[:,0]]=[0,0,255] 
#img[res[:,3],res[:,2]] = [0,255,0]
cv2.imshow('image',img)
cv2.setMouseCallback("image", img_window_click)
ask_for_point()

cv2.waitKey(0)
cv2.destroyAllWindows()