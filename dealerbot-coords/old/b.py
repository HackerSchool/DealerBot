#https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
#http://www.dmi.unict.it/~furnari/teaching/CV1617/lab1/
import cv2
import numpy as np
import math

world_points = []
corners = []
test_point = 10
filename = 'c.png'
table_size = (7,6) #numero de casas do xadrez

def calibrate(img, table_size):
    global world_points, corners
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)


    ret, corners = cv2.findChessboardCorners(img, table_size)
    if not ret:
        print("Error")
        quit()

    corners = corners.reshape(-1,2)

    #desenhar
    if False:
        cv2.drawChessboardCorners(img, table_size, corners, ret) 
        cv2.imshow('image',img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    total_points = table_size[0]*table_size[1]
    x,y = np.meshgrid(range(table_size[0]),range(table_size[1]))
    world_points = np.hstack((x.reshape(total_points,1),y.reshape(total_points,1),np.zeros((total_points,1)))).astype(np.float32)

    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera([world_points], [corners], gray.shape[::-1],None,None)
    rotMat, jacobian = cv2.Rodrigues(rvecs[0]) #rotation matrix
    extMat = np.concatenate((rotMat, tvecs[0]), axis=1) #extrinsic matrix
    return cameraMatrix, extMat

#convert real coordinates to screen coords
#tem algum erro comparado com cv2.projectPoints
def my_project(projMat, real_point):
    uv = 1*np.matmul(projMat, np.append(real_point.tolist(), 1))
    uv2 = uv/uv[2]
    return np.delete(uv2, 2, 0)

def my_unproject(projMat, screen_point):
    projMat_without_col = np.delete(projMat, 2, 1) # tira a terceira coluna da matriz, porque se assume que o Z=0
    sol = np.linalg.solve(projMat_without_col, np.append(screen_point, 1))
    return np.delete(sol/sol[2], 2, 0)

cameraMatrix, extMat = calibrate(cv2.imread(filename), table_size)
#matriz A[R|t] do sistema da https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
calibMatrix = np.matmul(cameraMatrix, extMat)


#imprime valores de teste
print("Real: ", corners[test_point])
print("My proj: ", my_project(calibMatrix, world_points[test_point]))

print("Real: ", world_points[test_point])
print("My unproj", my_unproject(calibMatrix, corners[test_point]))

print(corners[test_point])