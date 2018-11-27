import cv2
import numpy as np

def calibrate(img, table_size, world_points):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)


    ret, corners = cv2.findChessboardCorners(img, table_size)
    if not ret:
        return False, None

    corners = corners.reshape(-1,2)

    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera([world_points], [corners], gray.shape[::-1],None,None)
    if not ret:
        return False, None

    rotMat, jacobian = cv2.Rodrigues(rvecs[0]) #rotation matrix
    extMat = np.concatenate((rotMat, tvecs[0]), axis=1) #extrinsic matrix
    return True, np.matmul(cameraMatrix, extMat)

#converte coordenadas reais para coordenadas da imagem
#tem algum erro comparado com cv2.projectPoints (penso ser porque não tem em conta
# a distorção da imagem (distCoeffs))
def coords_project(projMat, real_point):
    uv = 1*np.matmul(projMat, np.append(real_point, [0, 1]))
    uv2 = uv/uv[2] #mete a solução da forma [u, v, 1] (faz com que a terceira coordenada seja 1)
    return np.delete(uv2, 2, 0)


#converte coordenadas da imagem para coordenadas reais
def coords_unproject(projMat, screen_point):
    #tira a terceira coluna da matriz, porque se assume que o Z=0 o que permite resolver o sistema
    projMat_without_col = np.delete(projMat, 2, 1)

    #resolve a equação projMat * real_point = screen_point
    real_point = np.linalg.solve(projMat_without_col, np.append(screen_point, 1))

    #real_point é do tipo [x, y, 1] (faz com que a terceira coordenada seja 1)
    return np.delete(real_point/real_point[2], 2, 0)