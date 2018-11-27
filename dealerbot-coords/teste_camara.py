#!/usr/bin/python3
import cv2
import numpy as np
import camera_coords

frame = None
table_size = (9, 7) #numero de casas do xadrez
world_points = None

#apenas para o teste estacionario
projection_matrix = None

#gera coordenadas para atribuir ao xadrez, para ser substituido quando se souber
# o sistema de coordenadas e o tamanho do xadrez
def gerar_pontos_reais():
    total_points = table_size[0]*table_size[1]
    x,y = np.meshgrid(range(table_size[0]),range(table_size[1]))
    return np.hstack((x.reshape(total_points,1),y.reshape(total_points,1),np.zeros((total_points,1)))).astype(np.float32)

#teste da calibração quando se carrega na imagem
def video_window_click(event, x, y, flags, param):
    global projection_matrix
    if event == cv2.EVENT_LBUTTONUP:
        ret, projection_matrix = camera_coords.calibrate(frame, table_size, world_points)
        if ret:
            cv2.imshow('pause', frame)
            cv2.setMouseCallback("pause", pause_window_click)
        else:
            print("Erro a calibrar")
            return

def pause_window_click(event, x, y, flags, param):
    global projection_matrix
    if event == cv2.EVENT_LBUTTONUP:
        coords_unprojected = camera_coords.coords_unproject(projection_matrix, [x, y])
        print("Carregaste em: ", coords_unprojected, " (~ ", coords_unprojected.round().astype(np.int32), ")")



world_points = gerar_pontos_reais()

print("'q' para sair")
cap = cv2.VideoCapture(0)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, table_size)

    # Display the resulting frame
    if ret:
        frame_copy = frame.copy()
        cv2.drawChessboardCorners(frame_copy, table_size, corners, ret)
        cv2.imshow('frame', frame_copy)
    else:
        cv2.imshow('frame', frame)
    
    cv2.setMouseCallback("frame", video_window_click)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()