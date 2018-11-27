#!/usr/bin/python3
import cv2
import numpy as np
import camera_coords

## Para implementar:
# Implementar algo para fazer uma calibração inicial. Caso se decida que a camera
#  é fixa basta correr camera_coords.calibrate com uma imagem da camera com o xadrez
# O table_size é o numero de pontos do xadrez, o world_points é o valor em
#  coordenadas reais de cada ponto do xadrez

table_size = (9, 7) #numero de casas do xadrez
calib_image = cv2.imread('teste4.png')

#gera coordenadas para atribuir ao xadrez, para ser substituido quando se souber
# o sistema de coordenadas e o tamanho do xadrez
def gerar_pontos_reais():
    total_points = table_size[0]*table_size[1]
    x,y = np.meshgrid(range(table_size[0]),range(table_size[1]))
    return np.hstack((x.reshape(total_points,1),y.reshape(total_points,1),np.zeros((total_points,1)))).astype(np.float32)

world_points = gerar_pontos_reais()

#calibração da camera:
#devolve um bool a dizer se foi calibrado com sucesso
#gera a matriz que permite converter os pontos reais em pontos da imagem(projetar), e o contrário (desprojetar)
#   permite resolver o sitema em https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
ret, projection_matrix = camera_coords.calibrate(calib_image, table_size, world_points)
if not ret:
    print("Erro a calibrar")
    quit()

print(projection_matrix)




#
#
# Testes
ponto_de_teste = [3, 6]

#projeta o ponto real e descobre o ponto do ecrã correspondente
ponto_de_teste_proj = camera_coords.coords_project(projection_matrix, ponto_de_teste)
print(ponto_de_teste_proj)

#faz o inverso
print(camera_coords.coords_unproject(projection_matrix, ponto_de_teste_proj))


print("carregar no X e depois ctrl-C para fechar")

def img_window_click(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONUP:
		print("Carregaste em: ", camera_coords.coords_unproject(projection_matrix, [x, y]))

ret, corners = cv2.findChessboardCorners(calib_image, table_size)
cv2.drawChessboardCorners(calib_image, table_size, corners, ret)
cv2.imshow('image',calib_image)
cv2.setMouseCallback("image", img_window_click)

#A forma de fechar não está muito boa mas carregar no X e depois ctrl-C funciona
while cv2.getWindowProperty('image', 0) >= 0:
    keyCode = cv2.waitKey(50)
 
cv2.destroyAllWindows()