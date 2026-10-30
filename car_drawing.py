import cv2
import numpy as np
import os

# Filas y columnas del tablero (puede variar)
COLS, ROWS = 9,6       # Patron circular es 11x8 Patrones rectangulares 9x6 y 12x8
# Constantes de colores
AZUL = (255,100,100)
ROJO = (50,50,255)
AZUL_2 = (255,20,20)
# Cantidad de nubes de puntos para generar movimiento en cohces
CAR_POINTS = 50
# Resolucion de la imagen
WIDTH_RES = 640    
HEIGHT_RES = 480     

# Se obtiene matriz de parametros
matrix_file = "./Matrix/" + "matrix_" + str(COLS) + "x" + str(ROWS) + ".npz"
with np.load(matrix_file) as file:
    mtx,dist=[file[i] for i in ['mtx','dist']]

# Se crean COLS*ROWS puntos tridimensionales correspondientes a las esquinas del tablero
objp = np.zeros((COLS*ROWS,3), np.float32)
objp[:,:2] = np.mgrid[0:COLS,0:ROWS].T.reshape(-1,2)


# Dibujo de sistemas de coordenadas
axis_drawing = np.float32([
                             [0,0,0], [3,0,0], [0,3,0], [0,0,3] # X, Y, Z
                        ])    

# Funcion para obtener puntos de coche parametrizado (para obtener movimiento)
x,y,w,h,z = 3,2,5,2,-1           # x_start, y_start, largo, ancho, altura
def get_car(offset):
    axis_car = np.float32([
                            [x+offset,y,0],        [x+w+offset,y,0],  [x+w+offset,y+h,0],    [x+offset,y+h,0],                          # Plancha abajo base    0:4
                            [x+offset,y,-1],       [x+w+offset,y,-1], [x+w+offset,y+h,-1],   [x+offset,y+h,-1],                         # Plancha arriba techo  4:8
                            [x+offset,y,0],        [x+w+offset,y,0],  [x+w+offset,y,-1],     [x+offset,y,-1],                           # Plancha Lateral 1     8:12
                            [x+w+offset,y+h,0],    [x+offset,y+h,0],  [x+offset,y+h,-1],     [x+w+offset,y+h,-1],                       # Plancha Lateral 2     12:16
                            [x+offset,y,0],        [x+offset,y,-1],   [x+offset,y+h,-1],     [x+offset,y+h,0],                          # Plancha Lateral 3     16:20
                            [x+w+offset,y,0],      [x+w+offset,y,-1], [x+offset+w,y+h,-1],   [x+offset+w,y+h,0],                        # Plancha Lateral 4     20:24
                            [(x+offset+w/4),y,z/4], [(x+offset+3*w/4),y,z/4], [(x+offset+w/4),y+h,z/4], [(x+offset+3*w/4),y+h,z/4],     # Ruedas                24:28
                            [x+offset,y,-1],       [x+offset+w-1,y,-1], [x+offset+w-2,y,z-1],   [x+offset+1,y,z-1],                     # Arriba Lat Izq        28:32
                            [x+offset,y+h,-1],      [x+offset+w-1,y+h,-1], [x+offset+w-2,y+h,z-1],   [x+offset+1,y+h,z-1],              # Arriba Lat Der        32:36
                            [x+w-1+offset,y,-1],    [x+offset+w-2,y,z-1], [x+offset+w-2,y+h,z-1],   [x+offset+w-1,y+h,-1],              # Arriba frente         36:40
                            [x+offset,y,-1],        [x+offset+1,y,z-1],   [x+offset+1,y+h,z-1],     [x+offset,y+h,-1],                  # Arriba atrás          40:44
                            [x+w-2+offset,y,z-1],   [x+offset+w-2,y+h,z-1], [x+offset+1,y+h,z-1], [x+offset+1,y,z-1]                    # Techo                 44:48
                        ])    

    return axis_car

# Se obtienen 50 nubes de puntos con la diferencia del offset para generar movimiento
axis_car = []
for i in range(CAR_POINTS):
    axis_car.append(get_car((i-(CAR_POINTS/2))/6))  #El coche se mueve 0,16 unidades en cada iteracion
 
# Funcion para dibujar coche
def draw_car(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    img = cv2.drawContours(img, [imgpts[0:4]],-1,ROJO,thickness=cv2.FILLED)      
    img = cv2.drawContours(img, [imgpts[8:12]],-1,ROJO,thickness=cv2.FILLED)
    img = cv2.drawContours(img, [imgpts[12:16]],-1,ROJO,thickness=cv2.FILLED)
    img = cv2.drawContours(img, [imgpts[16:20]],-1,ROJO,thickness=cv2.FILLED)
    img = cv2.drawContours(img, [imgpts[20:24]],-1,ROJO,thickness=cv2.FILLED)
    img = cv2.circle(img, tuple(imgpts[24]), 12, (255,255,255), -1)                     # Ruedas
    img = cv2.circle(img, tuple(imgpts[25]), 12, (255,255,255), -1)
    img = cv2.circle(img, tuple(imgpts[26]), 12, (255,255,255), -1)
    img = cv2.circle(img, tuple(imgpts[27]), 12, (255,255,255), -1)
    img = cv2.drawContours(img, [imgpts[4:8]],-1,ROJO,thickness=cv2.FILLED)        # Techo
    img = cv2.drawContours(img, [imgpts[28:32]],-1,AZUL_2,thickness=cv2.FILLED)    # Lateral Izq
    img = cv2.drawContours(img, [imgpts[32:36]],-1,AZUL_2,thickness=cv2.FILLED)    # Lateral Der
    img = cv2.drawContours(img, [imgpts[36:40]],-1,AZUL,thickness=cv2.FILLED)      # Arriba frente
    img = cv2.drawContours(img, [imgpts[40:44]],-1,AZUL,thickness=cv2.FILLED)      # Arriba atras
    img = cv2.drawContours(img, [imgpts[44:48]],-1,ROJO,thickness=cv2.FILLED)      # Techo 2

    return img

# Resolución de la imagen reescalada para su impresion (opcional)
NEW_W = 640
NEW_H = 480

# Si se desease rescalar la imagen, estas dos funciones obtienen de nuevo la matriz de parametros intrínsecos
#newMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (WIDTH_RES, HEIGHT_RES), 1, (NEW_W, NEW_H))
#mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newMtx, (NEW_W, NEW_H), 5)

# Parámetro para contar frames y mover el objeto e inicio de cámara
index = 0

cam=cv2.VideoCapture(0)
cv2.namedWindow('Dibujo', cv2.WINDOW_NORMAL)

while True:

    # Se invierte imagen en eje Y y se obtiene nuevas dimensiones tras reescalado (opcional)
    posible,frame=cam.read()
    frame = cv2.flip(frame,1)

    if posible==False:
        break
    
    # Se obtienen las esquinas del tablero
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret,corners = cv2.findChessboardCorners(gray,(COLS,ROWS),None)

    #Si el patrón es circular usar este
    #ret, corners = cv2.findCirclesGrid(gray, (COLS,ROWS),None)


    if ret == True:

        # Se estima la pose del objeto a partir de puntos 3D-2D con el algoritmo de Ransac.
        _,rvec,tvec,_=cv2.solvePnPRansac(objp,corners,mtx,dist) 

        # Se obtiene la nube de puntos correspondiente al coche
        imgpts,_=cv2.projectPoints(axis_car[index],rvec,tvec,mtx,dist)

        # Si se llega hasta el final:
        if(index==CAR_POINTS-1):
            index=0

        # Sino, se aumenta indice y se dibuja coche
        index += 1
        frame = draw_car(frame,corners,imgpts)

    cv2.imshow('Dibujo',frame)
    if cv2.waitKey(1) & 0xFF == ord('E'):
        break

cv2.destroyAllWindows()
cam.release()
