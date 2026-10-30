import numpy as np
import cv2
import glob

# Filas y columnas del tablero (puede variar)
COLS, ROWS = 11,8

# Lote de imagenes
LOTE = 'Lote_6_Molera/'

# Ubicacion de imagenes sin calibrar y directorios para guardar imagenes
FOLDER_DISTORTED = './Images/Not_Calib/' + LOTE
FOLDER_SAVE = './Images/Calib/' + LOTE
FOLDER_DRAWING = './Images/Drawing/' + LOTE

# Resolución de la imagen 
WIDTH_RES = 640     # Webcam
HEIGHT_RES = 480     # Webcam
#WIDTH_RES = 4032   # Iphone 11
#HEIGHT_RES = 3024   # Iphone 11
 
# Se crean COLS*ROWS puntos tridimensionales correspondientes a las esquinas del tablero
objp = np.zeros((COLS*ROWS,3), np.float32)
objp[:,:2] = np.mgrid[0:COLS,0:ROWS].T.reshape(-1,2)
 
# Puntos reales y puntos de imagen
objpoints = [] # 3d 
imgpoints = [] # 2d 
 
# Se obtiene el conjunto de imagenes (cuidado con formato)
imagenes = glob.glob('{}*.jpg'.format(FOLDER_DISTORTED))

# Resolución de la imagen reescalada para guardar (opcional)
# NEW_W = 640
# NEW_H = 480

# Propiedades de la imagen
print( "Resolucion: " + str(WIDTH_RES) + "x" + str(HEIGHT_RES))

# Parámetros y precisión de Subpix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# Obtencion de puntos 2d
i = 1
for fname in imagenes:

    # Se carga la imagen
    img = cv2.imread(fname)
    # img = cv2.resize(imagen, (NEW_W, NEW_H))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Búsqueda de círculos
    ret, corners = cv2.findCirclesGrid(gray, (COLS,ROWS),None)


    # Si se hane encontrado puntos, los añade
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        i += 1


# Se calibra la camara relacionando los puntos 3D y 2D obtenidos. Se obtiene la matriz de parametros,
# vectores de rotacion, de traslacion, y coeficientes de distorsion.
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

# Se obtiene parametros de rectificacion / desdistorsion
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (WIDTH_RES, HEIGHT_RES), 5)

# Se desdistorsionan las imagenes para guardarlas en su carpeta
i = 1
for fname in imagenes:

  # Se desdistorsiona la imagen con los parametros obtenidos anteriormente
  imagen = cv2.imread(fname)
  img = cv2.remap(imagen, mapx, mapy, cv2.INTER_LINEAR)
  
  # Se guarda la imagen
  cv2.imwrite(FOLDER_SAVE + "calibrated" + str(i) + ".jpg",img)
  i += 1

# Se calcula error acumulado y el posterior error de reproyeccion
total_error = 0
for i in range(len(objpoints)):

  # Se obtiene la distancia entre los puntos sacados de forma bruta con los proyectados en la 
  # realidad tras obtenerse la matriz de calibracion. 
  imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
  
  # Se divide el entre el numero de puntos para obtener el error medio de cada imagen
  error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
  total_error += error



total_error = total_error/len(objpoints)
# Se divide el entre el numero de puntos 3D para obtener el error medio de todas las imagenes
print("Error total: " + str(total_error))

# Almacenar los parámetros de la calibración 
model = "matrix_{}x{}".format(COLS,ROWS)
np.savez(model, mtx=mtx, dist=dist)