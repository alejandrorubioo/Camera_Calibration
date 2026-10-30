import numpy as np
import cv2
import glob

# Filas y columnas del Images (puede variar)
COLS, ROWS = 9,6   #11x7 - 9x6

# Lote de imagenes
LOTE = 'Lote_2_50images/'

# Ubicacion de imagenes sin calibrar y directorios para guardar imagenes
FOLDER_DISTORTED = './Images/Not_Calib/' + LOTE
FOLDER_SAVE = './Images/Calib/' + LOTE
FOLDER_DRAWING = './Images/Drawing/' + LOTE

# Resolución de la imagen 
WIDTH_RES = 640     # Webcam
HEIGHT_RES = 480     # Webcam
# WIDTH_RES = 4032   # Iphone 11
# HEIGHT_RES = 3024   # Iphone 11
 
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
print(" ")
print( "Resolucion: " + str(WIDTH_RES) + "x" + str(HEIGHT_RES))
print(" ")


# Obtención de puntos 2D
i = 1
for fname in imagenes:

  # Se carga la imagen
  img = cv2.imread(fname)
  # img = cv2.resize(imagen, (NEW_W, NEW_H))
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  # Búsqueda de esquinas
  ret, corners = cv2.findChessboardCorners(gray, (COLS,ROWS),None)

  # Si se hane encontrado puntos, los añade
  if ret == True:
    objpoints.append(objp)

    # Se calculan los gradientes entorno a las esquinas, se obtienen ecuaciones y 
    # se ajusta la precisión de las esquinas
    vec = (7,7)
    zero_vec = (-1,-1) 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # Parametros
    corners_upd = cv2.cornerSubPix(gray,corners,vec,zero_vec,criteria)  
    imgpoints.append(corners_upd)

    # Se muestran las esquinas en el tablero
    img = cv2.drawChessboardCorners(img, (COLS,ROWS), corners_upd, ret)
    cv2.imwrite(FOLDER_DRAWING + "tablero" + str(i) + ".png",img)
    cv2.imshow('img',img)
    cv2.waitKey(12)        # Se espera un pequeño tiempo y pasa a la siguiente imagen
    i += 1


# Se calibra la camara relacionando los puntos 3D y 2D obtenidos. Se obtiene la matriz de parametros,
# vectores de rotacion, de traslacion, y coeficientes de distorsion.
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print("Matriz de parámetros intrínsecos")
print(mtx)
print(" ")

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
print(" ")

# La matriz representa en su diagonal las distancias focales en píexeles
focal_length_x, focal_length_y = mtx[0,0], mtx[1,1]
focal_length = np.sqrt(focal_length_x**2 + focal_length_y**2)
print ("Distancia focal media:", focal_length)
print ("Error normalizado: ", total_error/focal_length)
print(" ")

# Almacenar los parámetros de la calibración 
model = "matrix_{}x{}".format(COLS,ROWS)
np.savez(model, mtx=mtx, dist=dist)