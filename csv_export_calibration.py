import numpy as np
import cv2
import glob
from csv import writer
import os 

# Filas y columnas del tablero (puede variar)
COLS, ROWS = 9,6 #11x7 - 9x6

# Lote de imagenes
LOTE = 'Lote_8_Batch/'

# Ubicacion de imagenes sin calibrar y directorios para guardar imagenes
FOLDER_DISTORTED = './Images/Not_Calib/' + LOTE
FOLDER_SAVE = './Images/Calib/' + LOTE
FOLDER_DRAWING = './Images/Drawing/' + LOTE

# Numero de imagenes
IMAGE_NUM = sum([len(files) for r, d, files in os.walk(FOLDER_DISTORTED)])
print ("Imagenes: ", IMAGE_NUM)

# Resolución de la imagen 
WIDTH_RES = 640      # Webcam
HEIGHT_RES = 480     # Webcam
#WIDTH_RES = 4032    # Iphone 11
#HEIGHT_RES = 3024   # Iphone 11
 
# Se crean COLS*ROWS puntos tridimensionales correspondientes a las esquinas del tablero
objp = np.zeros((COLS*ROWS,3), np.float32)
objp[:,:2] = np.mgrid[0:COLS,0:ROWS].T.reshape(-1,2)
 
# Puntos reales y puntos de imagen
objpoints = [] # 3d 
imgpoints = [] # 2d 
 
# Se imprime cabecera en archivo csv
with open('error_2.csv', 'w') as f_object:
  writer_object = writer(f_object)
  writer_object.writerow(["Imágenes", "Error"])
  f_object.close()

# Se genera por numero de imagenes ascendente
for k in range(1,IMAGE_NUM+3, 3):   # Paso: 3 imagenes en cada analisis
  # Lista para guardar imagenes
  imagenes = []
  index = 0
  
  # Iterate directory
  for path in os.listdir(FOLDER_DISTORTED):
    index += 1
    # Se añade archivo a lista
    if os.path.isfile(os.path.join(FOLDER_DISTORTED, path)):
      imagenes.append(FOLDER_DISTORTED + path)
    # Si se alcanza el valor de k, salta del bucle.
    if index == k:
      break;
  
  # Parámetros y precisión de Subpix
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
  
  # Relacion de puntos 3D con puntos 2D
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
      vecindad = (7,7)
      zero_vec = (-1,-1) 
      corners_upd = cv2.cornerSubPix(gray,corners,vecindad,zero_vec,criteria)  
      imgpoints.append(corners_upd)

      # Se muestran las esquinas en el tablero
      img = cv2.drawChessboardCorners(img, (COLS,ROWS), corners_upd, ret)
      cv2.imshow('img',img)
      cv2.waitKey(5)        # Se espera un pequeño tiempo y pasa a la siguiente imagen
      i += 1


  # Se calibra la camara relacionando los puntos 3D y 2D obtenidos. Se obtiene la matriz de parametros,
  # vectores de rotacion, de traslacion, y coeficientes de distorsion.
  ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

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
  print("Error total: " + str(total_error) + "/ Imagenes: " + str(k))

  # Se crea fichero csv y se añaden lineas
  with open('error_2.csv', 'a') as f_object:
    writer_object = writer(f_object)
    writer_object.writerow([k, total_error])
    f_object.close()
