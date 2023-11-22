#Import libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.morphology as morph

###-----------------------Lineal ilumination / intensity correction-----------------###

# Definir algorithmo white-patch para crear invariancia a los cambios del color de la luz
def white_patch_correction(image):
    # Convertir la imagen a flotante para cálculos precisos
    image_float = image.astype(float)

    max_canal0 = image_float[:,:,0].max()
    max_canal1 = image_float[:,:,1].max()
    max_canal2 = image_float[:,:,2].max()
    
    # Create image copy
    img_copy = image.copy().astype(float)
    img_copy[:,:,0] = 255*img_copy[:,:,0]/max_canal0
    img_copy[:,:,1] = 255*img_copy[:,:,1]/max_canal1
    img_copy[:,:,2] = 255*img_copy[:,:,2]/max_canal2
    
    # Asegurarse de que los valores estén en el rango 0-255
    corrected_image = np.clip(img_copy, 0, 255)

    # Convertir de nuevo a tipo de datos entero
    corrected_image = corrected_image.astype(np.uint8)

    return corrected_image


#Definir la función expansión del histograma cuyo parámetro es una imagen
def histogram_expansion(img):
    
    #Crear matriz de ceros del tamaño de la imagen y tipo de dato flotante
    res = np.zeros([img.shape[0], img.shape[1]], dtype=np.float32)
    
    #Extraer el mínimo y el máximo del conjunto de datos
    m = float(np.min(img))
    M = float(np.max(img))+0.0001
    #Aplicar la función de expansión(normalización) y asegurar datos uint8
    res = (img-m)*255.0/(M-m)
    res = np.clip(res, 0, 255).astype(np.uint8)
    
    return res

###----------------------Color space conversion-----------------###

# Define Color spaces
MODES = {'RGB': cv2.COLOR_BGR2RGB,
         'GRAY': cv2.COLOR_BGR2GRAY,
         'YUV': cv2.COLOR_BGR2YUV, 
         'HSV': cv2.COLOR_BGR2HSV, 
         'LAB': cv2.COLOR_BGR2LAB, 
         'HLS': cv2.COLOR_BGR2HLS, 
         'XYZ': cv2.COLOR_BGR2XYZ, 
         'YCRCB':cv2.COLOR_BGR2YCrCb, 
         'CMY': None, 
         'YIQ': None}

# Define Color space transformation RGB->YIQ
def convert_to_yiq(img): 
    M = np.array([[0.299, 0.587, 0.114], 
                [0.596, -0.274, -0.322], 
                [0.212, -0.523, 0.311]])
    YIQ = np.matmul(img, M)
    return YIQ

# Define function to convert color space
def convert_image(img, color_space='RGB'):
    color_space = color_space.upper()
    if (space:= MODES[color_space]):
        img_out = cv2.cvtColor(img, space)
    elif color_space == 'CMY':
        img_out = 255 - img
    elif color_space == 'YIQ':
        img_out = convert_to_yiq(img)
    else:
        raise Exception('INPUT ERROR: Espacio de color incorrecto')
    return img_out

def channels(img_in): 
    for mode in MODES :
        # Lectura de imagen y extracción de canales
        if mode == 'GRAY': continue
        img = convert_image(img_in, mode)
        c0 = img[:,:,0]
        c1 = img[:,:,1]
        c2 = img[:,:,2]

        # Crear el gráfico
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(30, 10))
        fig.suptitle(f"Canales {mode}", fontsize=15, y=1.003)

        if mode == 'YCRCB' :
            ax1.set_title(f'Canal {mode[0]}')
            ax2.set_title(f'Canal {mode[1:3]}')
            ax3.set_title(f'Canal {mode[3:]}')
        else :
            ax1.set_title(f'Canal {mode[0]}')
            ax2.set_title(f'Canal {mode[1]}')
            ax3.set_title(f'Canal {mode[2]}')

        ax1.imshow(c0)
        ax2.imshow(c1)
        ax3.imshow(c2)



###-----------------------Non-lineal image correction-----------------###

#Definir la función de transformación de la imagen (corrección gamma)
def gamma_correction(img, a=1.0, gamma=2): 
    #Crear copia de la imagen tipo flotante dada la normalización
    img_copy = img.copy().astype(np.float32)/255.0
    #La función corrección gamma es de la forma ax^gamma, donde x es la imagen de entrada
    res_gamma = cv2.pow(img_copy,gamma)
    res = cv2.multiply(res_gamma, a)
    #Asegurar que la los datos queden entre 0 y 255 y sean uint8
    res[res<0] = 0
    res = res*255.0
    res[res>255] = 255
    res = res.astype(np.uint8)
    return res

###---------------------Masking functions-------------------###

# Función para crear una mascara negada: True -> False; False -> True
def inverse_mask(mask):
    mask = np.array(mask, dtype=bool)
    res = np.ones_like(mask, dtype=bool)
    res[mask] = False
    return res

# Función para segmentación de la imagen original a través de applicación de la mascara
def apply_mask(img, mask):
    canal_1 = img[:,:,0]
    canal_2 = img[:,:,1]
    canal_3 = img[:,:,2]
    canal_1[mask] = 0
    canal_2[mask] = 0
    canal_3[mask] = 0
    img_return = np.zeros(img.shape, np.uint8)
    img_return[:,:,0] = canal_1
    img_return[:,:,1] = canal_2
    img_return[:,:,2] = canal_3
    return img_return

def segment_image(img, mask):
    # Crear la mascara negada
    mascara_neg = inverse_mask(mask)
    # Segmentación de la imagen origional con las mascaras
    img_segmented = apply_mask(img.copy(), mascara_neg)
    return img_segmented



###--------------------------- Preprocessing functions---------------------###

# Preprocessing for license plates that are still yellow 
def preprocessing_1_segmentation(img, threshold=45): 
    #Apply White-Patch algorithm to image
    img_wp = white_patch_correction(img)

    # Convert image to HSV and extract channel S
    s_de_hsv = convert_image(img_wp, 'HSV')[:,:,1]

    # Apply Gamma transformation
    # img_after_gamma = gamma_correction(s_de_hsv, a=1.0, gamma=1.5) #Disabled as no improvement registered

    # Thresholding the image to produce a mask
    _,s_de_hsv_thresholded = cv2.threshold(s_de_hsv,threshold,255,cv2.THRESH_BINARY)

    #Crear un kernel for dilatation
    kernel = np.ones((2,2),np.uint8)
    #Aplicar la operación gradiente de OpenCv con el kernel, calcular erosion y dilatación
    dilatacion = morph.binary_dilation(s_de_hsv_thresholded,kernel)

    #Realizar un llenado de agujeros pequeños, cuyos parámetros son la imagen que deseo rellenar
    #y el área mínima a partir de la cual se llenará el hueco
    filled = morph.remove_small_holes(dilatacion,area_threshold=100) #area_threshold es el area en pixeles de los huecos a rellenar

    #Aplicar la función de skimage de eliminación de objetos pequeños, seleccionando un área mínima
    #con min_size y la conexión con los vecinos del píxel
    objetos_pequeños_eliminados = morph.remove_small_objects(filled,min_size = 100, connectivity = 1)

    # mask_1 = objetos_pequeños_eliminados.copy()

    segmented_lp = segment_image(img_wp, objetos_pequeños_eliminados)

    return segmented_lp




# Preprocessing for license plates that are not yellow anymore but white
def preprocessing_2_segmentation(img, threshold=160): 
    #Apply White-Patch algorithm to image
    img_wp = white_patch_correction(img)

    # Convert image to HSV and extract channel S
    z_in_XYZ = convert_image(img_wp, 'XYZ')[:,:,2]

    # Apply Gamma transformation
    # img_after_gamma = gamma_correction(s_de_hsv, a=1.0, gamma=1.5) #Disabled as no improvement registered

    # Thresholding the image to produce a mask
    _,s_de_hsv_thresholded = cv2.threshold(z_in_XYZ,threshold,255,cv2.THRESH_BINARY)

    #Crear un kernel for dilatation
    kernel = np.ones((2,2),np.uint8)
    #Aplicar la operación gradiente de OpenCv con el kernel, calcular erosion y dilatación
    dilatacion = morph.binary_dilation(s_de_hsv_thresholded,kernel)

    #Realizar un llenado de agujeros pequeños, cuyos parámetros son la imagen que deseo rellenar
    #y el área mínima a partir de la cual se llenará el hueco
    filled = morph.remove_small_holes(dilatacion,area_threshold=100) #area_threshold es el area en pixeles de los huecos a rellenar

    #Aplicar la función de skimage de eliminación de objetos pequeños, seleccionando un área mínima
    #con min_size y la conexión con los vecinos del píxel
    objetos_pequeños_eliminados = morph.remove_small_objects(filled,min_size = 100, connectivity = 1)

    # mask_1 = objetos_pequeños_eliminados.copy()

    segmented_lp = segment_image(img_wp, objetos_pequeños_eliminados)

    return segmented_lp

# Color and contrast correction for yellow license plates (to be used after preprocessing segmentation)
def preprocessing_1_color_correction(img, gamma_2=0.8):

    # Convert the image to YIQ color space and extract the Y channel
    y_of_yiq = convert_image(img, 'YIQ')[:,:,0]

    # Apply histogram extension to create a stronger contrast 
    hist_stretched = histogram_expansion(y_of_yiq)

    # Apply gamma correction on the image to make bright areas brighter 
    hist_stretched_gamma = gamma_correction(hist_stretched, a=1.0, gamma=gamma_2)

    return hist_stretched_gamma



# Color and contrast correction for white license plates (to be used after preprocessing segmentation)
def preprocessing_2_color_correction(img, gamma_2=5.0):

    # Convert the image to a grayscale image
    r_of_rgb = convert_image(img, 'GRAY')

    # Apply histogram extension to create a stronger contrast 
    # hist_stretched = histogram_expansion(r_of_rgb) # Didn´t show a big benefit

    # Apply gamma correction on the image to make bright areas brighter 
    hist_stretched_gamma = gamma_correction(r_of_rgb, a=1.0, gamma=gamma_2)

    return hist_stretched_gamma