import tensorflow as tf
from keras.models import load_model
import cv2
import os
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np

DIR_KNOWNS = '../Knowns'
DIR_UNKNOWNS = '../Unknowns'

# Leer mobilenet_graph.pb
with tf.io.gfile.GFile('../Facenet/frozen_inference_graph_face.pb', 'rb') as f:  # Leer modelo con tensorflow
    graph_def = tf.compat.v1.GraphDef()  # Información en bruto
    graph_def.ParseFromString(f.read())  # Leer linea por linea la información

with tf.Graph().as_default() as mobilenet:  # Creación de grafo con tensorflow con el nombre mobilenet
    tf.import_graph_def(graph_def, name='')  # Almacena los datos de la definicion del grafo en mobilenet

# FaceNet
modelo = load_model('../Facenet/model/facenet_keras.h5')
modelo.load_weights('../Facenet/weights/facenet_keras_weights.h5')

# Entrenamiento
# Crear dataset
dataset = np.loadtxt("log.csv", delimiter=",")
X = dataset[:, 0:128]
Y = dataset[:, 128]
# Modelos de clasificación
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='brute', leaf_size=30, p=2, metric='minkowski',
                           metric_params=None, n_jobs=None)
bNB = BernoulliNB(alpha=25)
gNB = GaussianNB(var_smoothing=25)
svm = SVC(C=1e3, gamma=0.0001, kernel='linear', class_weight='balanced', probability=True)
# Separar entrenamiento del testeo
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=4)


# Cargar imagen
def load_image(dir):
    return cv2.cvtColor(cv2.imread(f'{dir}'), cv2.COLOR_BGR2RGB)  # Leer la imagen y cambiar el color de BGR a RGB


# Cargar imagen
def load_image_name(dir, name):
    # Leer la imagen y cambiar el color de BGR a RGB
    return cv2.cvtColor(cv2.imread(f'{dir}\{name}'), cv2.COLOR_BGR2RGB)


# Detección de imagenes
def detect_faces(image, score_threshold=0.7):
    global boxes, scores
    (imh, imw) = image.shape[:-1]
    img = np.expand_dims(image, axis=0)

    # Inicializar mobilenet
    sess = tf.compat.v1.Session(graph=mobilenet)
    image_tensor = mobilenet.get_tensor_by_name('image_tensor:0')
    boxes = mobilenet.get_tensor_by_name('detection_boxes:0')
    scores = mobilenet.get_tensor_by_name('detection_scores:0')

    # Predicción (detección)
    (boxes, scores) = sess.run([boxes, scores], feed_dict={image_tensor: img})

    # Reajustar tamaños boxes, scores
    boxes = np.squeeze(boxes, axis=0)
    scores = np.squeeze(scores, axis=0)

    # Depurar bounding boxes
    idx = np.where(scores >= score_threshold)[0]

    # Crear bounding boxes
    bboxes = []
    for index in idx:
        ymin, xmin, ymax, xmax = boxes[index, :]
        (left, right, top, bottom) = (xmin * imw, xmax * imw, ymin * imh, ymax * imh)
        left, right, top, bottom = int(left), int(right), int(top), int(bottom)
        bboxes.append([left, right, top, bottom])

    return bboxes


# Dibujar bounding boxes
def draw_box(image, box, color, line_width=6):
    if not box:
        return image
    else:
        cv2.rectangle(image, (box[0], box[2]), (box[1], box[3]), color, line_width)
    return image


# Extraer rostros
def extract_faces(image, bboxes, new_size=(160, 160)):
    cropped_faces = []
    for box in bboxes:
        left, right, top, bottom = box
        face = image[top:bottom, left:right]
        cropped_faces.append(cv2.resize(face, dsize=new_size))
    return cropped_faces


def compute_embedding(model, face):
    face = face.astype('float32')

    mean, std = face.mean(), face.std()
    face = (face - mean) / std

    face = np.expand_dims(face, axis=0)

    embedding = model.predict(face)
    return embedding


def compare_faces(embs_ref, emb_desc, umbral=10):
    distancias = []
    for emb_ref in embs_ref:
        distancias.append(np.linalg.norm(emb_ref - emb_desc))
    distancias = np.array(distancias)
    return distancias, list(distancias <= umbral)


def guardar_embeddings_csv():
    with open('log.csv', "w") as file:
        contador = 0
        numImagenes = 0
        for dir in os.listdir(DIR_KNOWNS):
            contador += 1
            dir_path = './' + DIR_KNOWNS + '/' + dir
            for imgName in os.listdir(dir_path):
                if imgName.endswith(('.jpg', '.png')):
                    image = load_image_name(dir_path, imgName)
                    bboxes = detect_faces(image)
                    faces = extract_faces(image, bboxes)
                    if len(faces) is not 0:
                        f_list = compute_embedding(modelo, faces[0]).flatten()
                        f_list = np.append(f_list, contador)
                        writer = csv.writer(file, delimiter=',')
                        writer.writerow(f_list)
                numImagenes += 1
    return numImagenes


def KNNAcurracy():
    knn.fit(X_train, y_train)
    return knn.score(X_test, y_test)


def SVMAcurracy():
    svm.fit(X_train, y_train)
    return svm.score(X_test, y_test)


def bernoulliNBAcurracy():
    bNB.fit(X_train, y_train)
    return bNB.score(X_test, y_test)


def gaussianNBAcurracy():
    gNB.fit(X_train, y_train)
    return gNB.score(X_test, y_test)


def algoritmoKNN(embeddings):
    knn.fit(X, Y)
    resultado = [-1] * len(embeddings)
    i = 0
    for emb in embeddings:
        array, indices = knn.kneighbors(emb, return_distance=True)
        print(array[0][0])
        if array[0][0] <= 10:  # Cojo la menor de las distancias a los 5 vecinos proximos
            resultado[i] = knn.predict(emb)
        i += 1
    return resultado


def bernoulliNB(embeddings):
    bNB.fit(X, Y)
    resultado = [-1] * len(embeddings)
    i = 0
    for emb in embeddings:
        x = bNB.predict_proba(emb)
        if np.amax(x) >= 0.9439747219702983:
            resultado[i] = bNB.predict(emb)
        i += 1
    return resultado


def gaussianNB(embeddings):
    gNB.fit(X, Y)
    resultado = [-1] * len(embeddings)
    i = 0
    for emb in embeddings:
        x = gNB.predict_proba(emb)
        if np.amax(x) >= 0.24:
            resultado[i] = gNB.predict(emb)
        i += 1
    return resultado


def algoritmoSVM(embeddings):
    svm.fit(X, Y)
    i = 0
    umbral_decision = 0.45
    resultado = [-1] * len(embeddings)
    for emb in embeddings:
        prob = svm.predict_proba(emb)
        if np.amax(prob) >= umbral_decision:
            resultado[i] = svm.predict(emb)
        i += 1
    return resultado


def calcularEmbeddingsCarasImagen(path):
    embeddings = []
    image_input = load_image(path)
    bboxes = detect_faces(image_input)
    faces = extract_faces(image_input, bboxes)
    for face in faces:
        embeddings.append(compute_embedding(modelo, face))
    return embeddings


def dibujarBoundingBoxes(path, arrayResultado):
    nombres_conocidos = []
    for directorio in os.listdir(DIR_KNOWNS):
        nombres_conocidos.append(directorio)

    image_input = load_image(path)
    result = []
    # RELLENAR ARRAY DE RESULTADO CON EL NÚMERO DE CARAS QUE HALLA EN LA IMAGEN
    bboxes = detect_faces(image_input)
    for b in bboxes:
        result.append('???')
    x = 0
    for i in arrayResultado:
        if i != -1:
            result[x] = nombres_conocidos[int(i - 1)]
        x += 1

    # Cargamos una fuente de texto:
    font = cv2.FONT_HERSHEY_COMPLEX
    # Dibujamos un recuadro rojo alrededor de los rostros desconocidos, y uno verde alrededor de los conocidos:
    for (left, right, top, bottom), nombre in zip(bboxes, result):

        # Cambiar el color segun el nombre:
        if nombre != "???":
            color = (0, 255, 0)  # Verde
        else:
            color = (255, 0, 0)  # Rojo

        # Dibujar los recuadros alrededor del rostro:
        cv2.rectangle(image_input, (left, top), (right, bottom), color, 2)
        cv2.rectangle(image_input, (left, bottom - 20), (right, bottom), color, -1)

        # Escribir el nombre de la persona:
        cv2.putText(image_input, nombre, (left, bottom - 6), font, 0.6, (0, 0, 0), 1)

    cv2.imshow('Imagen identificada.', cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR))
