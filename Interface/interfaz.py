from interfaz_ui import *
from algoritmos import *
from tkinter import filedialog, messagebox
from tkinter import *
from time import time

DIR_RESULTS = '../results'
DIR_UNKNOWNS = '../Unknowns'
DIR_DEFAULT = '../'


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon("./icons/logo.png"))
        self.setFixedWidth(657)
        self.setFixedHeight(261)
        self.carpetaImagen.clicked.connect(self.buscar_archivo)
        self.boton_knn.clicked.connect(self.KNN)
        self.boton_svm.clicked.connect(self.SVM)
        self.boton_nb.clicked.connect(self.NaiveBayes)
        self.boton_gnb.clicked.connect(self.GaussianNaibeBayes)
        self.actionGuardar_Encoddings_2.triggered.connect(self.saveKnownEncodings)
        self.actionKNN.triggered.connect(self.KNN)
        self.actionSVM.triggered.connect(self.SVM)
        self.actionNaive_bayes.triggered.connect(self.NaiveBayes)
        self.actionGaussian_Naive_bayes.triggered.connect(self.GaussianNaibeBayes)

    def buscar_archivo(self):
        root = Tk()
        root.withdraw()
        root.filename = filedialog.askopenfilename(initialdir=DIR_DEFAULT, title="Escoger imagen",
                                                   filetypes=[("Image File", '.jpg')])
        if root.filename != "":
            self.line_editImagen.setText(str(root.filename))

    def KNN(self):
        path = self.line_editImagen.text()
        if path != "":
            embeddings = calcularEmbeddingsCarasImagen(path)
            tiempo_inicial = time()
            resultado = algoritmoKNN(embeddings)
            tiempo_final = time()
            tiempo_ejecucion = tiempo_final - tiempo_inicial
            dibujarBoundingBoxes(path, resultado)
            self.tiempo_knn.setText("Tiempo que tarda en identificar KNN:")
            self.result_tiempo_knn.setText(str(round(tiempo_ejecucion, 4)) + " segundos.")
            self.result_acurracy_knn.setText("Acurracy trainingSet: "+str(KNNAcurracy()))

    def SVM(self):
        path = self.line_editImagen.text()
        if path != "":
            embeddings = calcularEmbeddingsCarasImagen(path)
            tiempo_inicial = time()
            resultado = algoritmoSVM(embeddings)
            tiempo_final = time()
            tiempo_ejecucion = tiempo_final - tiempo_inicial
            dibujarBoundingBoxes(path, resultado)
            self.tiempo_svm.setText("Tiempo que tarda en identificar SVM:")
            self.result_tiempo_svm.setText(str(round(tiempo_ejecucion, 4)) + " segundos.")
            self.result_acurracy_svm.setText("Acurracy trainingSet: " + str(SVMAcurracy()))

    def NaiveBayes(self):
        path = self.line_editImagen.text()
        if path != "":
            embeddings = calcularEmbeddingsCarasImagen(path)
            tiempo_inicial = time()
            resultado = bernoulliNB(embeddings)
            tiempo_final = time()
            tiempo_ejecucion = tiempo_final - tiempo_inicial
            dibujarBoundingBoxes(path, resultado)
            self.tiempo_nb.setText("Tiempo que tarda en identificar NB:")
            self.result_tiempo_nb.setText(str(round(tiempo_ejecucion, 4)) + " segundos.")
            self.result_acurracy_nb.setText("Acurracy trainingSet: " + str(bernoulliNBAcurracy()))

    def GaussianNaibeBayes(self):
        path = self.line_editImagen.text()
        if path != "":
            embeddings = calcularEmbeddingsCarasImagen(path)
            tiempo_inicial = time()
            resultado = gaussianNB(embeddings)
            tiempo_final = time()
            tiempo_ejecucion = tiempo_final - tiempo_inicial
            dibujarBoundingBoxes(path, resultado)
            self.tiempo_gnb.setText("Tiempo que tarda en identificar GNB:")
            self.result_tiempo_gnb.setText(str(round(tiempo_ejecucion, 4)) + " segundos.")
            self.result_acurracy_gnb.setText("Acurracy trainingSet: " + str(gaussianNBAcurracy()))

    def saveKnownEncodings(self):
        #Guardar los encodings en el archivo csv
        numImg = guardar_embeddings_csv()
        suma = 0
        #Contar las lineas que se han escrito
        with open('log.csv', "r") as file:
            suma += sum(1 for line in file)
        numImgReconocidas = (suma / 2)
        root = Tk()
        root.withdraw()
        #Mostrar en una ventana el número de imágenes reconocidas
        messagebox.showinfo(message="Numero total de imágenes en la base de datos: " + str(
            numImg) + "\nImágenes reconocidas por el sistema: " + str(int(numImgReconocidas)), title="Información")
        root.lift()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
