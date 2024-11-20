from PyQt5 import QtCore, QtGui, QtWidgets
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 800)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Main Layout
        self.layout = QtWidgets.QVBoxLayout(self.centralwidget)

        # Title
        self.label_title = QtWidgets.QLabel("TESO Tomato Leaf Disease Prediction App")
        font = QtGui.QFont("Times New Roman", 20, QtGui.QFont.Bold)
        self.label_title.setFont(font)
        self.label_title.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.label_title)

        # Image Preview
        self.imageLbl = QtWidgets.QLabel("Image Preview")
        self.imageLbl.setFrameShape(QtWidgets.QFrame.Box)
        self.imageLbl.setAlignment(QtCore.Qt.AlignCenter)
        self.imageLbl.setMinimumSize(400, 300)
        self.layout.addWidget(self.imageLbl)

        # Buttons Layout
        self.buttonLayout = QtWidgets.QHBoxLayout()

        # Browse Button
        self.BrowseImage = QtWidgets.QPushButton("Open Image")
        self.BrowseImage.setStyleSheet("background-color: #4CAF50; color: white; font-size: 16px; font-weight: bold;")
        self.BrowseImage.clicked.connect(self.loadImage)
        self.buttonLayout.addWidget(self.BrowseImage)

        # Classify Button
        self.Classify = QtWidgets.QPushButton("Classify Image")
        self.Classify.setStyleSheet("background-color: #2196F3; color: white; font-size: 16px; font-weight: bold;")
        self.Classify.clicked.connect(self.classifyFunction)
        self.buttonLayout.addWidget(self.Classify)

        # Training Button
        self.Training = QtWidgets.QPushButton("Train Model")
        self.Training.setStyleSheet("background-color: #FF5722; color: white; font-size: 16px; font-weight: bold;")
        self.Training.clicked.connect(self.trainingFunction)
        self.buttonLayout.addWidget(self.Training)

        self.layout.addLayout(self.buttonLayout)

        # Prediction Label
        self.resultLabel = QtWidgets.QLabel("Prediction: N/A")
        font_result = QtGui.QFont("Arial", 14, QtGui.QFont.Bold)
        self.resultLabel.setFont(font_result)
        self.resultLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.resultLabel)

        # Confidence Level
        self.confidenceLabel = QtWidgets.QLabel("Confidence: N/A")
        font_confidence = QtGui.QFont("Arial", 12)
        self.confidenceLabel.setFont(font_confidence)
        self.confidenceLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.confidenceLabel)

        # Status Bar
        self.statusBar = QtWidgets.QLabel("Status: Ready")
        self.statusBar.setAlignment(QtCore.Qt.AlignRight)
        self.layout.addWidget(self.statusBar)

        MainWindow.setCentralWidget(self.centralwidget)

        self.file = None

    def loadImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "",
                                                            "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)")
        if fileName:
            self.file = fileName
            pixmap = QtGui.QPixmap(fileName).scaled(
                self.imageLbl.width(), self.imageLbl.height(), QtCore.Qt.KeepAspectRatio)
            self.imageLbl.setPixmap(pixmap)
            self.statusBar.setText(f"Status: Loaded image {fileName}")
        else:
            self.statusBar.setText("Status: No file selected")

    def classifyFunction(self):
        if not self.file:
            self.statusBar.setText("Status: No image to classify")
            return
        try:
            loaded_model = tf.keras.models.load_model('leaf_model.h5')
            labels = ['Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold',
                      'Septoria Leaf Spot', 'Spider Mites', 'Target Spot',
                      'Yellow Leaf Curl Virus', 'Healthy', 'Mosaic Virus']

            test_image = image.load_img(self.file, target_size=(224, 224))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            result = loaded_model.predict(test_image)
            prediction = labels[np.argmax(result)]
            confidence = np.max(result) * 100

            self.resultLabel.setText(f"Prediction: {prediction}")
            self.confidenceLabel.setText(f"Confidence: {confidence:.2f}%")
            self.statusBar.setText("Status: Classification complete")
        except Exception as e:
            self.statusBar.setText(f"Status: Error in classification - {str(e)}")

    def trainingFunction(self):
        train_dir = "train"
        val_dir = "val"
        test_dir = "val"

        train_data = tf.keras.preprocessing.image_dataset_from_directory(directory=train_dir,
                                                                         image_size=(
                                                                             224, 224),
                                                                         label_mode="categorical",
                                                                         batch_size=32)
        val_data = tf.keras.preprocessing.image_dataset_from_directory(directory=val_dir,
                                                                       image_size=(
                                                                           224, 224),
                                                                       label_mode="categorical",
                                                                       batch_size=32)

        # 1.Create baseline model with tf.keras.applications
        base_model = tf.keras.applications.EfficientNetV2B0(include_top=False)

        # To begin fine-tuning, let's start by setting the last 10 layers of our base_model.trainable = True
        base_model.trainable = False

        # 3.Create inputs to the model
        inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")

        # 5.pass the inputs to the baseline model
        x = base_model(inputs, training=False)
        print(
            f"Shape of after passing inputs to the baseline model is {x.shape}")

        # 6.Average pool the outputs of the base model
        x = tf.keras.layers.GlobalAveragePooling2D(
            name="global_average_pooling_layer")(x)
        print(f"Shape after passing GlobalAveragePooling layer is {x.shape}")

        # 7.Ouput layer
        outputs = tf.keras.layers.Dense(
            10, activation="softmax", name="output_layer")(x)

        # 8.Combine the model
        model_0 = tf.keras.Model(inputs, outputs)

        # 9.Compile the model
        model_0.compile(loss="categorical_crossentropy",
                        optimizer=tf.keras.optimizers.Adam(
                            learning_rate=0.005),
                        metrics=['accuracy'])

        # 11.Fit the model
        history_0 = model_0.fit(train_data,
                                steps_per_epoch=len(train_data),
                                epochs=40,
                                validation_data=val_data,
                                validation_steps=len(val_data)

                                )
        model_0.save('leaf_model.h5')


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
