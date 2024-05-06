import matplotlib.pyplot as plt
import numpy as np
import PIL as image_lib
import tensorflow as tflow
import keras
import cv2
from keras.layers import Dense, Activation, Dropout, Reshape, Permute
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
# from keras.utils.vis_utils import plot_model

from sklearn.metrics import confusion_matrix
import pathlib
import os
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


# Import dataset 
data_set = "datasetTest"
# directory = tflow.keras.utils.get_file('dataset', origin=data_set, untar=True)
directory = data_set
data_dir = pathlib.Path(directory)

def main():
    # create the training subset
    img_height,img_width=180,180
    batch_size=32
    train_ds = tflow.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        label_mode='categorical',
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # create the validation subset
    validation_ds = tflow.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        label_mode='categorical',
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # import the pre-trained learning model
    resnet_model = Sequential()# create an empty sequential model
    pretrained_model= tflow.keras.applications.ResNet50(
                include_top=False, # allow us to use the pre-trained model for transfer learning. It allow adding input and output layers custom to a problem.
                input_shape=(180,180,3), # the size of the input image
                pooling='avg', # pooling type to use
                classes=3, # number of classes in the dataset
                weights='imagenet' #model is initialized with pre-trained weights
                )

    for each_layer in pretrained_model.layers: # freezes all the layers in the pre-trained model to prevent retraining them.
        each_layer.trainable=False             # This is common practice in transfer learning when we want to use the pre-trained model as a feature extractor.
    resnet_model.add(pretrained_model)         # add the pre-trained model to the empty sequential model 
    # add a fully connected layer to the model
    resnet_model.add(Flatten())
    resnet_model.add(Dense(512, activation='relu'))
    resnet_model.add(Dense(3, activation='softmax')) # use the softmax activation function to output the probability of each class.

    # Train and the Model
    resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = resnet_model.fit(train_ds, validation_data=validation_ds, epochs=10)

    # Evaluate the Model and Visualize the Results
    plt.figure(figsize=(8,8))
    epochs_range=range(10)
    plt.plot(epochs_range, history.history['accuracy'], label='Training Accuracy')
    plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
    plt.axis(ymin=0.4, ymax=1)
    plt.title('Training and Validation Accuracy')
    plt.grid()
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'])
    plt.show()

    # Confusion Matrix Calculation
    y_true = []
    y_pred = []
    class_names = ["LTE","NR","NR_LTE"]
    for images, labels in validation_ds:
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        predictions = resnet_model.predict(images)
        y_pred.extend(np.argmax(predictions, axis=1))

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

    # predict a class for the image
    sample_image = cv2.imread(str('test.png'))
    sample_image_resized = cv2.resize(sample_image, (180,180))
    sample_image = np.expand_dims(sample_image_resized, axis=0)
    # predict a class for the image
    image_pred = resnet_model.predict(sample_image) # array of 3 numbers since the output layer uses softmax classifier

    # convert into human readable output label
    image_output_class = class_names[np.argmax(image_pred)]
    print("The predicted class is: ", image_output_class)

    # Plot the model architecture and save it as an image
    plot_model(resnet_model, to_file='resnet50_model.png', show_shapes=True)

if __name__ == '__main__':
    main()
