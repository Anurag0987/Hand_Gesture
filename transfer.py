from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.applications.vgg16 import preprocess_input
# from tensorflow.keras.applications.vgg19 import VGG19
# from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications.vgg19 import preprocess_input
# from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sn
# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'data3/train'
valid_path = 'data3/test'

# add preprocessing layer to the front of VGG
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in vgg.layers:
  layer.trainable = False
  

  
  # useful for getting number of classes
folders = glob('data3/train/*')
  

# our layers - you can add more if you want
x = Flatten()(vgg.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)



train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('data3/train',
                                                 target_size = (224, 224),
                                                 batch_size = 16,
                                                 class_mode = 'categorical',shuffle=False)

test_set = test_datagen.flow_from_directory('data3/test',
                                            target_size = (224, 224),
                                            batch_size = 16,
                                            class_mode = 'categorical',shuffle=False)

'''r=model.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 5,
                         validation_data = test_set,
                         nb_val_samples = 2000)'''

# fit the model

from tensorflow.keras.callbacks import ModelCheckpoint

filepath = "checkpoint_vgg16_8.4/weights-improvement-{epoch:02d}.h5"
checkpoint  = ModelCheckpoint(filepath,monitor = 'val_loss', verbose=1, save_best_only=True, mode="min")

# early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

callback_list = [checkpoint]


r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=6,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set),callbacks = callback_list)
# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss_2')

# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc_2')

num_of_test_samples = 5695
batch_size = 16
Y_pred = model.predict_generator(test_set, test_set.samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)

y_pred.shape

print('Confusion Matrix')
print(confusion_matrix(test_set.classes, y_pred))
cm = confusion_matrix(test_set.classes, y_pred)
print('Classification Report')
target_names = ['blank','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
print(classification_report(test_set.classes, y_pred, target_names=target_names))



 
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
  
Y_pred = model.predict_generator(test_set, num_of_test_samples // batch_size+1)

# =============================================================================
# test_data_array = np.asarray(test_set)
# test_data_array
# # 
# data_count, batch_count, w, h, c = test_data_array.shape
# 
# test_data_array=np.reshape(test_data_array, (data_count*batch_count, w, h, c))
# test_labels_array = np.reshape(test_labels_array , (data_count*batch_count, -1)) 
# cnf_matrix = confusion_matrix(test_set , Y_pred,labels=['blank','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])
#  
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=['blank','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'],
#                       title='Confusion matrix')
# =============================================================================
# 
# =============================================================================
# cm = confusion_matrix(training_set, test_set.argmax(axis=1))
# print(classification_report(training_set,Y_pred))
# =============================================================================

df_cm = pd.DataFrame(cm, index = [i for i in "0ABCDEFGHIJKLMNOPQRSTUVWXYZ"],
                  columns = [i for i in "0ABCDEFGHIJKLMNOPQRSTUVWXYZ"])
plt.figure(figsize = (25,20))
sn.heatmap(df_cm, annot=True)
model.save('handgesture_vgg16_8.4.h5')