import os 
from tensorflow.keras.layers import Dense , Conv2D, Flatten, Dropout, AveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam , SGD 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle 
from tensorflow.keras.callbacks import EarlyStopping 
train_path = '/Users/omkarnaik/Covid19-Detection/Covid19-dataset/train'
test_path = '/Users/omkarnaik/Covid19-Detection/Covid19-dataset/test'
classes = os.listdir(test_path)
def read_process_images(train_path,test_path):
    '''Using ImageDataGenerator from tensorflow to extract the data from the images normalizing it and performing
    data augmentation.'''
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                    rotation_range = 10,
                                    width_shift_range = 0.1,
                                    height_shift_range = 0.1,
                                    shear_range = 0.1,
                                    zoom_range = 0.1,
                                    horizontal_flip = True)
    train_gen = train_datagen.flow_from_directory(directory = train_path,
                                                batch_size = 16,
                                                class_mode = 'categorical',
                                                target_size = (128,128))
    
    test_datagen = ImageDataGenerator(rescale = 1./255,
                                    rotation_range =10,
                                    width_shift_range = 0.1,
                                    height_shift_range = 0.1,
                                    shear_range = 0.1,
                                    zoom_range = 0.1,
                                    horizontal_flip = 'True',
                                    fill_mode = 'nearest')
    test_gen = test_datagen.flow_from_directory(directory = test_path,
                                                batch_size = 16,
                                                class_mode = 'categorical',
                                                target_size = (128,128))
    return train_gen,test_gen
if __name__ == '__main__' :
    
    train_gen ,test_gen = read_process_images(train_path,test_path)
    ## model creation 
    model = Sequential()
    
    # layer 1 - Convolutional Layer accompanied by a Average Pooling layer
    model.add(Conv2D(32,(3,3),activation = 'relu', input_shape = (128,128,3),kernel_initializer = 'he_uniform',padding = 'same'))
    model.add(AveragePooling2D((2,2)))
    
    # layer 2 - Convolutional Layer accompanied by a Average Pooling layer 
    model.add(Conv2D(64,(3,3), activation = 'relu',kernel_initializer = 'he_uniform',padding = 'same'))
    model.add(AveragePooling2D((2,2)))
    
    # layer 3 - a layer to flatten the 2D data to single dimension
    model.add(Flatten())
    
    # layer 4 - Dense layer followed by a dropout layer
    model.add(Dense(64,activation = 'relu',kernel_initializer = 'he_uniform', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.25))
    
    # layer 5 - Dense layer which is responsible for the classifying the class of the input
    model.add(Dense(3,activation = 'softmax'))
    
    params = {'learning_rate': 0.00075, 'epsilon' : 1e-07}
    
    optimizer =  Adam(**params)
    
    model.compile(optimizer = optimizer, 
                loss = 'categorical_crossentropy',metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss',patience=10, mode = 'min')
    model.fit(train_gen,epochs = 20, batch_size = 16,validation_data = test_gen,callbacks = early_stopping)
    
    pickle.dump(model,open('model.pkl','wb'))
    