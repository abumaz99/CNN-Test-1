from modules import *

np.random.seed(2)

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

x_train = train.drop(['label'], axis=1).values.astype('float32') 
y_train = train['label'].values.astype('int32') 
x_test = test.values.astype('float32')
x_train = x_train.reshape(x_train.shape[0], 28, 28) / 255.0
x_test = x_test.reshape(x_test.shape[0], 28, 28) / 255.0

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.10, random_state=2)

x_train = x_train.reshape(x_train.shape[0], 28, 28,1)  
x_val = x_val.reshape(x_val.shape[0], 28, 28,1)  
x_test = x_test.reshape(x_test.shape[0], 28, 28,1) 

def greatModel(input_shape):

    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(32, (5, 5), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((2, 2), name='max_pool0')(X)

    X = Conv2D(64, (3, 3), strides = (1, 1), name = 'conv1')(X)
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((2, 2), name='max_pool1')(X)

    X = Conv2D(64, (3, 3), strides = (1, 1), name = 'conv2')(X)
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((2, 2), name='max_pool2')(X)

    X = Flatten()(X)
    X = Dense(10, activation='softmax', name='fc')(X)

    model = Model(inputs=X_input, outputs=X, name='greatModel')

    return model
    

model_run = greatModel(x_train.shape[1:])

model_run.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_run.fit(x_train, y_train, epochs=3, batch_size=64)

preds = model_run.predict(x_test, batch_size=64)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


