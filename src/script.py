import datumaro as dm
import pickle
import json
import random
import cv2
import numpy as np
import concurrent.futures
import sys
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gc
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

#####################################################################################################################################
# Prep
#####################################################################################################################################

resolutions = [64, 128, 256, 384, 512, 1024]
with open('./datasets/categories.json', 'r') as file:
    categories_dict = json.load(file)
with open('./datasets/categories_inverse.json', 'r') as file:
    categories_dict_inverse = json.load(file)

#####################################################################################################################################
# Create Datasets
#####################################################################################################################################

def process_images(category, image_paths, image_dimensions):
    processed_images = []
    processed_images_grayscale = []
    for image_path in image_paths:
        original_image = cv2.imread(image_path)

        image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image, dsize=image_dimensions, interpolation=cv2.INTER_CUBIC)

        grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        resized_grayscale = cv2.resize(grayscale_image, dsize=image_dimensions, interpolation=cv2.INTER_CUBIC)
        resized_grayscale = np.expand_dims(resized_grayscale, axis=-1)
        
        processed_images.append((resized, category, categories_dict_inverse[str(category)]))
        processed_images_grayscale.append((resized_grayscale, category, categories_dict_inverse[str(category)]))
    return processed_images, processed_images_grayscale

def create_datasets():
    dataset = dm.Dataset.import_from('../../fastener_dataset/annotations/instances_default.json', format='coco')
    dataset_list = list(dataset)

    data = {}
    for i in range(0, 16970):
        item = dataset_list[i]
        item_annotation = item.annotations[0]
        
        new_path = item.media.path.replace(':', '_')
        
        item_category = str(item_annotation.as_dict()['attributes']['category'])
        if(data.get(item_category) == None):
            data[item_category] = []
        data[item_category].append(new_path)

    for resolution in resolutions:
        print(f'Processing {resolution}x{resolution}')
        image_dimensions = (resolution, resolution)

        # Parallel:
        dataset = []
        dataset_no_other = []
        grayscale_dataset = []

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for category in data.keys():
                random.shuffle(data[category])
                futures.append(executor.submit(process_images, category, data[category], image_dimensions))
            for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
                dataset.extend(future.result()[0])
                grayscale_dataset.extend(future.result()[1])

        print('\t\tRemoving Misc.')
        for member in dataset:
            if str(member[2]) != '0':
                dataset_no_other.append(member)
        print(f'\t\t{len(dataset) > len(dataset_no_other)}')
        random.shuffle(dataset)
        random.shuffle(dataset_no_other)
        random.shuffle(grayscale_dataset)

        with open(f'./datasets/full/{str(resolution)}/dataset.pkl', 'wb') as file:
            print('\tSaving dataset...')
            pickle.dump(dataset, file)
            print('\tDone!')
        with open(f'./datasets/full/{str(resolution)}/dataset_NO_OTHER.pkl', 'wb') as file:
            print('\tSaving dataset (NO MISC)...')
            pickle.dump(dataset_no_other, file)
            print('\tDone!')
        with open(f'./datasets/full/{str(resolution)}/dataset_GRAYSCALE.pkl', 'wb') as file:
            print('\tSaving dataset (GRAYSCALE)...')
            pickle.dump(grayscale_dataset, file)
            print('\tDone!')


#####################################################################################################################################
# Train Models
#####################################################################################################################################

def simple_model(image_dimensions, num_classes):
    inputs = keras.Input(shape=(image_dimensions[0], image_dimensions[1], 3))
    x = layers.Rescaling(1./255)(inputs)

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomFlip("vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
        ]
    )

    x = data_augmentation(x)

    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)  
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=512, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=1024, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)  

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
    # model.summary()

def data_generator(X, y, batch_size):
    num_samples = X.shape[0]
    while True:
        for i in range(0, num_samples, batch_size):
            yield X[i:i+batch_size], y[i:i+batch_size]

def split_dataset(dataset, resolution):
    # Encode
    all_labels = [label for _, label, _ in dataset]
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    dataset_encoded = [(image, label_encoder.transform([label])[0], cat) for image, label, cat in dataset]
    with open(f'./datasets/full/{str(resolution)}/encoder.pkl', "wb") as file:
        print('Saving encoder...')
        pickle.dump(label_encoder, file)
        print('Done!')

    # Create sets
    with open('./datasets/category_count.json', 'rb') as file:
        category_count = json.load(file)
    train_set_encoded = []
    validation_set_encoded = []
    test_set_encoded = []
    category_count_new = {}
    for image, l, cat in dataset_encoded:
        label = label_encoder.inverse_transform([l])[0]
        if category_count_new.get(label) is None:
            category_count_new[label] = 0
        category_count_new[label] += 1

        if category_count_new[label] <= 0.7 * category_count[label]:
            train_set_encoded.append((image, l, cat))
        elif category_count_new[label] > 0.7 * category_count[label] and category_count_new[label] <= 0.85 * category_count[label]:
            validation_set_encoded.append((image, l, cat))
        else:
            test_set_encoded.append((image, l, cat))
    del dataset, dataset_encoded, category_count, category_count_new

    (train_images, train_labels, train_cats) = zip(*train_set_encoded)
    (validation_images, validation_labels, validation_cats) = zip(*validation_set_encoded)
    (test_images, test_labels, test_cats) = zip(*test_set_encoded)

    train_set_len = len(train_labels)
    validation_set_len = len(validation_labels)
    test_set_len = len(test_labels)
    del train_set_encoded, validation_set_encoded, test_set_encoded
    train_images = np.array(train_images)
    train_labels = np.array([int(label) for label in train_labels])
    train_cats = np.array([int(cat) for cat in train_cats])
    validation_images = np.array(validation_images)
    validation_labels = np.array([int(label) for label in validation_labels])
    validation_cats = np.array([int(cat) for cat in validation_cats])
    # test_images = np.array(test_images)
    # test_labels = np.array([int(label) for label in test_labels])
    # test_cats = np.array([int(cat) for cat in test_cats])

    # del test_images, test_labels, test_cats
    train_generator_142 = data_generator(train_images, train_labels, batch_size)
    train_generator_6 = data_generator(train_images, train_cats, batch_size)
    del train_images, train_labels, train_cats
    validation_generator_142 = data_generator(validation_images, validation_labels, batch_size)
    validation_generator_6 = data_generator(validation_images, validation_cats, batch_size)
    del validation_images, validation_labels, validation_cats

    return train_generator_142, train_generator_6, validation_generator_142, validation_generator_6, train_set_len, validation_set_len, test_set_len

def save_plot(history):
    # Extracting the history values
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(accuracy) + 1)

    # Setting a style for the plots
    # plt.style.use('seaborn-darkgrid')

    # Plotting Training and Validation Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracy, "o-", label="Training Accuracy", color='blue')
    plt.plot(epochs, val_accuracy, "o-", label="Validation Accuracy", color='orange')
    plt.title("Training and Validation Accuracy", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True)
    plt.savefig(f'./model_results/full/{str(resolution)}/accuracy_history.png', bbox_inches='tight')

    # Plotting Training and Validation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, "o-", label="Training Loss", color='blue')
    plt.plot(epochs, val_loss, "o-", label="Validation Loss", color='orange')
    plt.title("Training and Validation Loss", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(loc="upper right", fontsize=12)
    plt.grid(True)
    plt.savefig(f'./model_results/full/{str(resolution)}/loss_history.png', bbox_inches='tight')


def train_models():
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(gpu_devices) > 0:
        print('GPU active')
    else:
        print('GPU not found')
        exit(1)
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    for resolution in resolutions:
        gc.collect() 
        keras.backend.clear_session()
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(gpu_devices) > 0:
            print('GPU active')
        else:
            print('GPU not found')
            exit(1)

        image_dimensions = (resolution, resolution)
        if resolution <= 64:
            batch_size = 32
        elif resolution <= 512:
            batch_size = 16
        else:
            batch_size = 8

        dataset_filepath = f"./datasets/full/{str(resolution)}/dataset.pkl"
        with open(dataset_filepath, "rb") as file:
            print('Loading dataset...')
            dataset = pickle.load(file)
            print('Done!')

        train_generator_142, train_generator_6, validation_generator_142, validation_generator_6, train_set_len, validation_set_len, test_set_len = split_dataset(dataset, resolution)

        for num_classes in [142]:
            model = simple_model(image_dimensions, num_classes)
            model.compile(loss="sparse_categorical_crossentropy",
                optimizer=keras.optimizers.Adam(),
                metrics=["accuracy"])

            callbacks = [
                keras.callbacks.ModelCheckpoint(
                    filepath=f"./datasets/full/fastener_net_{str(num_classes)}_{str(resolution)}.keras",
                    save_best_only=True,
                    monitor='val_loss')
            ]

            history = model.fit(train_generator,
                                steps_per_epoch=math.ceil(train_set_len / batch_size),
                                epochs=int(epochs),
                                validation_data=validation_generator,
                                validation_steps=math.ceil(validation_set_len / batch_size),
                                callbacks=callbacks)
            save_plot(history)


        
    







#####################################################################################################################################
# Analyze Results
#####################################################################################################################################







#####################################################################################################################################
# Main
#####################################################################################################################################

if len(sys.argv) < 2:
    print("Usage: script.py <arg1> <arg2> ...")
    exit(1)

args = sys.argv[1:] 

if 'c' in args:
    create_datasets()
if 't' in args:
    train_models()
if 'a' in args:
    analyze_results()