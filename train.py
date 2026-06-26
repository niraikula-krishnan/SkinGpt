import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input


def find_dataset_root(path):
    for root, dirs, files in os.walk(path):
        if 'train' in dirs and 'test' in dirs:
            return root
    return None


def extract_dataset(dataset_path):
    if dataset_path.lower().endswith('.zip'):
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(dataset_path, 'r') as z:
            z.extractall(temp_dir)

        dataset_root = find_dataset_root(temp_dir)
        if dataset_root:
            return dataset_root, temp_dir
        shutil.rmtree(temp_dir)
        raise FileNotFoundError('Zip archive does not contain train/ and test/ folders.')

    if os.path.isdir(dataset_path):
        dataset_root = find_dataset_root(dataset_path)
        if dataset_root:
            return dataset_root, None
        raise FileNotFoundError('Dataset folder must contain train/ and test/ subfolders.')

    raise FileNotFoundError(f'Dataset path not found: {dataset_path}')


def build_model(num_classes, img_size=224):
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = layers.RandomFlip('horizontal')(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    x = preprocess_input(x)

    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=x, pooling='avg')
    base_model.trainable = False

    x = layers.Dropout(0.3)(base_model.output)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs)


def train_model(dataset_root, img_size=224, batch_size=32, epochs=10, model_path='skin_model.h5'):
    train_dir = os.path.join(dataset_root, 'train')
    test_dir = os.path.join(dataset_root, 'test')

    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        raise FileNotFoundError('Dataset must contain train/ and test/ folders.')

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode='int',
        validation_split=0.2,
        subset='training',
        seed=123
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode='int',
        validation_split=0.2,
        subset='validation',
        seed=123
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode='int'
    )

    class_names = train_ds.class_names
    print('Found classes:', class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    model = build_model(len(class_names), img_size=img_size)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    checkpoint = callbacks.ModelCheckpoint(
        model_path,
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        verbose=1
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint, early_stop, reduce_lr]
        )

    with open('class_names.json', 'w', encoding='utf-8') as f:
        json.dump(class_names, f)

    print('Model saved to', model_path)
    print('Class names saved to class_names.json')


def main():
    parser = argparse.ArgumentParser(description='Train the skin disease classifier')
    parser.add_argument(
        '--dataset',
        nargs='?',
        const='dataset',
        default='dataset',
        help='Path to disease.zip or folder containing train/ and test. Defaults to ./dataset if not provided.'
    )
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--model-out', default='skin_model.h5')
    args = parser.parse_args()

    try:
        dataset_root, temp_dir = extract_dataset(args.dataset)
    except FileNotFoundError as e:
        print(f'Error: {e}')
        print('Please provide a valid dataset path using --dataset or create a ./dataset folder with train/ and test/ subfolders.')
        sys.exit(1)

    try:
        train_model(
            dataset_root,
            img_size=args.img_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            model_path=args.model_out
        )
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    main()
