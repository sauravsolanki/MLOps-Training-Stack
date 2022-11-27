from typing import Dict

import mlflow
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pickle

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, \
    confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt

from dataclasses import dataclass


@dataclass
class Config:
    DATASET_FILE_PATH = "mnist_test.csv"
    DATASET_ROOT_PATH = "./dataset"
    IMAGE_WIDTH = 28
    IMAGE_HEIGHT = 28
    batch_size = 2
    DATASET_SPLIT = "0.7:.15:0.15"
    epochs = 10
    num_classes: int = 0
    class_names = []

    model_path: str = "saved-model/best-model"
    dict_path: str = "saved-model/training_history.pickle"


@task
def get_dataset(config: Config):
    # tensorflow Image Dataset
    params = {
        "directory": config.DATASET_ROOT_PATH,
        "validation_split": 0.3,
        "subset": "training",
        "seed": 123,
        "image_size": (config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        "batch_size": config.batch_size,
        "color_mode": "grayscale"
    }

    train_ds = tf.keras.utils.image_dataset_from_directory(**params)
    val_ds = tf.keras.utils.image_dataset_from_directory(**params)

    class_names = train_ds.class_names
    test_ds = val_ds.take(len(val_ds) // 2)
    val_ds = val_ds.skip(len(val_ds) - len(test_ds))

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return (train_ds, val_ds, test_ds), class_names


@task
def get_model(config: Config):
    model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 1)),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(config.num_classes)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()
    return model


def save_hist(history: Dict, path="saved-model/training_history.pickle"):
    with open(path, 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_model(model, path):
    model.save(path)


@task
def measure_performance(model, test_ds):
    # image classification report
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        predictions = model.predict(images)
        pred_labels = tf.argmax(predictions, axis=1)
        y_true.extend(labels.numpy().ravel())
        y_pred.extend(pred_labels.numpy().ravel())

    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_true, y_pred)
    print('Accuracy: %f' % accuracy)

    # precision tp / (tp + fp)
    precision = precision_score(y_true, y_pred, average="micro")
    print('Precision: %f' % precision)

    # recall: tp / (tp + fn)
    recall = recall_score(y_true, y_pred, average="micro")
    print('Recall: %f' % recall)

    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_true, y_pred, average="micro")
    print('F1 score: %f' % f1)

    # kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    print('Cohens kappa: %f' % kappa)

    # cm = confusion_matrix(y_true, y_pred)
    # f, ax = plt.subplots(figsize=(5, 5))
    # sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax)
    # plt.xlabel("y_pred")
    # plt.ylabel("y_true")
    # plt.show()


@flow(task_runner=SequentialTaskRunner())
def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("sense-hawk-assigment")

    config = Config()
    (train_ds, val_ds, test_ds), class_names = get_dataset(config)

    config.class_names = class_names
    config.num_classes = len(class_names)

    model = get_model(config)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs
    )

    save_model(model, config.model_path)
    save_hist(history.history, config.dict_path)

    measure_performance()


if __name__ == '__main__':
    main()
