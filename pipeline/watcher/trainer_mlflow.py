from typing import Dict
from uuid import uuid4

import mlflow
import tensorflow as tf
from PIL import Image
from prefect import flow, task
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Accuracy, Precision, AUC, Recall
import numpy as np

import pickle

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, \
    confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt

from dataclasses import dataclass


@dataclass
class Config:
    DATASET_FILE_PATH = "mnist_test.csv"
    DATASET_ROOT_PATH = "/app/monitored_dataset"
    IMAGE_WIDTH = 28
    IMAGE_HEIGHT = 28
    batch_size = 4
    DATASET_SPLIT = "0.7:.15:0.15"
    epochs = 1
    num_classes: int = 0
    class_names = []

    model_path: str = "/app/saved-model/best-model"
    dict_path: str = "/app/saved-model/training_history.pickle"


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
    # [Accuracy(), Precision(), Recall(), AUC()]
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics="accuracy")

    model.summary()
    return model


@task
def save_hist(history: Dict, path="saved-model/training_history.pickle"):
    with open(path, 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)


@task
def save_model(model, path):
    model.save(path)


@task
def get_measure_performance_report(model, test_ds):
    # image classification report
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        predictions = model.predict(images)
        pred_labels = tf.argmax(predictions, axis=1)
        y_true.extend(labels.numpy().ravel())
        y_pred.extend(pred_labels.numpy().ravel())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    report = {}
    # accuracy: (tp + tn) / (p + n)
    report['accuracy'] = accuracy_score(y_true, y_pred)

    # precision tp / (tp + fp)
    report['precision'] = precision_score(y_true, y_pred, average="micro")

    # recall: tp / (tp + fn)
    report['recall'] = recall_score(y_true, y_pred, average="micro")

    # f1: 2 tp / (2 tp + fp + fn)
    report['f1'] = f1_score(y_true, y_pred, average="micro")

    # kappa
    report['kappa'] = cohen_kappa_score(y_true, y_pred)

    # cm = confusion_matrix(y_true, y_pred)
    # f, ax = plt.subplots(figsize=(5, 5))
    # sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax)
    # plt.xlabel("y_pred")
    # plt.ylabel("y_true")
    # plt.savefig('/tmp/cm.jpg')

    return report


@task
def save_to_mlflow(model, test_ds, config, history, report):
    test_loss, test_acc = model.evaluate(test_ds)
    print("test_acc:", test_acc)
    print("test_loss:", test_loss)

    # Save as TensorFlow SavedModel format (MLflow Keras default)
    mlflow.keras.log_model(model, "keras-model", registered_model_name=uuid4().hex)

    # write model summary
    summary = []
    model.summary(print_fn=summary.append)
    summary = "\n".join(summary)
    with open("model_summary.txt", "w") as f:
        f.write(summary)
    mlflow.log_artifact("model_summary.txt")

    # TODO: save the history diagram
    # cm_arr = Image.open('/tmp/cm.jpg')
    # mlflow.log_image(cm_arr, "confusion.matrix")

    # save the report
    for k, v in report.items():
        mlflow.log_metric(k, v)


@flow
def train_model():
    mlflow.set_tracking_uri("sqlite:////app/mlflow/mlflow.db")
    mlflow.set_experiment("sense-hawk-assigment")

    print("Current registry uri: {}".format(mlflow.get_registry_uri()))
    print("Current tracking uri: {}".format(mlflow.get_tracking_uri()))

    mlflow.tensorflow.autolog()

    config = Config()

    with mlflow.start_run() as run:
        mlflow.set_tag("version.mlflow", mlflow.__version__)
        mlflow.set_tag("version.tensorflow", tf.__version__)

        (train_ds, val_ds, test_ds), class_names = get_dataset(config)

        config.class_names = class_names
        config.num_classes = len(class_names)

        model = get_model(config)
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.epochs
        )
        report = get_measure_performance_report(model, test_ds)
        save_to_mlflow(model, test_ds, config, history.history, report)
