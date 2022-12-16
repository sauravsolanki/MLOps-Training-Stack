import sys
import time
import logging
from collections import defaultdict
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler, FileSystemEventHandler
import os

from trainer_mlflow import train_model

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

class Watcher:
    def __init__(self, path):
        self.observer = Observer()
        self.path = path

        # create the folder
        create_folder("./data/dataset")
        create_folder("./data/mlflow/mlruns")
        create_folder("./data/prefect")
        create_folder("./data/state")
        create_folder("./data/saved-model")
        create_folder("./data/monitored_dataset")


    def run(self):
        logger_handler = LoggingEventHandler()
        folder_handler = DatasetFolderHandler()
        self.observer.schedule(logger_handler, self.path, recursive=True)
        self.observer.schedule(folder_handler, self.path, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(1)
        except:
            self.observer.stop()
            print("Error")

        self.observer.join()


class DatasetFolderHandler(FileSystemEventHandler):
    dataset_state = defaultdict(lambda: 0)

    @staticmethod
    def on_any_event(event):
        # if event.is_directory:
        #     return None
        # [Sun Nov 27 15:18:38 2022] noticed: [modified] on: [/home/saurav/Downloads/mlops/monitored_dataset/0]
        print(
            "[{}] noticed: [{}] on: [{}] ".format(
                time.asctime(), event.event_type, event.src_path
            )
        )

        # validate trigger
        if os.path.isdir(event.src_path):
            return

        changed_folder_path = event.src_path.rsplit("/", maxsplit=1)[0]
        changed_labeled_name = event.src_path.rsplit("/", maxsplit=1)[1]
        new_len = len(os.listdir(changed_folder_path))

        if new_len - DatasetFolderHandler.dataset_state[changed_labeled_name] >= int(os.getenv("MAX_IMAGE_WAIT")):
            DatasetFolderHandler.dataset_state[changed_labeled_name] = new_len
            print("Retraining Model")
            train_model()
        else:
            print(f"Files Changes is not more than {int(os.getenv('MAX_IMAGE_WAIT'))}")


if __name__ == "__main__":
    path = "/app/monitored_dataset"
    w = Watcher(path)
    while True:
        try:
            w.run()
        except FileNotFoundError as f:
            print(f)
        except Exception as f:
            print(f)
