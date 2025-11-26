import os

class Config:
    # Hyperparameters
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EPOCHS = 100

    # File paths
    DATASET_PATH = os.path.join("data", "dataset.csv")
    MODEL_SAVE_PATH = os.path.join("models", "trained_model.h5")
    LOGS_PATH = os.path.join("logs")

    # Other constants
    RANDOM_SEED = 42
    VALIDATION_SPLIT = 0.2

    @staticmethod
    def get_dataset_path():
        return Config.DATASET_PATH

    @staticmethod
    def get_model_save_path():
        return Config.MODEL_SAVE_PATH

    @staticmethod
    def get_logs_path():
        return Config.LOGS_PATH