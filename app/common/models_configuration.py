import os

from tensorflow.keras.models import load_model


def clear_directory(directory):
    """
    Czyści folder przechowujący zdjęcia
    :param directory: nazwa destynacji
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


# Funkcja pomocnicza do walidacji ścieżki pliku
def validate_file_path(file_path):
    """
    Sprawdza, czy plik istnieje.
    :param file_path: Ścieżka do pliku.
    :return: Ścieżka, jeśli istnieje.
    :raises FileNotFoundError: Jeśli plik nie istnieje.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Plik nie istnieje: {file_path}")
    return file_path


# Funkcja pomocnicza do ładowania modelu
def load_model_with_path(model_path):
    """
    Ładuje model z podanej ścieżki po uprzedniej walidacji.
    :param model_path: Ścieżka do modelu.
    :return: Załadowany model.
    """
    validated_path = validate_file_path(model_path)
    return load_model(validated_path)


# Definiowanie katalogu bazowego (root projektu)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Konfiguracja modelu

model_config = {
    "path": "models/chexnet_combined.keras",
    "last_conv_name": "conv5_block16_2_conv",
    "last_layer_name": "dense",
    "model_name": "model_1",
}


model_path = os.path.join(BASE_DIR, model_config["path"])
loaded_model = load_model_with_path(model_path)
MODEL = {
    "model": loaded_model,
    "last_conv_name": model_config["last_conv_name"],
    "last_layer_name": model_config["last_layer_name"],
    "model_name": model_config["model_name"],
}
print(f"Załadowano model: {model_config['model_name']} z {model_path}")
