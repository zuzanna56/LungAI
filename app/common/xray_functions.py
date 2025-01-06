import glob
import os
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import resample, shuffle
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def import_chest_xray_data():
    """
    Wczytywanie i przygotowywania danych z X-ray Chest Data
    """
    train_dir = "chest_xray/chest_xray/train"
    test_dir = "chest_xray/chest_xray/test"
    val_dir = "chest_xray/chest_xray/val"

    pneumonia = os.listdir("chest_xray/chest_xray/train/PNEUMONIA")
    pneumonia_dir = "chest_xray/chest_xray/train/PNEUMONIA"

    normal = os.listdir("chest_xray/chest_xray/train/NORMAL")
    normal_dir = "chest_xray/chest_xray/train/NORMAL"

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )
    training_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(224, 224), batch_size=16, class_mode="binary"
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    validation_generator = val_datagen.flow_from_directory(
        val_dir, target_size=(224, 224), batch_size=16, class_mode="binary"
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    test_generator = test_datagen.flow_from_directory(
        test_dir, target_size=(224, 224), batch_size=16, class_mode="binary"
    )

    return training_generator, validation_generator, test_generator


def import_nih_data():
    """
    Wczytywanie i przygotowywania danych z NIH Data
    """
    pattern = os.path.join("nih", "images*", "images", "*_*_*.png")

    image_paths = glob.glob(pattern)

    all_image_paths = {os.path.basename(x): x for x in image_paths}

    for dirname, _, filenames in os.walk("nih"):
        for filename in filenames:
            if filename.endswith(".png"):
                all_image_paths[filename] = os.path.join(dirname, filename)

    print(f"found: {len(all_image_paths)} images")
    data = pd.read_csv("nih/Data_Entry_2017.csv")

    data["Finding Labels"] = data["Finding Labels"].apply(lambda x: x.split("|"))
    mlb = MultiLabelBinarizer()
    data_labels = mlb.fit_transform(data["Finding Labels"])

    data["full_path"] = data["Image Index"].map(all_image_paths)

    target_size = (224, 224)
    batch_size = 64

    data["binary_label"] = data["Finding Labels"].apply(map_to_binary_labels)
    data = data[["Image Index", "binary_label", "full_path"]]

    pneumonia = data[data["binary_label"] == "pneumonia"]

    non_pneumonia = data[data["binary_label"] == "non-pneumonia"]

    non_pneumonia_downsampled = resample(
        non_pneumonia, replace=False, n_samples=(len(pneumonia)), random_state=4289
    )

    data_balanced = pd.concat([pneumonia, non_pneumonia_downsampled])

    train_data, temp_data = train_test_split(
        data_balanced,
        test_size=0.3,
        random_state=42,
        stratify=data_balanced["binary_label"],
    )

    test_data, validation_data = train_test_split(
        temp_data, test_size=1 / 3, random_state=42, stratify=temp_data["binary_label"]
    )

    train_datagen_nih = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    train_generator_nih = train_datagen_nih.flow_from_dataframe(
        dataframe=train_data,
        directory=None,
        x_col="full_path",
        y_col="binary_label",
        target_size=(224, 224),
        batch_size=64,
        class_mode="binary",
        shuffle=True,
    )

    validation_datagen_nih = ImageDataGenerator(rescale=1.0 / 255.0)

    validation_generator_nih = validation_datagen_nih.flow_from_dataframe(
        dataframe=validation_data,
        directory=None,
        x_col="full_path",
        y_col="binary_label",
        target_size=(224, 224),
        batch_size=64,
        class_mode="binary",
        shuffle=False,
    )
    test_datagen_nih = ImageDataGenerator(rescale=1.0 / 255.0)

    test_generator_nih = test_datagen_nih.flow_from_dataframe(
        dataframe=test_data,
        directory=None,
        x_col="full_path",
        y_col="binary_label",
        target_size=(224, 224),
        batch_size=64,
        class_mode="binary",
        shuffle=False,
    )
    return train_generator_nih, validation_generator_nih, test_generator_nih


def map_to_binary_labels(labels):
    """
    Pomocnicza funkcja, która sprawdza czy w liście etykiet jest słowo "pneumonia"
    """
    for label in labels:
        if "pneumonia" in label.lower():
            return "pneumonia"
    return "non-pneumonia"


def plot_training_metrics(history):
    """
    Generuje wykresy precyzji i recall na podstawie historii trenowania modelu.

    Parametry:
    history (History): Obiekt historii zwracany przez metodę model.fit(), zawierający dane dotyczące uczenia modelu.
    """
    # Wyodrębnienie metryk z historii trenowania
    precision = history.history["precision"]
    val_precision = history.history["val_precision"]
    recall = history.history["recall"]
    val_recall = history.history["val_recall"]

    # Liczba epok
    epochs = range(1, len(precision) + 1)

    plt.figure(figsize=(14, 6))

    # Wykres precyzji
    plt.subplot(1, 2, 1)
    plt.plot(epochs, precision, "b", label="Precyzja treningowa")
    plt.plot(epochs, val_precision, "r", label="Precyzja walidacyjna")
    plt.title("Precyzja treningowa i walidacyjna")
    plt.xlabel("Epoki")
    plt.ylabel("Precyzja")
    plt.legend()

    # Wykres recall
    plt.subplot(1, 2, 2)
    plt.plot(epochs, recall, "b", label="Recall treningowy")
    plt.plot(epochs, val_recall, "r", label="Recall walidacyjny")
    plt.title("Recall treningowy i walidacyjny")
    plt.xlabel("Epoki")
    plt.ylabel("Recall")
    plt.legend()

    plt.tight_layout()
    plt.show()


def get_gradcam_heatmap(
    model, conv_layer_name, last_layer_name, img_path, pred_index=None
):
    # Pobieranie warstwy konwolucyjnej i definiowanie modelu Gradient Tape
    conv_layer = model.get_layer(conv_layer_name)
    grad_model = Model(
        model.inputs, [conv_layer.output, model.get_layer(last_layer_name).output]
    )

    # Wczytywanie i przygotowanie obrazu
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img_array = img / 255.0
    img_temp = np.expand_dims(img_array, axis=0)

    # Predykcja klasy i pewności
    prediction_prob = model.predict(img_temp)[0][
        0
    ]  # Wartość prawdopodobieństwa dla klasy "1"
    predicted_class = int(
        prediction_prob > 0.5
    )  # 1 jeśli prawdopodobieństwo > 0.5, w przeciwnym razie 0
    confidence_percent = (
        prediction_prob * 100 if predicted_class == 1 else (1 - prediction_prob) * 100
    )

    # Tworzenie mapy ciepła (Grad-CAM)
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_temp)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalizacja mapy ciepła
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    gamma = 12.0
    heatmap = np.power(heatmap, gamma)  # Użycie NumPy
    heatmap = heatmap / np.max(heatmap)
    # Nakładanie mapy ciepła na obraz
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Zwracanie klasy, procentowej pewności predykcji i obrazu z mapą ciepła
    return predicted_class, confidence_percent, superimposed_img


def display_gradcam_result(model, last_conv_name, last_layer_name, img_path):
    """
    Automatyczna generacja i wyświetlanie mapy ciepła Grad-CAM wraz z informacjami o predykcji.

    Parametry:
    - model: Wytrenowany model sieci neuronowej.
    - last_conv_name: Nazwa ostatniej warstwy konwolucyjnej modelu.
    - last_layer_name: Nazwa ostatniej warstwy klasyfikacyjnej modelu.
    - img_path: Ścieżka do obrazu wejściowego.

    Wyświetla:
    - Mapę ciepła Grad-CAM z nałożonym oryginalnym obrazem.
    - Informacje o przewidywanej klasie i pewności.
    """
    # Generowanie mapy ciepła i predykcji
    predicted_class, confidence, heatmap_img = get_gradcam_heatmap(
        model, last_conv_name, last_layer_name, img_path
    )

    # Przygotowanie etykiety i wyniku
    class_label = (
        "Chory na zapalenie płuc"
        if predicted_class == 1
        else "Nie jest chory na zapalenie płuc"
    )
    title_text = f"{class_label} (Pewność: {confidence:.2f}%)"

    # Wyświetlenie obrazu z mapą ciepła
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB))
    plt.title(title_text)
    plt.axis("off")
    plt.show()


def combined_generator(generator1, generator2):
    """
    Generator łączący dane z dwóch innych generatorów.
    """
    while True:
        # Pobranie danych z pierwszego generatora
        batch_x1, batch_y1 = next(generator1)
        # Pobranie danych z drugiego generatora
        batch_x2, batch_y2 = next(generator2)

        # Połączenie danych (X) i etykiet (y)
        combined_x = np.concatenate((batch_x1, batch_x2), axis=0)
        combined_y = np.concatenate((batch_y1, batch_y2), axis=0)
        combined_x, combined_y = shuffle(combined_x, combined_y)

        # Zwrócenie połączonych danych
        yield combined_x, combined_y


def compile_model(checkpoint_name):
    # Load the DenseNet121 model with ImageNet weights, without the top fully connected layer
    base_model = DenseNet121(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )

    # Add a global average pooling layer to reduce dimensions
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Add a fully connected layer with a single output and sigmoid activation for binary classification
    output = Dense(1, activation="sigmoid")(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model with Adam optimizer and binary crossentropy loss
    model.compile(
        optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
        loss="binary_crossentropy",
        metrics=["accuracy", "precision", "recall"],
    )

    # Callbacks
    # 1. ReduceLROnPlateau to reduce the learning rate when the validation loss plateaus
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=2, min_lr=1e-6, verbose=1
    )

    # 2. EarlyStopping to stop training early if validation loss doesn't improve
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    )

    # 3. ModelCheckpoint to save the model with the lowest validation loss
    model_checkpoint = ModelCheckpoint(
        f"{checkpoint_name}.keras", monitor="val_loss", save_best_only=True, verbose=1
    )
    return model, reduce_lr, early_stopping, model_checkpoint


def add_noise(image, noise_type="salt_pepper", noise_level=0.05):
    """
    Dodaje szum do obrazu.
    - noise_type: Typ szumu ("gaussian" lub "salt_pepper").
    - noise_level: Poziom szumu (procent pikseli zakłóconych).
    """
    if noise_type == "gaussian":
        noise = np.random.normal(0, noise_level, image.shape)  # Szum Gaussowski
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 1)  # Utrzymanie zakresu pikseli [0, 1]
    elif noise_type == "salt_pepper":
        noisy_image = image.copy()
        # Procent zakłóconych pikseli
        num_salt = int(noise_level * image.size * 0.5)
        num_pepper = int(noise_level * image.size * 0.5)

        # Dodanie soli (białe piksele)
        coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
        noisy_image[coords[0], coords[1], :] = 1

        # Dodanie pieprzu (czarne piksele)
        coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
        noisy_image[coords[0], coords[1], :] = 0
    else:
        raise ValueError("Invalid noise type. Use 'gaussian' or 'salt_pepper'.")
    return noisy_image


def display_integrated_griadients(
    model, conv_layer_name, last_layer_name, img_path, method="gradcam", pred_index=None
):
    # Wczytywanie i przygotowanie obrazu
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img_array = img / 255.0
    img_temp = np.expand_dims(img_array, axis=0)

    # Predykcja klasy i pewności
    predictions = model.predict(img_temp)
    prediction_prob = predictions[0][0]  # Prawdopodobieństwo dla klasy "1"
    predicted_class = int(
        prediction_prob > 0.5
    )  # 1 jeśli prawdopodobieństwo > 0.5, w przeciwnym razie 0
    confidence_percent = (
        prediction_prob * 100 if predicted_class == 1 else (1 - prediction_prob) * 100
    )

    # Integrated Gradients
    baseline = np.zeros_like(img_temp)  # Wartości bazowe
    steps = 50
    interpolated_images = tf.concat(
        [
            baseline + (step / steps) * (img_temp - baseline)
            for step in range(steps + 1)
        ],
        axis=0,
    )

    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        preds = model(interpolated_images)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, interpolated_images)
    avg_grads = tf.reduce_mean(grads, axis=0)
    heatmap = tf.reduce_sum(avg_grads * (img_temp - baseline), axis=-1)[0]

    # Normalizacja mapy ciepła
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # Nakładanie mapy ciepła na obraz
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Zwracanie klasy, procentowej pewności predykcji i obrazu z mapą ciepła
    return predicted_class, confidence_percent, superimposed_img


def get_saliency_map(model, img_path, target_class_index=None):
    """
    Generuje mapę uwag (Saliency Map) dla podanego modelu i obrazu.

    Parametry:
    - model: Wytrenowany model sieci neuronowej.
    - img_path: Ścieżka do obrazu wejściowego.
    - target_class_index: (Opcjonalny) Indeks klasy, dla której obliczyć gradient.
                          Jeśli None, wybierana jest klasa z najwyższym prawdopodobieństwem.

    Zwraca:
    - predicted_class: Przewidziana klasa.
    - confidence_percent: Pewność w procentach dla przewidzianej klasy.
    - saliency_map: Obraz mapy uwag.
    """
    import cv2
    import numpy as np
    import tensorflow as tf

    # Wczytanie obrazu w oryginalnym rozmiarze
    original_img = cv2.imread(img_path)
    original_size = original_img.shape[:2]

    # Przetworzenie obrazu do rozmiaru wymaganego przez model
    img = cv2.resize(original_img, (224, 224))
    img_array = img / 255.0
    img_temp = np.expand_dims(img_array, axis=0)

    # Obliczanie predykcji modelu
    predictions = model.predict(img_temp)[0]
    if target_class_index is None:
        target_class_index = np.argmax(predictions)
    predicted_class = target_class_index
    confidence_percent = predictions[target_class_index] * 100

    # Tworzenie mapy uwag za pomocą GradientTape
    img_tensor = tf.convert_to_tensor(img_temp, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        preds = model(img_tensor)
        target_output = preds[:, target_class_index]

    # Obliczanie gradientów
    gradients = tape.gradient(target_output, img_tensor)[0]
    saliency_map = tf.reduce_max(tf.abs(gradients), axis=-1)

    # Normalizacja mapy uwag
    saliency_map = tf.maximum(saliency_map, 0)
    saliency_map = saliency_map / tf.reduce_max(saliency_map)
    saliency_map = saliency_map.numpy()

    # Skalowanie do oryginalnego rozmiaru obrazu
    saliency_map = cv2.resize(saliency_map, (original_size[1], original_size[0]))
    saliency_map = np.uint8(255 * saliency_map)

    # Zwrócenie wyników
    return predicted_class, confidence_percent, saliency_map


def display_saliency_result(model, img_path, target_class_index=None):
    """
    Generuje i wyświetla mapę uwag (Saliency Map) wraz z informacjami o predykcji.

    Parametry:
    - model: Wytrenowany model sieci neuronowej.
    - img_path: Ścieżka do obrazu wejściowego.
    - target_class_index: (Opcjonalny) Indeks klasy, dla której obliczyć gradient.

    Wyświetla:
    - Mapę uwag nałożoną na obraz wejściowy.
    - Informacje o przewidywanej klasie i pewności.
    """
    import cv2
    import matplotlib.pyplot as plt

    # Generowanie mapy uwag i predykcji
    predicted_class, confidence, saliency_map = get_saliency_map(
        model, img_path, target_class_index
    )

    # Wczytanie oryginalnego obrazu
    original_img = cv2.imread(img_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # Przygotowanie etykiety i wyniku
    if predicted_class == 0:
        out = "Chory na zapalenie płuc"
    else:
        out = "Nie wykryto zapalenia płuc"
    class_label = f"Klasa: {out} (Pewność: {confidence:.2f}%)"

    # Wyświetlenie obrazu z mapą uwag
    plt.figure(figsize=(8, 8))
    plt.imshow(original_img)
    plt.imshow(saliency_map, cmap="hot", alpha=0.5)  # Nakładanie mapy uwag
    plt.title(class_label)
    plt.axis("off")
    plt.show()
    return predicted_class, confidence, saliency_map


class CustomDataGenerator(tf.keras.utils.Sequence):
    """
    Generator słuzacy do dodania szumu do zdjec
    """

    def __init__(self, generator, noise_type="gaussian", noise_level=0.05):
        self.generator = generator
        self.noise_type = noise_type
        self.noise_level = noise_level

    def __len__(self):
        return len(self.generator)

    def __getitem__(self, idx):
        # Pobranie batcha danych
        batch_x, batch_y = self.generator[idx]
        # Dodanie szumu do batcha
        noisy_batch_x = np.array(
            [add_noise(x, self.noise_type, self.noise_level) for x in batch_x]
        )
        return noisy_batch_x, batch_y

    def __iter__(self):
        # Definicja iteracji
        for i in range(len(self)):
            yield self[i]
