import logging
import os
import re

import cv2
import matplotlib
matplotlib.use("Agg")  # Ustawienie backendu na "Agg" przed importem matplotlib.pyplot

import matplotlib.pyplot as plt

import pdfkit
from flask import (Flask, make_response, redirect, render_template, request,
                   send_from_directory, session, url_for)
from flask_uploads import IMAGES, UploadSet, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, FileField, FileRequired
from wtforms import SubmitField

from app.common.model_configuration import MODEL, clear_directory
from app.common.xray_functions import (display_integrated_griadients,
                                       get_gradcam_heatmap, get_saliency_map)

IMAGES_DEST = {
    "original": "UPLOADED_IMAGES_DEST",
    "heatmap": "PROCESSED_IMAGES_DEST",
    "saliency_heatmap": "SALIENCY_IMAGES_DEST",
    "int_grad_heatmap": "INTEGRATED_GRADIENTS_IMAGES_DEST",
}


def create_app():
    """
    Tworzy i konfiguruje aplikację Flask.

    Returns:
        Flask: Skonfigurowana instancja aplikacji.
    """
    app = Flask(__name__)

    # Konfiguracja aplikacji
    app.config["SECRET_KEY"] = "secret_key"  # Klucz tajny do obsługi sesji

    # Ścieżka do przesyłanych obrazów
    app.config[IMAGES_DEST["original"]] = "uploaded_images"
    # Ścieżka do przetworzonych obrazów grad cam
    app.config[IMAGES_DEST["heatmap"]] = "processed_images"
    # Ścieżka do przetworzonych obrazów saliency
    app.config[IMAGES_DEST["saliency_heatmap"]] = "saliency_images"
    # Ścieżka do przetworzonych obrazów integrated gradients
    app.config[IMAGES_DEST["int_grad_heatmap"]] = "integrated_gradients_images"

    # ---------------Konfiguracja do pytest-----------------------------------
    # app.config["UPLOADED_IMAGES_DEST"] = "tests/images/uploaded"  # Testowa ścieżka
    # app.config["PROCESSED_IMAGES_DEST"] = "tests/images/processed"  # Testowa ścieżka
    # app.config["WTF_CSRF_ENABLED"] = False  # Do testów do wysyłania formularzy
    # ------------------------------------------------------------------------

    # Clearing previously uploaded images
    for dest in IMAGES_DEST.values():
        clear_directory(app.config[dest])

    # Konfiguracja obsługi przesyłanych obrazów
    images = UploadSet("images", IMAGES)
    configure_uploads(app, images)

    # Rejestracja tras i logiki aplikacji
    register_routes(app, images)

    return app


class UploadForm(FlaskForm):
    """
    Klasa formularza do przesyłania obrazów.

    Atrybuty:
        image (FileField): Pole wyboru pliku, obsługujące obrazy.
        submit (SubmitField): Przycisk wysyłania formularza.
    """

    image = FileField(
        validators=[
            FileAllowed(IMAGES, "Only images can be passed"),  # Walidacja typu pliku
            FileRequired("File field should not be empty"),  # Walidacja obecności pliku
        ]
    )
    submit = SubmitField("klasyfikacja")  # Przycisk wysyłania formularza


def register_routes(app, images):
    @app.route("/uploaded_images/<filename>")
    def get_file(filename):
        """
        Pobiera przesłany obraz z lokalizacji UPLOADED_IMAGES_DEST.

        Args:
            filename (str): Nazwa pliku obrazu.

        Returns:
            Response: Obraz w odpowiedzi HTTP.
        """
        return send_from_directory(app.config["UPLOADED_IMAGES_DEST"], filename)

    @app.route("/processed_images/<filename>")
    def get_processed_file(filename):
        """
        Pobiera przetworzony obraz z lokalizacji PROCESSED_IMAGES_DEST.

        Args:
            filename (str): Nazwa przetworzonego pliku.

        Returns:
            Response: Obraz w odpowiedzi HTTP.
        """
        return send_from_directory(app.config["PROCESSED_IMAGES_DEST"], filename)

    @app.route("/saliency_images/<filename>")
    def get_saliency_file(filename):
        """
        Pobiera przetworzony obraz z lokalizacji SALIENCY_IMAGES_DEST.

        Args:
            filename (str): Nazwa przetworzonego pliku.

        Returns:
            Response: Obraz w odpowiedzi HTTP.
        """
        return send_from_directory(app.config["SALIENCY_IMAGES_DEST"], filename)

    @app.route("/integrated_gradients_images/<filename>")
    def get_integrated_gradients_file(filename):
        """
        Pobiera przetworzony obraz z lokalizacji INTEGRATED_GRADIENTS_IMAGES_DEST.

        Args:
            filename (str): Nazwa przetworzonego pliku.

        Returns:
            Response: Obraz w odpowiedzi HTTP.
        """
        return send_from_directory(
            app.config["INTEGRATED_GRADIENTS_IMAGES_DEST"], filename
        )

    @app.route("/", methods=["GET", "POST"])
    def index():
        """
        Główna strona aplikacji obsługująca przesyłanie obrazów i klasyfikację.

        Returns:
            str: Strona HTML głównego widoku lub przekierowanie na wyniki.
        """
        form = UploadForm()  # Tworzenie instancji formularza
        if (
            form.validate_on_submit()
        ):  # Sprawdzenie, czy formularz został poprawnie przesłany

            match = re.search(
                r"([^/\\]+)$", form.image.data.filename
            )  # Matches the last part of the path, used in testing
            if match:
                filename = match.group(1)
            else:
                raise AttributeError("Filename not found")

            filename = images.save(form.image.data, name=filename)
            file_path = os.path.join(app.config["UPLOADED_IMAGES_DEST"], filename)

            # Wyliczanie pewnosci predykcji i tworznie mapy ciepla
            predicted_class, confidence, heatmap_img = get_gradcam_heatmap(
                MODEL["model"],
                MODEL["last_conv_name"],
                MODEL["last_layer_name"],
                file_path,
            )
            confidence = round(float(confidence), 2)  # Zaokrąglenie wartości pewności
            class_label = (
                "chory" if predicted_class == 1 else "nie wykryto zapalenia pluc"
            )

            # Zapis mapy cieplnej dla danego modelu
            heatmap_filename = f"{MODEL['model_name']}_heatmap_{filename}"
            heatmap_path = os.path.join(
                app.config["PROCESSED_IMAGES_DEST"], heatmap_filename
            )
            cv2.imwrite(heatmap_path, heatmap_img)

            # Tworzenie mapy dla integrated gradients
            pred_class_int_grad, int_grad_confidence, int_grad_heatmap = (
                display_integrated_griadients(
                    MODEL["model"],
                    MODEL["last_conv_name"],
                    MODEL["last_layer_name"],
                    file_path,
                )
            )
            int_grad_heatmap_filename = (
                f"{MODEL['model_name']}_int_grad_heatmap_{filename}"
            )
            int_grad_heatmap_path = os.path.join(
                app.config["INTEGRATED_GRADIENTS_IMAGES_DEST"],
                int_grad_heatmap_filename,
            )
            cv2.imwrite(int_grad_heatmap_path, int_grad_heatmap)

            # Tworzenie mapy dla saliency
            pred_class_saliency, saliency_confidence, saliency_heatmap = (
                get_saliency_map(MODEL["model"], file_path)
            )
            saliency_heatmap_filename = (
                f"{MODEL['model_name']}_saliency_heatmap_{filename}"
            )
            saliency_heatmap_path = os.path.join(
                app.config["SALIENCY_IMAGES_DEST"], saliency_heatmap_filename
            )

            original_img = cv2.imread(file_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(8, 8))
            plt.imshow(original_img)
            plt.imshow(saliency_heatmap, cmap="hot", alpha=0.5)  # Nakladanie heatmapy
            plt.axis("off")
            plt.savefig(saliency_heatmap_path, bbox_inches="tight", pad_inches=0)
            plt.close()

            # Zapisanie wyników w sesji
            session_content = {
                "original": filename,
                "heatmap": heatmap_filename,
                "int_grad_heatmap": int_grad_heatmap_filename,
                "saliency_heatmap": saliency_heatmap_filename,
                "confidence": confidence,
            }
            session["content"] = session_content

            # Przekierowanie na stronę wyników z najlepszym wynikiem
            return redirect(
                url_for(
                    "results",
                    filename=filename,
                    heatmap_filename=heatmap_filename,
                    class_label=class_label,
                    confidence=confidence,
                )
            )
        else:
            logging.error(f"Form submission failed. Errors: {form.errors}")

        return render_template("index.html", form=form)  # Renderowanie strony głównej

    @app.route("/results")
    def results():
        """
        Wyświetla wyniki analizy obrazu.

        Query Params:
            filename (str): Nazwa przesłanego obrazu.
            heatmap_filename (str): Nazwa pliku mapy cieplnej.
            class_label (str): Wynik klasyfikacji.
            confidence (float): Pewność klasyfikacji.

        Returns:
            str: Strona HTML z wynikami analizy.
        """
        filename = request.args.get("filename")  # Pobranie nazwy przesłanego pliku
        heatmap_filename = request.args.get(
            "heatmap_filename"
        )  # Pobranie nazwy mapy cieplnej
        class_label = request.args.get("class_label")  # Pobranie etykiety klasyfikacji
        confidence = request.args.get("confidence")  # Pobranie wartości pewności

        # Renderowanie strony z wynikami
        return render_template(
            "results.html",
            filename=filename,
            heatmap_filename=heatmap_filename,
            class_label=class_label,
            confidence=float(confidence),
        )

    @app.route("/generate_report")
    def generate_report():
        """
        Generuje raport PDF na podstawie wyników analizy obrazu.

        Query Params:
            filename (str): Nazwa przesłanego obrazu.

        Returns:
            Response: Raport PDF w odpowiedzi HTTP.
        """
        filename = request.args.get("filename")  # Pobranie nazwy przesłanego pliku

        # Pobranie wyników z sesji
        content = session.get("content", dict())

        for map_type in IMAGES_DEST.keys():
            img_dest = IMAGES_DEST[map_type]
            name = content[map_type]

            transformed_file_path = os.path.abspath(
                os.path.join(app.config[img_dest], name)
            ).replace("\\", "/")

            content[map_type] = transformed_file_path

        # Renderowanie treści raportu w HTML
        html_content = render_template(
            "generated_report.html",
            filename=filename,
            content=content,
        )

        # Generowanie raportu PDF z treści HTML
        pdf = pdfkit.from_string(
            html_content,
            False,
            options={
                "enable-local-file-access": "",
                "no-outline": None,
                "javascript-delay": 2000,
            },
        )

        # Przygotowanie odpowiedzi z plikiem PDF
        response = make_response(pdf)
        response.headers["Content-Type"] = "application/pdf"
        response.headers["Content-Disposition"] = "inline; filename=report.pdf"
        return response

    @app.route("/information")
    def information():
        """
        Wyświetla informacje na temat projektu

        Returns:
            str: Strona HTML z informacjami na temat projektu
        """

        return render_template("information.html")


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)  # tryb debuggowania
