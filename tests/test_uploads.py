import os

import cv2
from bs4 import BeautifulSoup


def test_uploaded_file_saved(client, test_image_path):
    """Test that an uploaded file is saved correctly."""
    with open(test_image_path, "rb") as img:
        client.post("/", data={"image": img}, content_type="multipart/form-data")
    saved_file_path = os.path.join("tests/images/uploaded", "example_image.png")
    assert os.path.exists(saved_file_path)


def test_processed_heatmap_saved(client, test_image_path):
    """Test that a processed heatmap is saved correctly."""
    with open(test_image_path, "rb") as img:
        client.post("/", data={"image": img}, content_type="multipart/form-data")
    processed_heatmap_path = os.path.join(
        "tests/images/processed", "model_1_heatmap_example_image.png"
    )
    assert os.path.exists(processed_heatmap_path)
    assert cv2.imread(processed_heatmap_path) is not None


def test_incorrect_file_type(client, test_incorrect_image):
    """
    Testuje obsługę pliku o niepoprawnym typie wysyłanego do endpointu /upload.
    """
    with open(test_incorrect_image, "rb") as img:
        response = client.post(
            "/", data={"image": img}, content_type="multipart/form-data"
        )

    assert response.status_code == 200

    soup = BeautifulSoup(response.text, "html.parser")
    error_span = soup.find("span", class_="error")

    # Check if the text content matches the expected error message
    assert error_span is not None, "Error span not found in response"
    assert (
        error_span.text.strip() == "Only images can be passed"
    ), "Error message text does not match"


def test_upload_empty_request(client):
    """
    Testuje obsługę pustego żądania POST do endpointu /upload.
    """
    response = client.post("/upload", data={})  # Wysyłamy puste dane
    assert response.status_code == 404  # Oczekiwany kod błędu
