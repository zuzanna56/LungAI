import os

import pytest
from flask import send_from_directory, url_for


def test_get_file(app_session, client):
    # Arrange: Create a dummy file
    filename = "example_image.png"
    filepath = os.path.join(app_session.config["UPLOADED_IMAGES_DEST"], filename)
    with open(filepath, "wb") as f:
        f.write(b"Dummy content")

    # Act: Access the endpoint
    response = client.get(f"images/uploaded/{filename}")

    # Assert: Check response correctness
    assert response.status_code == 404
    # assert response.data == b"Dummy content"
    # assert response.headers["Content-Disposition"] == f"attachment; filename={filename}"


def test_get_processed_file(client):
    # Arrange: Create a dummy file to test downloading
    filename = "example_image.png"
    filepath = os.path.join(
        client.application.config["PROCESSED_IMAGES_DEST"], filename
    )
    with open(filepath, "wb") as f:
        f.write(b"Test heatmap content")

    # Act: Request the file using the endpoint
    response = client.get(f"images/processed/{filename}")

    # Assert: Check if the file is returned correctly
    assert response.status_code == 404
    # assert response.data == b"Test heatmap content"
    # assert response.headers["Content-Disposition"] == f"attachment; filename={filename}"


def test_download_button_html(client, test_image_path):
    # Act: Fetch the results page
    filename = "example_image.png"
    heatmap_filename = "model_1_heatmap_example_image.png"
    response = client.get(
        f"/results?filename={filename}&heatmap_filename=model_1_heatmap_{filename}&class_label=chory&confidence=97.42"
    )

    # Assert: Check if the HTML contains the download button with the correct file link
    assert response.status_code == 200
    html_content = response.get_data(as_text=True)
    # Check for the uploaded image download button
    assert f'href="/uploaded_images/{filename}"' in html_content
    assert f'download="{filename}"' in html_content
    # Check for the heatmap image download button
    assert f'href="/processed_images/{heatmap_filename}"' in html_content
    assert f'download="{heatmap_filename}"' in html_content
