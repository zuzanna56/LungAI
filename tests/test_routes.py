import os


def test_index_page(client):
    """Test the index page loads correctly."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"klasyfikacja" in response.data  # Verify button or form text


def test_redirection_after_file_upload(client, test_image_path):
    """Test file upload and redirection to results."""
    with open(test_image_path, "rb") as img:
        response = client.post(
            "/", data={"image": img}, content_type="multipart/form-data"
        )
    assert response.status_code == 302  # Redirection to results page
    assert "/results?" in response.headers["Location"]


def test_results_page(client, test_image_path):
    """Test results page displays after upload."""
    filename = "example_image.png"
    response = client.get(
        f"/results?filename={filename}g&heatmap_filename=model_1_heatmap_{filename}&class_label=chory&confidence=97.42"
    )
    assert response.status_code == 200
    assert b"97.42" in response.data  # Verify confidence is displayed
    assert b"chory" in response.data


def test_generate_report(client, test_image_path):
    """Test generating a PDF report."""
    filename = "example_image.png"
    response = client.get(f"/generate_report?filename={filename}")
    assert response.status_code == 200
    assert response.content_type == "application/pdf"
