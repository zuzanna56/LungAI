import os

import pytest

from app.app import create_app


def clear_directory(directory):
    """Helper function to clear all files in the given directory."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


@pytest.fixture
def app_session():
    """Fixture for creating the Flask app instance."""
    app = create_app()
    app.config.update(
        {
            "TESTING": True,
            "UPLOADED_IMAGES_DEST": "tests/images/uploaded/",
            "PROCESSED_IMAGES_DEST": "tests/images/processed/",
        }
    )

    # Ensure test directories exist
    os.makedirs(app.config["UPLOADED_IMAGES_DEST"], exist_ok=True)
    os.makedirs(app.config["PROCESSED_IMAGES_DEST"], exist_ok=True)

    # Clear the directories before the test
    clear_directory(app.config["UPLOADED_IMAGES_DEST"])
    clear_directory(app.config["PROCESSED_IMAGES_DEST"])

    yield app


@pytest.fixture
def client(app_session):
    """Fixture for creating a test client."""
    return app_session.test_client()


@pytest.fixture
def test_image_path():
    """Fixture providing the path to a test image."""
    return "tests/images/example_image.png"


@pytest.fixture
def test_incorrect_image():
    """Fixture providing the path to a incorrect test image format."""
    return "tests/images/example_image.txt"
