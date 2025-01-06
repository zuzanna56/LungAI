// file-upload-handler.js
document.addEventListener("DOMContentLoaded", function () {
  const fileInput = document.getElementById("file-upload");
  const fileNameDisplay = document.getElementById("file-name-display");

  if (fileInput && fileNameDisplay) {
    fileInput.addEventListener("change", function () {
      const file = this.files[0];
      if (file) {
        fileNameDisplay.textContent = `Wybrany plik: ${file.name}`;
      } else {
        fileNameDisplay.textContent = "";
      }
    });
  }
});

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("upload-form");
  const spinner = document.getElementById("loading-spinner");

  if (form && spinner) {
    form.addEventListener("submit", () => {
      spinner.style.display = "block"; // Show spinner when the form is submitted
    });
  }
});
