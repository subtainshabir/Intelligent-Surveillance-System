const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const filenameLabel = document.getElementById("filename");
const form = document.getElementById("upload-form");
const submitButton = document.getElementById("submit-button");

function showFilename(file) {
  filenameLabel.textContent = file ? file.name : "";
}

fileInput.addEventListener("change", () => {
  showFilename(fileInput.files[0]);
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) {
    fileInput.files = e.dataTransfer.files;
    showFilename(file);
  }
});

form.addEventListener("submit", () => {
  submitButton.disabled = true;
  submitButton.textContent = "Running…";
});
