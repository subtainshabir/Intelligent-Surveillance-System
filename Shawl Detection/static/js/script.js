// script.js
// Handles the upload form: detects whether the file is an image or a video,
// posts it to the right backend endpoint, and displays the result.

const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const fileNameLabel = document.getElementById("file-name");
const submitBtn = document.getElementById("submit-btn");
const loading = document.getElementById("loading");
const errorMessage = document.getElementById("error-message");

const placeholderText = document.getElementById("placeholder-text");
const resultImage = document.getElementById("result-image");
const resultStream = document.getElementById("result-stream");

fileInput.addEventListener("change", () => {
  fileNameLabel.textContent = fileInput.files.length
    ? fileInput.files[0].name
    : "Choose an image or video…";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = fileInput.files[0];
  if (!file) return;

  const isVideo = file.type.startsWith("video/");
  const isImage = file.type.startsWith("image/");

  if (!isVideo && !isImage) {
    showError("Please select a valid image or video file.");
    return;
  }

  resetResult();
  setLoading(true);

  try {
    const formData = new FormData();
    formData.append("file", file);

    const endpoint = isVideo ? "/predict/video" : "/predict/image";
    const response = await fetch(endpoint, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || "Something went wrong while processing the file.");
    }

    const data = await response.json();

    if (data.type === "image") {
      showImageResult(data.result_url);
    } else if (data.type === "video") {
      showVideoStream(data.stream_url);
    }
  } catch (err) {
    showError(err.message || "Unexpected error occurred.");
  } finally {
    setLoading(false);
  }
});

function setLoading(isLoading) {
  loading.classList.toggle("hidden", !isLoading);
  submitBtn.disabled = isLoading;
}

function resetResult() {
  errorMessage.classList.add("hidden");
  placeholderText.classList.add("hidden");
  resultImage.classList.add("hidden");
  resultStream.classList.add("hidden");
  resultImage.src = "";
  resultStream.src = "";
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.classList.remove("hidden");
  placeholderText.classList.remove("hidden");
}

function showImageResult(url) {
  resultImage.src = `${url}?t=${Date.now()}`; // cache-bust
  resultImage.classList.remove("hidden");
}

function showVideoStream(url) {
  resultStream.src = url; // MJPEG stream, browser renders it live via <img>
  resultStream.classList.remove("hidden");
}
