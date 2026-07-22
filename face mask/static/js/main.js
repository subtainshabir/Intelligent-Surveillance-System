// ---------- Section switching ----------
const optionCards = document.querySelectorAll(".option-card");
const panels = document.querySelectorAll(".detection-panel");

optionCards.forEach((card) => {
  card.addEventListener("click", () => {
    optionCards.forEach((c) => c.classList.remove("active"));
    card.classList.add("active");

    const target = card.dataset.target;
    panels.forEach((p) => p.classList.toggle("d-none", p.id !== target));
  });
});

function showAlert(el, message, type = "danger") {
  el.textContent = message;
  el.className = `alert alert-${type} mt-3`;
}

// ---------- Image Detection ----------
const imageForm = document.getElementById("imageForm");
const imageAlert = document.getElementById("imageAlert");
const imageResult = document.getElementById("imageResult");

imageForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = document.getElementById("imageInput").files[0];
  if (!file) return;

  imageAlert.classList.add("d-none");
  imageResult.classList.add("d-none");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/detect/image", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) {
      showAlert(imageAlert, data.error || "Detection failed.");
      return;
    }

    document.getElementById("imgOutput").src = data.output_image + "?t=" + Date.now();
    document.getElementById("imgDetections").textContent = data.total_detections;
    document.getElementById("imgInferenceTime").textContent = data.inference_time_ms + " ms";
    imageResult.classList.remove("d-none");
  } catch (err) {
    showAlert(imageAlert, "Could not reach the server. Please try again.");
  }
});

// ---------- Video Detection ----------
const videoForm = document.getElementById("videoForm");
const videoAlert = document.getElementById("videoAlert");
const videoResult = document.getElementById("videoResult");
let videoPollTimer = null;

videoForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = document.getElementById("videoInput").files[0];
  if (!file) return;

  videoAlert.classList.add("d-none");
  videoResult.classList.add("d-none");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/detect/video", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) {
      showAlert(videoAlert, data.error || "Video upload failed.");
      return;
    }

    // Start the live MJPEG stream
    document.getElementById("videoStream").src = "/video_feed?t=" + Date.now();
    videoResult.classList.remove("d-none");

    if (videoPollTimer) clearInterval(videoPollTimer);
    videoPollTimer = setInterval(pollVideoStats, 1000);
  } catch (err) {
    showAlert(videoAlert, "Could not reach the server. Please try again.");
  }
});

async function pollVideoStats() {
  try {
    const res = await fetch("/video/stats");
    const data = await res.json();
    document.getElementById("videoStatus").textContent = data.status;
    document.getElementById("videoFps").textContent = data.fps;
    document.getElementById("videoDetections").textContent = data.detections;

    if (data.status === "finished" || data.status === "error") {
      clearInterval(videoPollTimer);
    }
  } catch (err) {
    clearInterval(videoPollTimer);
  }
}

// ---------- Live Camera ----------
const cameraAlert = document.getElementById("cameraAlert");
const cameraStream = document.getElementById("cameraStream");
let cameraPollTimer = null;

document.getElementById("startCameraBtn").addEventListener("click", async () => {
  cameraAlert.classList.add("d-none");
  try {
    const res = await fetch("/camera/start");
    const data = await res.json();

    if (!res.ok) {
      showAlert(cameraAlert, data.error || "Camera unavailable.");
      return;
    }

    cameraStream.src = "/camera/feed?t=" + Date.now();

    if (cameraPollTimer) clearInterval(cameraPollTimer);
    cameraPollTimer = setInterval(pollCameraStats, 1000);
  } catch (err) {
    showAlert(cameraAlert, "Could not reach the server. Please try again.");
  }
});

document.getElementById("stopCameraBtn").addEventListener("click", async () => {
  try {
    await fetch("/camera/stop");
  } finally {
    cameraStream.src = "";
    clearInterval(cameraPollTimer);
    document.getElementById("cameraStatus").textContent = "stopped";
    document.getElementById("cameraFps").textContent = "0";
    document.getElementById("cameraDetections").textContent = "0";
  }
});

async function pollCameraStats() {
  try {
    const res = await fetch("/camera/stats");
    const data = await res.json();
    document.getElementById("cameraStatus").textContent = data.status;
    document.getElementById("cameraFps").textContent = data.fps;
    document.getElementById("cameraDetections").textContent = data.detections;

    if (!data.active) {
      clearInterval(cameraPollTimer);
    }
  } catch (err) {
    clearInterval(cameraPollTimer);
  }
}
