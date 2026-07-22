(() => {
  const fileInput = document.getElementById("video-input");
  const filePickerLabel = document.getElementById("file-picker-label");
  const uploadHint = document.getElementById("upload-hint");
  const startBtn = document.getElementById("start-btn");
  const resetBtn = document.getElementById("reset-btn");

  const videoFeed = document.getElementById("video-feed");
  const videoPlaceholder = document.getElementById("video-placeholder");

  const tableBody = document.getElementById("vehicle-table-body");

  const systemStatus = document.getElementById("system-status");
  const statusText = document.getElementById("status-text");

  const STATUS_POLL_MS = 1000;

  let pollTimer = null;
  let hasUploadedVideo = false;

  // --------------------------------------------------------------
  // Upload
  // --------------------------------------------------------------
  fileInput.addEventListener("change", async () => {
    const file = fileInput.files[0];
    if (!file) return;

    filePickerLabel.textContent = file.name;
    uploadHint.textContent = "Uploading...";
    startBtn.disabled = true;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/upload", { method: "POST", body: formData });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Upload failed");
      }

      hasUploadedVideo = true;
      uploadHint.textContent = `Ready: ${data.filename}`;
      startBtn.disabled = false;
      setSystemStatus("ready");
    } catch (err) {
      uploadHint.textContent = err.message;
      startBtn.disabled = true;
      hasUploadedVideo = false;
    }
  });

  // --------------------------------------------------------------
  // Start detection
  // --------------------------------------------------------------
  startBtn.addEventListener("click", async () => {
    if (!hasUploadedVideo) return;

    startBtn.disabled = true;
    uploadHint.textContent = "Starting detection...";

    try {
      const res = await fetch("/start_detection", { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to start detection");

      videoFeed.src = `/video_feed?t=${Date.now()}`;
      videoFeed.classList.add("is-active");
      videoPlaceholder.classList.add("is-hidden");

      uploadHint.textContent = "Detection running.";
      setSystemStatus("live");
      startPolling();
    } catch (err) {
      uploadHint.textContent = err.message;
      startBtn.disabled = false;
    }
  });

  // --------------------------------------------------------------
  // Reset
  // --------------------------------------------------------------
  resetBtn.addEventListener("click", async () => {
    stopPolling();

    try {
      await fetch("/reset", { method: "POST" });
    } catch (err) {
      // Best-effort; still reset the UI locally.
    }

    videoFeed.src = "";
    videoFeed.classList.remove("is-active");
    videoPlaceholder.classList.remove("is-hidden");

    fileInput.value = "";
    filePickerLabel.textContent = "Upload Video";
    uploadHint.textContent = "No video selected.";
    hasUploadedVideo = false;
    startBtn.disabled = true;

    renderTable([]);
    setSystemStatus("idle");
  });

  // --------------------------------------------------------------
  // Status polling
  // --------------------------------------------------------------
  function startPolling() {
    stopPolling();
    pollTimer = setInterval(fetchStatus, STATUS_POLL_MS);
    fetchStatus();
  }

  function stopPolling() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  async function fetchStatus() {
    try {
      const res = await fetch("/vehicle_status");
      if (!res.ok) return;
      const data = await res.json();

      renderTable(data.vehicles || []);

      if (!data.is_processing && data.frame_number > 0) {
        setSystemStatus("ready");
        uploadHint.textContent = "Detection finished.";
        stopPolling();
      }
    } catch (err) {
      // Network hiccup - try again on the next tick.
    }
  }

  // --------------------------------------------------------------
  // Rendering
  // --------------------------------------------------------------
  function renderTable(vehicles) {
    if (!vehicles.length) {
      tableBody.innerHTML = `<tr class="empty-row"><td colspan="4">No vehicles tracked yet.</td></tr>`;
      return;
    }

    tableBody.innerHTML = vehicles
      .map((v) => {
        const rowClass =
          v.status === "suspicious" ? "row--alert" : v.status === "warning" ? "row--warning" : "";
        const badgeClass =
          v.status === "suspicious" ? "badge--alert" : v.status === "warning" ? "badge--warning" : "badge--moving";
        const stateLabel =
          v.status === "suspicious" ? "ALERT" : v.status === "warning" ? "Watching" : "Monitoring";

        return `
          <tr class="${rowClass}">
            <td>ID ${v.id}</td>
            <td><span class="badge ${badgeClass}">${v.class_name}</span></td>
            <td>${formatDuration(v.stationary_time)}</td>
            <td>${stateLabel}</td>
          </tr>
        `;
      })
      .join("");
  }

  function formatDuration(seconds) {
    if (!seconds || seconds <= 0) return "—";
    return `${Math.round(seconds)} sec`;
  }

  function setSystemStatus(state) {
    systemStatus.classList.remove("is-live", "is-ready");
    if (state === "live") {
      systemStatus.classList.add("is-live");
      statusText.textContent = "LIVE";
    } else if (state === "ready") {
      systemStatus.classList.add("is-ready");
      statusText.textContent = "READY";
    } else {
      statusText.textContent = "IDLE";
    }
  }
})();
