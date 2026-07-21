(() => {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const fileInfo = document.getElementById('fileInfo');
  const fileName = document.getElementById('fileName');
  const clearFile = document.getElementById('clearFile');
  const sourcePreview = document.getElementById('sourcePreview');
  const predictBtn = document.getElementById('predictBtn');

  const statusValue = document.getElementById('statusValue');
  const recDot = document.getElementById('recDot');

  const monitorEmpty = document.getElementById('monitorEmpty');
  const monitorProcessing = document.getElementById('monitorProcessing');
  const outputStream = document.getElementById('outputStream');

  const alertLog = document.getElementById('alertLog');
  const alertLogEmpty = document.getElementById('alertLogEmpty');
  const alertCount = document.getElementById('alertCount');

  let selectedFile = null;
  let alertsPollTimer = null;
  let renderedAlertCount = 0;

  // ---- Helpers ---------------------------------------------------

  function setStatus(text, state) {
    statusValue.textContent = text;
    statusValue.dataset.state = state || '';
  }

  function showToast(message) {
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
  }

  function resetMonitor() {
    monitorEmpty.hidden = false;
    monitorProcessing.hidden = true;
    outputStream.hidden = true;
    outputStream.removeAttribute('src');
    recDot.hidden = true;
    alertLog.innerHTML = '';
    alertLog.appendChild(alertLogEmpty);
    alertLogEmpty.textContent = 'No wrong-direction events detected yet.';
    alertCount.textContent = '0 events';
    renderedAlertCount = 0;
    if (alertsPollTimer) {
      clearInterval(alertsPollTimer);
      alertsPollTimer = null;
    }
  }

  // ---- File selection ---------------------------------------------

  function handleFile(file) {
    if (!file) return;
    selectedFile = file;
    fileName.textContent = file.name;
    fileInfo.hidden = false;
    dropzone.hidden = true;
    sourcePreview.src = URL.createObjectURL(file);
    predictBtn.disabled = false;
    resetMonitor();
    setStatus('READY', '');
  }

  fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

  ['dragenter', 'dragover'].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add('is-dragover');
    })
  );
  ['dragleave', 'drop'].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove('is-dragover');
    })
  );
  dropzone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  clearFile.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    fileInfo.hidden = true;
    dropzone.hidden = false;
    predictBtn.disabled = true;
    resetMonitor();
    setStatus('IDLE', '');
  });

  // ---- Run detection (upload -> live stream -> poll alerts) --------

  predictBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    predictBtn.disabled = true;
    predictBtn.textContent = 'Uploading…';
    setStatus('UPLOADING', 'processing');

    resetMonitor();
    monitorEmpty.hidden = true;
    monitorProcessing.hidden = false;

    const formData = new FormData();
    formData.append('video', selectedFile);

    try {
      const response = await fetch('/upload', { method: 'POST', body: formData });

      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}));
        throw new Error(errBody.detail || `Server returned ${response.status}`);
      }

      const { job_id, stream_url, alerts_url } = await response.json();
      goLive(stream_url, alerts_url);
    } catch (err) {
      console.error(err);
      setStatus('ERROR', 'error');
      monitorProcessing.hidden = true;
      monitorEmpty.hidden = false;
      showToast(`Upload failed: ${err.message}`);
      predictBtn.disabled = false;
      predictBtn.textContent = 'Run Detection';
    }
  });

  function goLive(streamUrl, alertsUrl) {
    setStatus('LIVE', 'processing');
    recDot.hidden = false;
    predictBtn.textContent = 'Streaming…';

    monitorProcessing.hidden = true;
    outputStream.src = streamUrl;
    outputStream.hidden = false;

    // The <img> tag renders each MJPEG frame as it arrives from the
    // backend — this is what makes the processed feed feel "live"
    // instead of waiting for a finished file.
    outputStream.onerror = () => {
      // Fires once the multipart stream closes (video ended) or on failure.
      finishLive();
    };

    alertsPollTimer = setInterval(() => pollAlerts(alertsUrl), 1000);
  }

  async function pollAlerts(alertsUrl) {
    try {
      const res = await fetch(alertsUrl);
      if (!res.ok) return;
      const data = await res.json();
      renderAlerts(data.alerts || []);

      if (data.status === 'done') {
        finishLive();
      }
    } catch (err) {
      // Non-fatal — just skip this poll tick.
      console.warn('alert poll failed', err);
    }
  }

  function renderAlerts(alerts) {
    if (alerts.length === renderedAlertCount) return;

    if (renderedAlertCount === 0 && alerts.length > 0) {
      alertLog.innerHTML = '';
    }

    for (let i = renderedAlertCount; i < alerts.length; i++) {
      const alert = alerts[i];
      const li = document.createElement('li');
      li.className = 'alert-row';
      li.innerHTML = `
        <span class="alert-row-badge">WRONG DIR</span>
        <span class="alert-row-detail">Track #${alert.track_id ?? '—'} · angle ${alert.angle_deg ?? '—'}°</span>
        <span class="alert-row-time">frame ${alert.frame_idx ?? '—'}</span>
      `;
      alertLog.appendChild(li);
    }

    renderedAlertCount = alerts.length;
    alertCount.textContent = `${alerts.length} event${alerts.length === 1 ? '' : 's'}`;
    if (alerts.length > 0) setStatus('ALERT', 'alert');
  }

  function finishLive() {
    if (alertsPollTimer) {
      clearInterval(alertsPollTimer);
      alertsPollTimer = null;
    }
    recDot.hidden = true;
    predictBtn.disabled = false;
    predictBtn.textContent = 'Run Detection';
    if (statusValue.dataset.state !== 'alert') setStatus('DONE', 'done');
  }
})();