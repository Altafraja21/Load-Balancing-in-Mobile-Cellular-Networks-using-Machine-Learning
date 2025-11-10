// static/demo.js
let demoInterval = null;
let progressInterval = null;
let algorithms = ["baseline", "heuristic", "kmeans"];
let index = 0;
let switchDelay = 6000; // ms

document.addEventListener("DOMContentLoaded", () => {
  const demoBtn = document.getElementById("demoButton");
  const stopBtn = document.getElementById("stopDemoBtn");
  const demoPanel = document.getElementById("demoPanel");
  const demoProgress = document.getElementById("demoProgress");
  const demoStatus = document.getElementById("demoStatus");

  if (!demoBtn) return;

  // Start/Stop toggle
  demoBtn.addEventListener("click", () => {
    if (demoInterval) {
      stopDemo();
      return;
    }
    startDemo();
  });

  // Stop button inside panel
  stopBtn.addEventListener("click", () => stopDemo());

  function startDemo() {
    const form = document.querySelector("form");
    const methodSelect = form.querySelector("select[name='method']");
    const algoCount = algorithms.length;

    demoBtn.textContent = "🛑 Stop Demo";
    demoBtn.classList.remove("btn-warning");
    demoBtn.classList.add("btn-danger");
    demoPanel.style.display = "block";

    let timeLeft = switchDelay / 1000;
    demoStatus.innerHTML = `Running: <b>${algorithms[index % algoCount].toUpperCase()}</b>`;
    updateProgress(timeLeft);

    progressInterval = setInterval(() => {
      timeLeft -= 0.1;
      updateProgress(timeLeft);
    }, 100);

    demoInterval = setInterval(() => {
      // Switch algorithm
      index++;
      const nextAlgo = algorithms[index % algoCount];
      methodSelect.value = nextAlgo;
      form.submit();
      timeLeft = switchDelay / 1000;
      demoStatus.innerHTML = `Running: <b>${nextAlgo.toUpperCase()}</b>`;
    }, switchDelay);
  }

  function stopDemo() {
    clearInterval(demoInterval);
    clearInterval(progressInterval);
    demoInterval = null;
    progressInterval = null;
    demoBtn.textContent = "🎬 Demo Mode";
    demoBtn.classList.remove("btn-danger");
    demoBtn.classList.add("btn-warning");
    demoPanel.style.display = "none";
  }

  function updateProgress(timeLeft) {
    const percent = ((switchDelay / 1000 - timeLeft) / (switchDelay / 1000)) * 100;
    demoProgress.style.width = `${percent}%`;
    demoProgress.setAttribute("aria-valuenow", percent);
    if (timeLeft <= 0) demoProgress.style.width = "0%";
    demoStatus.innerHTML = `Next: <b>${algorithms[(index + 1) % algorithms.length].toUpperCase()}</b> in ${timeLeft.toFixed(1)}s`;
  }
});
