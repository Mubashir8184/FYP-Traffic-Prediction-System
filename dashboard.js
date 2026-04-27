// dashboard - call api and update page (use with flask server)
var API = "/api";
var startTime = Date.now();
var trafficChart = null;
var predictionChart = null;
var currentRange = "24h";

function updateStats() {
  fetch(API + "/stats")
    .then(function (res) { return res.json(); })
    .then(function (data) {
      setText("current-count", data.current_count);
      setText("daily-total", data.daily_total);
      setText("peak-time", data.peak_time || "--:--");
      setText("accuracy", data.accuracy + "%");
      setText("last-update", data.last_update || "--:--:--");
      setText("data-points", data.data_points || 0);
    })
    .catch(function () {
      setText("last-update", "--:--:--");
    });
}

function updatePrediction() {
  fetch(API + "/predict/next-hour")
    .then(function (res) { return res.json(); })
    .then(function (data) {
      setText("next-hour-pred", data.prediction);
      var el = document.getElementById("confidence-level");
      var textEl = document.getElementById("confidence-text");
      if (el) el.style.width = (data.confidence || 0) + "%";
      if (textEl) textEl.textContent = (data.confidence || 0) + "% Confidence";
    })
    .catch(function () {});
}

function updateMetrics() {
  fetch(API + "/metrics")
    .then(function (res) { return res.json(); })
    .then(function (data) {
      setText("lr-mae", data.lr.mae);
      setText("lr-rmse", data.lr.rmse);
      setText("dt-mae", data.dt.mae);
      setText("dt-rmse", data.dt.rmse);
      setText("rf-mae", data.rf.mae);
      setText("rf-rmse", data.rf.rmse);
    })
    .catch(function () {});
}

function setText(id, value) {
  var el = document.getElementById(id);
  if (el) el.textContent = value;
}

function updateUptime() {
  var sec = Math.floor((Date.now() - startTime) / 1000);
  var h = Math.floor(sec / 3600);
  var m = Math.floor((sec % 3600) / 60);
  var s = sec % 60;
  setText("uptime", [h, m, s].map(function (n) { return (n < 10 ? "0" : "") + n; }).join(":"));
}

function refreshAll() {
  updateStats();
  updatePrediction();
  updateMetrics();
  updateTrafficChart(currentRange);
  updatePredictionChart();
}

function initCharts() {
  var trafficCtx = document.getElementById("trafficChart");
  var predictionCtx = document.getElementById("predictionChart");
  if (!trafficCtx || !predictionCtx) return;

  trafficChart = new Chart(trafficCtx.getContext("2d"), {
    type: "line",
    data: { labels: [], datasets: [{ label: "Vehicle Count", data: [], borderColor: "#3182ce", backgroundColor: "rgba(49,130,206,0.1)", fill: true, tension: 0.3 }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true }, x: { ticks: { maxTicksLimit: 12 } } } }
  });

  predictionChart = new Chart(predictionCtx.getContext("2d"), {
    type: "line",
    data: {
      labels: [],
      datasets: [
        { label: "Actual", data: [], borderColor: "#38a169", backgroundColor: "rgba(56,161,105,0.1)", fill: false, tension: 0.3 },
        { label: "Predicted", data: [], borderColor: "#d69e2e", borderDash: [5, 5], fill: false, tension: 0.3 }
      ]
    },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "top" } }, scales: { y: { beginAtZero: true }, x: { ticks: { maxTicksLimit: 12 } } } }
  });
}

function updateTrafficChart(range) {
  currentRange = range || currentRange;
  fetch(API + "/chart/traffic?range=" + (range || currentRange))
    .then(function (res) { return res.json(); })
    .then(function (data) {
      if (trafficChart && data.labels && data.values) {
        trafficChart.data.labels = data.labels;
        trafficChart.data.datasets[0].data = data.values;
        trafficChart.update();
      }
    })
    .catch(function () {});
}

function updatePredictionChart() {
  fetch(API + "/chart/prediction")
    .then(function (res) { return res.json(); })
    .then(function (data) {
      if (predictionChart && data.labels) {
        predictionChart.data.labels = data.labels;
        predictionChart.data.datasets[0].data = data.actual || [];
        predictionChart.data.datasets[1].data = data.predicted || [];
        predictionChart.update();
      }
      if (typeof data.confidence === "number") {
        var el = document.getElementById("confidence-level");
        var textEl = document.getElementById("confidence-text");
        if (el) el.style.width = data.confidence + "%";
        if (textEl) textEl.textContent = data.confidence + "% Confidence";
      }
    })
    .catch(function () {});
}

if (typeof Chart !== "undefined") initCharts();
refreshAll();
setInterval(updateStats, 10000);
setInterval(updateUptime, 1000);

document.querySelectorAll(".time-btn").forEach(function (btn) {
  btn.addEventListener("click", function () {
    document.querySelectorAll(".time-btn").forEach(function (b) { b.classList.remove("active"); });
    this.classList.add("active");
    var range = this.getAttribute("data-range") || "24h";
    updateTrafficChart(range);
  });
});

var exportBtn = document.getElementById("export-data");
if (exportBtn) exportBtn.addEventListener("click", function () { window.location.href = API + "/export"; });

var refreshBtn = document.getElementById("refresh-data");
if (refreshBtn) refreshBtn.addEventListener("click", refreshAll);

var startPredBtn = document.getElementById("start-prediction");
if (startPredBtn) startPredBtn.addEventListener("click", updatePrediction);

var calibrateBtn = document.getElementById("calibrate-sensors");
if (calibrateBtn) {
  calibrateBtn.addEventListener("click", function () {
    alert("Calibrate Sensors: run calibration from your Arduino/sensor setup.");
  });
}
