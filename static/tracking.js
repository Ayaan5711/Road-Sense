// Panel Toggle Functionality
document.addEventListener("DOMContentLoaded", function () {
  // Panel toggle functionality
  const leftToggle = document.querySelector(".left-toggle");
  const rightToggle = document.querySelector(".right-toggle");
  const leftPanel = document.getElementById("left-panel");
  const rightPanel = document.getElementById("right-panel");

  leftToggle.addEventListener("click", () => {
    leftPanel.classList.toggle("active");
    leftToggle.classList.toggle("active");
  });

  rightToggle.addEventListener("click", () => {
    rightPanel.classList.toggle("active");
    rightToggle.classList.toggle("active");
  });

  // Initialize all panels
  initializeTrafficPanel();
  initializeAlertPanel();
  initializeAccidentPanel();
  initializeAnalyticsPanel();

  // Set up periodic updates
  setInterval(updateAllPanels, 5000);
});

async function updateAllPanels() {
  await Promise.all([
    updateTrafficPanel(),
    updateAlertPanel(),
    updateAccidentPanel(),
    updateAnalyticsPanel(),
  ]);
}

// Traffic Panel Functions
async function initializeTrafficPanel() {
  await updateTrafficPanel();
}

async function updateTrafficPanel() {
  const response = await fetch("/api/traffic-stats");
  const data = await response.json();

  const trafficPanel = document.querySelector(".left-panel .panel-content");
  trafficPanel.innerHTML = data.zones
    .map(
      (zone) => `
      <div class="zone-card">
          <h3>${zone.location}</h3>
          <div class="zone-stats">
              <p>Total Vehicles: ${zone.total_vehicles}</p>
              <p>Density Level: <span class="density-${zone.density_level.toLowerCase()}">${
        zone.density_level
      }</span></p>
              <p>Last Update: ${zone.last_update}</p>
              <p>Average Speed: ${zone.average_speed} km/h</p>
          </div>
          <div class="vehicle-breakdown">
              <h4>Vehicle Breakdown</h4>
              <div class="breakdown-grid">
                  <div>Cars: ${zone.vehicle_breakdown.cars}</div>
                  <div>Buses: ${zone.vehicle_breakdown.buses}</div>
                  <div>Trucks: ${zone.vehicle_breakdown.trucks}</div>
                  <div>Two-wheelers: ${
                    zone.vehicle_breakdown.two_wheelers
                  }</div>
              </div>
          </div>
      </div>
  `
    )
    .join("");
}

// Alert Panel Functions
async function initializeAlertPanel() {
  await updateAlertPanel();
}

async function updateAlertPanel() {
  const response = await fetch("/api/alerts");
  const data = await response.json();

  const alertPanel = document.querySelector(".right-panel .panel-content");
  alertPanel.innerHTML = `
      <div class="alert-section">
          <h3>Overspeeding Alerts</h3>
          ${data.overspeeding
            .map(
              (alert) => `
              <div class="alert-item">
                  <p>Vehicle ${alert.vehicle_id} - ${alert.speed} km/h in ${alert.zone}</p>
                  <small>${alert.timestamp}</small>
              </div>
          `
            )
            .join("")}
      </div>
      <div class="alert-section">
          <h3>Stopped Vehicles</h3>
          ${data.stopped_vehicles
            .map(
              (vehicle) => `
              <div class="alert-item">
                  <p>Vehicle ${vehicle.vehicle_id} stopped for ${vehicle.duration} in ${vehicle.zone}</p>
              </div>
          `
            )
            .join("")}
      </div>
      <div class="alert-section">
          <h3>Proximity Alerts</h3>
          ${data.proximity_alerts
            .map(
              (alert) => `
              <div class="alert-item">
                  <p>Vehicles ${alert.vehicle1} and ${alert.vehicle2} - ${alert.distance} in ${alert.zone}</p>
              </div>
          `
            )
            .join("")}
      </div>
      <div class="alert-section">
          <h3>Accident Alerts</h3>
          ${data.accidents
            .map(
              (accident) => `
              <div class="alert-item accident">
                  <p>Accident in ${
                    accident.zone
                  } involving vehicles ${accident.vehicles.join(", ")}</p>
                  <small>${accident.time}</small>
              </div>
          `
            )
            .join("")}
      </div>
  `;
}

// Accident Detection Panel Functions
async function initializeAccidentPanel() {
  await updateAccidentPanel();
}

async function updateAccidentPanel() {
  const response = await fetch("/api/accident-detection");
  const data = await response.json();

  const accidentPanel = document.getElementById("accident-panel");
  accidentPanel.innerHTML = data.accidents
    .map(
      (accident) => `
      <div class="accident-detection-card">
          <div class="accident-header">
              <h3>Accident Detected</h3>
              <span class="confidence-badge">${accident.confidence_level}</span>
          </div>
          <div class="accident-content">
              <div class="accident-image">
                  <div class="no-snapshot" style="display: none;">
                      <i class="fas fa-image"></i>
                      <p>Accident Snapshot is not available</p>
                  </div>
                  <img src="${accident.snapshot_url || ""}" 
                       alt="Accident Snapshot" 
                       onerror="this.style.display='none'; this.parentElement.querySelector('.no-snapshot').style.display='flex';"
                       onload="this.style.display='block'; this.parentElement.querySelector('.no-snapshot').style.display='none';">
              </div>
              <div class="accident-details">
                  <p>Zone: ${accident.zone}</p>
                  <p>Time: ${accident.time}</p>
                  <p>Confidence Score: ${accident.confidence_score}</p>
                  <p>Prediction: ${accident.prediction}</p>
                  <div class="accident-actions">
                      <button class="replay-btn">Replay Last 10s</button>
                      <button class="dispatch-btn">Simulate Emergency Dispatch</button>
                  </div>
              </div>
          </div>
      </div>
  `
    )
    .join("");
}

// Analytics Panel Functions
async function initializeAnalyticsPanel() {
  await updateAnalyticsPanel();
}

async function updateAnalyticsPanel() {
  const response = await fetch("/api/analytics");
  const data = await response.json();

  const analyticsPanel = document.getElementById("analytics-panel");
  analyticsPanel.innerHTML = ""; // Clear existing content

  // Vehicle Count Over Time Chart
  const vehicleCountContainer = document.createElement("div");
  vehicleCountContainer.className = "chart-container vehicle-count-chart";
  vehicleCountContainer.innerHTML =
    '<h3 class="chart-title">Vehicle Count Over Time</h3>';
  const vehicleCountCanvas = document.createElement("canvas");
  vehicleCountContainer.appendChild(vehicleCountCanvas);
  analyticsPanel.appendChild(vehicleCountContainer);

  new Chart(vehicleCountCanvas, {
    type: "line",
    data: {
      labels: data.vehicle_count_over_time.labels,
      datasets: [
        {
          label: "Vehicle Count",
          data: data.vehicle_count_over_time.data,
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59, 130, 246, 0.1)",
          tension: 0.4,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: "top",
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          grid: {
            color: "rgba(0, 0, 0, 0.1)",
          },
        },
        x: {
          grid: {
            display: false,
          },
        },
      },
    },
  });

  // Speed Distribution Chart
  const speedContainer = document.createElement("div");
  speedContainer.className = "chart-container speed-distribution-chart";
  speedContainer.innerHTML = '<h3 class="chart-title">Speed Distribution</h3>';
  const speedCanvas = document.createElement("canvas");
  speedContainer.appendChild(speedCanvas);
  analyticsPanel.appendChild(speedContainer);

  new Chart(speedCanvas, {
    type: "bar",
    data: {
      labels: data.speed_distribution.ranges,
      datasets: [
        {
          label: "Number of Vehicles",
          data: data.speed_distribution.counts,
          backgroundColor: [
            "rgba(34, 197, 94, 0.8)",
            "rgba(59, 130, 246, 0.8)",
            "rgba(234, 179, 8, 0.8)",
            "rgba(249, 115, 22, 0.8)",
            "rgba(239, 68, 68, 0.8)",
            "rgba(139, 92, 246, 0.8)",
          ],
          borderColor: [
            "rgb(34, 197, 94)",
            "rgb(59, 130, 246)",
            "rgb(234, 179, 8)",
            "rgb(249, 115, 22)",
            "rgb(239, 68, 68)",
            "rgb(139, 92, 246)",
          ],
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          display: false,
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          grid: {
            color: "rgba(0, 0, 0, 0.1)",
          },
        },
        x: {
          grid: {
            display: false,
          },
        },
      },
    },
  });

  // Vehicle Type Distribution Chart
  const typeContainer = document.createElement("div");
  typeContainer.className = "chart-container vehicle-type-chart";
  typeContainer.innerHTML =
    '<h3 class="chart-title">Vehicle Type Distribution</h3>';
  const typeCanvas = document.createElement("canvas");
  typeContainer.appendChild(typeCanvas);
  analyticsPanel.appendChild(typeContainer);

  new Chart(typeCanvas, {
    type: "doughnut",
    data: {
      labels: data.vehicle_type_distribution.labels,
      datasets: [
        {
          data: data.vehicle_type_distribution.data.map((arr) =>
            arr.reduce((a, b) => a + b, 0)
          ),
          backgroundColor: [
            "rgba(59, 130, 246, 0.8)",
            "rgba(16, 185, 129, 0.8)",
            "rgba(245, 158, 11, 0.8)",
            "rgba(239, 68, 68, 0.8)",
          ],
          borderColor: [
            "rgb(59, 130, 246)",
            "rgb(16, 185, 129)",
            "rgb(245, 158, 11)",
            "rgb(239, 68, 68)",
          ],
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: "right",
        },
      },
    },
  });

  // Zone Congestion Chart
  const congestionContainer = document.createElement("div");
  congestionContainer.className = "chart-container congestion-chart";
  congestionContainer.innerHTML =
    '<h3 class="chart-title">Zone Congestion Levels</h3>';
  const congestionCanvas = document.createElement("canvas");
  congestionContainer.appendChild(congestionCanvas);
  analyticsPanel.appendChild(congestionContainer);

  new Chart(congestionCanvas, {
    type: "bar",
    data: {
      labels: data.zone_congestion.zones,
      datasets: [
        {
          label: "Congestion Level",
          data: data.zone_congestion.congestion_levels,
          backgroundColor: "rgba(239, 68, 68, 0.8)",
          borderColor: "rgb(239, 68, 68)",
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          display: false,
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          grid: {
            color: "rgba(0, 0, 0, 0.1)",
          },
        },
        x: {
          grid: {
            display: false,
          },
        },
      },
    },
  });
}

// Event Listeners
document.addEventListener("click", function (e) {
  if (e.target.classList.contains("replay-btn")) {
    // Implement replay functionality
    console.log("Replay requested");
  }

  if (e.target.classList.contains("dispatch-btn")) {
    // Implement emergency dispatch simulation
    console.log("Emergency dispatch simulated");
  }
});
