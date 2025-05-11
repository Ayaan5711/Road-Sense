async function fetchLog() {
  try {
    const response = await fetch("/log");
    const data = await response.json();
    const logContainer = document.getElementById("log-container");

    logContainer.innerHTML = "";

    data.lines.forEach((line) => {
      const div = document.createElement("div");
      div.className = "log-line";
      div.textContent = line.trim();
      logContainer.appendChild(div);
    });
  } catch (err) {
    console.error("Error fetching logs:", err);
  }
}

setInterval(fetchLog, 3000);
window.onload = fetchLog;
