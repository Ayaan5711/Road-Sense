// async function fetchLog() {
//   try {
//     const response = await fetch("/log");
//     const data = await response.json();
//     const logContainer = document.getElementById("log-container");

//     logContainer.innerHTML = "";

//     data.lines.forEach((line) => {
//       const div = document.createElement("div");
//       div.className = "log-line";
//       div.textContent = line.trim();
//       logContainer.appendChild(div);
//     });
//   } catch (err) {
//     console.error("Error fetching logs:", err);
//   }
// }

// setInterval(fetchLog, 3000);
// window.onload = fetchLog;
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
async function fetchAccidentLog() {
  try {
    const response = await fetch("/accident-log");
    const data = await response.json();
    const accidentContainer = document.getElementById("accident-container");

    accidentContainer.innerHTML = "";

    if (
      data.lines.length === 0 ||
      (data.lines.length === 1 && data.lines[0].trim() === "")
    ) {
      const div = document.createElement("div");
      div.className = "accident-line";
      div.textContent = "No accident data available here.";
      accidentContainer.appendChild(div);
    } else {
      data.lines.forEach((line) => {
        const div = document.createElement("div");
        div.className = "accident-line";
        div.textContent = line.trim();
        accidentContainer.prepend(div); // Show newest on top
      });
    }
  } catch (err) {
    console.error("Error fetching accident logs:", err);
  }
}


setInterval(fetchLog, 3000);
setInterval(fetchAccidentLog, 3000);

window.onload = () => {
  fetchLog();
  fetchAccidentLog();
};
