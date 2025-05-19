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

    // Clear existing content
    logContainer.innerHTML = "";

    // Add each line to the container
    data.lines.forEach((line) => {
      if (line.trim()) {  // Only add non-empty lines
        const div = document.createElement("div");
        div.className = "log-line";
        div.textContent = line.trim();
        logContainer.appendChild(div);
      }
    });

    // Scroll to bottom to show latest data
    logContainer.scrollTop = logContainer.scrollHeight;
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
        if (line.trim()) {  // Only add non-empty lines
          const div = document.createElement("div");
          div.className = "accident-line";
          div.textContent = line.trim();
          accidentContainer.appendChild(div);
        }
      });
    }

    // Scroll to bottom
    accidentContainer.scrollTop = accidentContainer.scrollHeight;
  } catch (err) {
    console.error("Error fetching accident logs:", err);
  }
}

// Initial fetch
window.onload = () => {
  fetchLog();
  fetchAccidentLog();
};

// Set up periodic updates with shorter interval for more real-time feel
setInterval(fetchLog, 1000);  // Update every second
setInterval(fetchAccidentLog, 1000);

function togglePanel(side) {
    const panel = document.getElementById(`${side}-panel`);
    const toggle = document.querySelector(`.${side}-toggle`);
    const textSpan = toggle.querySelector('span');
    
    panel.classList.toggle('active');
    
    // Update toggle button icon and text visibility
    const icon = toggle.querySelector('i');
    if (side === 'left') {
        icon.classList.toggle('fa-chevron-right');
        icon.classList.toggle('fa-chevron-left');
        // Move the left toggle button
        if (panel.classList.contains('active')) {
            toggle.style.left = '450px';
        } else {
            toggle.style.left = '0';
        }
    } else {
        icon.classList.toggle('fa-chevron-left');
        icon.classList.toggle('fa-chevron-right');
        // Move the right toggle button
        if (panel.classList.contains('active')) {
            toggle.style.right = '450px';
        } else {
            toggle.style.right = '0';
        }
    }

    // Toggle text visibility
    if (panel.classList.contains('active')) {
        textSpan.style.display = 'none';
    } else {
        textSpan.style.display = 'inline';
    }

    // Scroll to bottom when panel is opened
    if (panel.classList.contains('active')) {
        const container = side === 'left' ? 
            document.getElementById('log-container') : 
            document.getElementById('accident-container');
        container.scrollTop = container.scrollHeight;
    }
}
