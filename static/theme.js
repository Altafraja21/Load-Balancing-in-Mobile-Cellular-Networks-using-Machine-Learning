// static/theme.js

document.addEventListener("DOMContentLoaded", function() {
  const toggle = document.getElementById("themeToggle");
  const html = document.querySelector("html");

  if (!toggle) return; // safety if toggle missing

  // Load saved theme
  const savedTheme = localStorage.getItem("theme");
  if (savedTheme === "dark") {
    html.classList.add("dark-mode");
    toggle.checked = true;
    switchPlotlyTheme("plotly_dark");
  }

  // Toggle event
  toggle.addEventListener("change", function() {
    if (toggle.checked) {
      html.classList.add("dark-mode");
      localStorage.setItem("theme", "dark");
      switchPlotlyTheme("plotly_dark");
    } else {
      html.classList.remove("dark-mode");
      localStorage.setItem("theme", "light");
      switchPlotlyTheme("plotly");
    }
  });

  // Apply dark mode immediately after page load
  setTimeout(() => {
    if (toggle.checked) switchPlotlyTheme("plotly_dark");
  }, 800);
});

// 🔹 Function to re-theme Plotly charts dynamically
function switchPlotlyTheme(theme) {
  if (window.Plotly) {
    const plots = document.querySelectorAll(".js-plotly-plot");
    plots.forEach((p) => {
      Plotly.relayout(p, { template: theme });
    });
  }
}
