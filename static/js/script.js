/**
 * script.js — PlacementPredict Interactive Client Controller
 * Manages Expandable M1–M6 Header Navigation, Dashboard Tabs, Presets, and Shortcuts.
 */

// Toggle Expandable Module Dropdown in Navigation Header
function toggleModuleNav(moduleId, event) {
  if (event) {
    event.stopPropagation();
  }
  const allDropdowns = document.querySelectorAll(".nav-module-dropdown");
  const targetDropdown = document.getElementById("nav-dropdown-" + moduleId);

  allDropdowns.forEach(dd => {
    if (dd !== targetDropdown) {
      dd.classList.remove("open");
    }
  });

  if (targetDropdown) {
    targetDropdown.classList.toggle("open");
  }
}

// Close Dropdowns on Click Outside
document.addEventListener("click", function (event) {
  if (!event.target.closest(".nav-module-dropdown")) {
    document.querySelectorAll(".nav-module-dropdown").forEach(dd => {
      dd.classList.remove("open");
    });
  }
});

// Toggle Mobile Navigation Drawer
function toggleMobileNav() {
  const navBar = document.getElementById("main-nav-bar");
  if (navBar) {
    navBar.classList.toggle("mobile-open");
  }
}

// Switch Active Module on Dashboard (M1 to M6)
function selectModule(moduleId) {
  if (typeof moduleId === "number" || (typeof moduleId === "string" && !moduleId.startsWith("m"))) {
    moduleId = "m" + moduleId;
  }
  moduleId = moduleId.toLowerCase();

  // Update Module Selector Cards on Dashboard
  document.querySelectorAll(".module-selector-card").forEach(card => {
    if (card.dataset.module === moduleId) {
      card.classList.add("active");
    } else {
      card.classList.remove("active");
    }
  });

  // Display ONLY the Concepts View for Selected Module
  document.querySelectorAll(".module-view-container").forEach(view => {
    if (view.id === "module-view-" + moduleId) {
      view.classList.add("active");
    } else {
      view.classList.remove("active");
    }
  });

  // Synchronize Browser Hash
  if (history.replaceState) {
    history.replaceState(null, null, "#" + moduleId);
  } else {
    location.hash = "#" + moduleId;
  }
}

// Form Preset Helpers
function setFormValue(id, val) {
  const el = document.getElementById(id);
  if (el) el.value = val;
}

function loadSafeProfile() {
  setFormValue("CGPA", "8.8");
  setFormValue("AttendancePercent", "92");
  setFormValue("Internships", "2");
  setFormValue("Projects", "3");
  setFormValue("Workshops", "3");
  setFormValue("Certifications", "2");
  setFormValue("AptitudeTestScore", "88");
  setFormValue("SoftSkillsRating", "4.5");
  setFormValue("CodingTestScore", "90");
  setFormValue("MockInterviewScore", "85");
  setFormValue("CollegeTier", "Tier1");
  setFormValue("Stream", "CS");
  setFormValue("HistoryOfBacklogs", "No");
}

function loadRiskProfile() {
  setFormValue("CGPA", "5.1");
  setFormValue("AttendancePercent", "55");
  setFormValue("Internships", "0");
  setFormValue("Projects", "0");
  setFormValue("Workshops", "0");
  setFormValue("Certifications", "0");
  setFormValue("AptitudeTestScore", "38");
  setFormValue("SoftSkillsRating", "2.0");
  setFormValue("CodingTestScore", "25");
  setFormValue("MockInterviewScore", "30");
  setFormValue("CollegeTier", "Tier3");
  setFormValue("Stream", "Mechanical");
  setFormValue("HistoryOfBacklogs", "Yes");
}

function loadDefaultProfile() {
  setFormValue("CGPA", "7.5");
  setFormValue("AttendancePercent", "80");
  setFormValue("Internships", "1");
  setFormValue("Projects", "2");
  setFormValue("Workshops", "2");
  setFormValue("Certifications", "2");
  setFormValue("AptitudeTestScore", "72");
  setFormValue("SoftSkillsRating", "3.8");
  setFormValue("CodingTestScore", "70");
  setFormValue("MockInterviewScore", "68");
  setFormValue("CollegeTier", "Tier2");
  setFormValue("Stream", "CS");
  setFormValue("HistoryOfBacklogs", "No");
}

// DOM Initialization
document.addEventListener("DOMContentLoaded", function () {
  let initialModule = "m1";
  const validMods = ["m1", "m2", "m3", "m4", "m5", "m6"];

  const urlParams = new URLSearchParams(window.location.search);
  const modParam = urlParams.get("module");
  if (modParam) {
    let clean = modParam.toLowerCase();
    if (!clean.startsWith("m")) clean = "m" + clean;
    if (validMods.includes(clean)) initialModule = clean;
  } else if (window.location.hash) {
    const cleanHash = window.location.hash.replace("#", "").toLowerCase();
    if (validMods.includes(cleanHash)) {
      initialModule = cleanHash;
    }
  }

  // Activate dashboard tab if element exists
  const hasSelectorCards = document.querySelector(".module-selector-card");
  if (hasSelectorCards && validMods.includes(initialModule)) {
    selectModule(initialModule);
  }

  // Global Ctrl + P Print Shortcut
  document.addEventListener("keydown", function (e) {
    if ((e.ctrlKey || e.metaKey) && (e.key === "p" || e.key === "P")) {
      e.preventDefault();
      window.print();
    }
  });
});
