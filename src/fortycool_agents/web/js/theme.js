/**
 * FortyCool theme toggle.
 *
 * Light/dark is controlled purely by the presence of the "dark" class on
 * <html>, which flips the CSS custom properties defined in
 * css/theme-vars.css (see that file). The actual color values live there —
 * this file only tracks the current choice and wires up any
 * [data-theme-toggle] button on the page.
 *
 * A tiny inline copy of `applyStoredTheme()` runs synchronously at the top
 * of <head> on every page (before Tailwind/CSS load) so there is no
 * flash-of-wrong-theme on load. This file re-applies the same logic
 * (harmless / idempotent) and adds the interactive bits.
 */
(function () {
  var STORAGE_KEY = "fortycool-theme";
  var DEFAULT_THEME = "dark"; // FortyCool ships dark-by-default; light is opt-in.

  function getStoredTheme() {
    try {
      return localStorage.getItem(STORAGE_KEY);
    } catch (e) {
      return null;
    }
  }

  function setStoredTheme(theme) {
    try {
      localStorage.setItem(STORAGE_KEY, theme);
    } catch (e) {
      /* private browsing / storage disabled — theme just won't persist */
    }
  }

  function currentTheme() {
    return document.documentElement.classList.contains("dark") ? "dark" : "light";
  }

  function applyTheme(theme) {
    document.documentElement.classList.toggle("dark", theme === "dark");
    document.documentElement.setAttribute("data-theme", theme);
    updateToggleIcons(theme);
  }

  function updateToggleIcons(theme) {
    var icons = document.querySelectorAll("[data-theme-toggle] .material-symbols-outlined");
    icons.forEach(function (icon) {
      icon.textContent = theme === "dark" ? "dark_mode" : "light_mode";
    });
    var buttons = document.querySelectorAll("[data-theme-toggle]");
    buttons.forEach(function (btn) {
      btn.setAttribute("aria-label", theme === "dark" ? "Switch to light theme" : "Switch to dark theme");
      btn.setAttribute("title", theme === "dark" ? "Switch to light theme" : "Switch to dark theme");
    });
  }

  function toggleTheme() {
    var next = currentTheme() === "dark" ? "light" : "dark";
    setStoredTheme(next);
    applyTheme(next);
    document.dispatchEvent(new CustomEvent("fortycool-theme-change", { detail: { theme: next } }));
  }

  function init() {
    applyTheme(getStoredTheme() || DEFAULT_THEME);
    document.querySelectorAll("[data-theme-toggle]").forEach(function (btn) {
      btn.addEventListener("click", toggleTheme);
    });
  }

  window.FortyCoolTheme = {
    toggle: toggleTheme,
    apply: applyTheme,
    current: currentTheme,
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
