import { Icon } from "@iconify/react";
import { setTheme, useTheme } from "../lib/theme";

export function ThemeToggle() {
  const theme = useTheme();
  const next = theme === "dark" ? "light" : "dark";
  return (
    <button type="button" className="icon-btn" onClick={() => setTheme(next)} aria-label={`Switch to ${next} theme`}>
      <Icon icon={theme === "dark" ? "solar:sun-2-linear" : "solar:moon-linear"} width={18} aria-hidden="true" />
    </button>
  );
}
