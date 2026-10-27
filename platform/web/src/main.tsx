import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./styles/base.css";
import "./lib/icons";

// ?mock=1 swaps fetch/WebSocket for fakes so the UI can be viewed without the backend.
let mock = new URLSearchParams(location.search).has("mock");
try { mock ||= sessionStorage.getItem("cpd-mock") === "1"; } catch { /* storage blocked */ }
if (mock) await import("./mock");

createRoot(document.getElementById("root")!).render(<StrictMode><App /></StrictMode>);
