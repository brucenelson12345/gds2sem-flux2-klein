/**
 * gds2sem — open the GDS->SEM workflow automatically on startup.
 *
 * ComfyUI has no server-side "default workflow" setting: which graph you see
 * on load is browser-local state (localStorage `Comfy.PreviousWorkflow` plus
 * the draft tabs). This extension supplies the missing piece — on startup it
 * loads a named workflow, so a fresh browser, a kiosk, or a colleague who has
 * never opened the UI all land on the right graph.
 *
 * The workflow is looked up in two places, in order:
 *   1. user/default/workflows/<file>   — the mounted, editable copy, so you
 *      can change the graph without rebuilding the image
 *   2. the copy bundled into this extension at image-build time (fallback,
 *      so it still works against an empty user/ volume)
 *
 * Settings live under "gds2sem" in the ComfyUI settings dialog.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EXT = "gds2sem.DefaultWorkflow";
const S_FILE = "gds2sem.DefaultWorkflow.File";
const S_MODE = "gds2sem.DefaultWorkflow.Mode";
const BUNDLED = "/extensions/gds2sem_default_workflow/workflows/";
const FALLBACK_FILE = "gds2sem_klein4b_base_lora.json";

/** Read a setting across frontend versions (new API first, then legacy). */
function getSetting(id, dflt) {
  try {
    const v = app.extensionManager?.setting?.get(id);
    if (v !== undefined && v !== null && v !== "") return v;
  } catch (e) { /* fall through */ }
  try {
    const v = app.ui?.settings?.getSettingValue(id);
    if (v !== undefined && v !== null && v !== "") return v;
  } catch (e) { /* fall through */ }
  return dflt;
}

async function fetchWorkflow(name) {
  // 1. the user's own copy under user/default/workflows/
  try {
    const r = await api.fetchApi(
      `/userdata/${encodeURIComponent("workflows/" + name)}`);
    if (r.ok) return await r.json();
  } catch (e) { /* fall through to the bundled copy */ }
  // 2. the copy baked into the image
  try {
    const r = await fetch(BUNDLED + encodeURIComponent(name));
    if (r.ok) return await r.json();
  } catch (e) { /* nothing else to try */ }
  return null;
}

app.registerExtension({
  name: EXT,
  settings: [
    {
      id: S_MODE,
      category: ["gds2sem", "Startup", "When to load"],
      name: "Load the gds2sem workflow on startup",
      tooltip:
        "new-session: only when this browser has no previously-open workflow " +
        "(default — never clobbers work in progress). always: on every page " +
        "load. off: disable.",
      type: "combo",
      options: ["new-session", "always", "off"],
      defaultValue: "new-session",
    },
    {
      id: S_FILE,
      category: ["gds2sem", "Startup", "Workflow file"],
      name: "Workflow filename (in user/default/workflows)",
      type: "text",
      defaultValue: FALLBACK_FILE,
    },
  ],

  async setup() {
    const mode = getSetting(S_MODE, "new-session");
    if (mode === "off") return;

    // A returning browser has its own restored graph; don't overwrite it
    // unless the operator explicitly asked for "always".
    const returning = !!localStorage.getItem("Comfy.PreviousWorkflow");
    if (mode !== "always" && returning) return;

    const name = getSetting(S_FILE, FALLBACK_FILE) || FALLBACK_FILE;
    const data = await fetchWorkflow(name);
    if (!data) {
      console.warn(
        `[${EXT}] "${name}" not found in user/default/workflows/ or bundled ` +
        `at ${BUNDLED} — leaving the default graph in place.`);
      return;
    }
    try {
      await app.loadGraphData(data);
      console.log(`[${EXT}] loaded "${name}"`);
    } catch (e) {
      console.error(`[${EXT}] failed to load "${name}":`, e);
    }
  },
});
