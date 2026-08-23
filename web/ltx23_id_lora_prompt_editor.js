import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// Displays the resolved [VISUAL]/[SPEECH]/[SOUNDS] fields on
// LTXV23IDLoraPromptEditor after each run, following ComfyUI-Custom-Scripts'
// ShowText convention: destroy and rebuild the display widgets fresh on
// every execution, no "only if empty" guard.
//
// This is deliberately kept separate from the node's real
// visual_override/speech_override/sounds_override widgets - those are
// plain editable inputs the JS never touches. Writing the generated value
// into the SAME widget used as the manual override (the first version of
// this file did) creates an unfixable ambiguity: once auto-filled, the box
// is non-empty, so the Python side can no longer tell "leftover auto-fill
// from last run" from "a deliberate override" - it just sees non-empty and
// treats it as an override, so edits appear to "stick" after the first run
// even when nothing was intentionally typed. Keeping display and override
// as separate widgets removes that ambiguity entirely.
const DISPLAY_FIELDS = ["visual", "speech", "sounds"];

app.registerExtension({
    name: "CCTech.LTXV23IDLoraPromptEditor",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "LTXV23IDLoraPromptEditor") return;

        function populate(message) {
            // Remove only the display widgets WE added on a previous run -
            // never touch the real override widgets or anything else.
            if (this._ccDisplayWidgets) {
                for (const w of this._ccDisplayWidgets) {
                    const idx = this.widgets.indexOf(w);
                    if (idx > -1) {
                        w.onRemove?.();
                        this.widgets.splice(idx, 1);
                    }
                }
            }
            this._ccDisplayWidgets = [];

            for (const field of DISPLAY_FIELDS) {
                const values = message?.[field];
                if (!values || !values.length) continue;
                const w = ComfyWidgets["STRING"](
                    this, `${field}_generated`, ["STRING", { multiline: true }], app
                ).widget;
                w.inputEl.readOnly = true;
                w.inputEl.style.opacity = 0.6;
                w.value = values[0];
                this._ccDisplayWidgets.push(w);
            }

            requestAnimationFrame(() => {
                const sz = this.computeSize();
                if (sz[0] < this.size[0]) sz[0] = this.size[0];
                if (sz[1] < this.size[1]) sz[1] = this.size[1];
                this.onResize?.(sz);
                app.graph.setDirtyCanvas(true, false);
            });
        }

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            populate.call(this, message);
        };
    },
});
