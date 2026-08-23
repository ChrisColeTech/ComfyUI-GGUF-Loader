import { app } from "../../scripts/app.js";

// Populates LTXV23IDLoraPromptEditor's three override widgets with the
// parsed [VISUAL]/[SPEECH]/[SOUNDS] text right after the node runs - but
// only when a widget is still empty, so a value you typed in yourself is
// never overwritten by a later run.
app.registerExtension({
    name: "CCTech.LTXV23IDLoraPromptEditor",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "LTXV23IDLoraPromptEditor") return;

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            const fields = {
                visual_override: message?.visual,
                speech_override: message?.speech,
                sounds_override: message?.sounds,
            };
            for (const [name, values] of Object.entries(fields)) {
                if (!values || !values.length) continue;
                const widget = this.widgets?.find((w) => w.name === name);
                if (widget && !widget.value) {
                    widget.value = values[0];
                }
            }

            this.onResize?.(this.size);
            app.graph.setDirtyCanvas(true, true);
        };
    },
});
