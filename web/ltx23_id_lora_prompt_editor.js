import { app } from "../../scripts/app.js";

// Writes LTXV23IDLoraPromptEditor's resolved visual/speech/sounds back into
// its three editable widgets after each run.
//
// Unconditional on purpose - no "only if the widget is empty" guard. The
// keep-the-edit-or-refresh decision is made server-side in Python (see the
// node's docstring: it compares this run's `source` against the one it
// remembered for this node id), so whatever arrives here is already the
// resolved truth - either the user's own edit echoed back, or a fresh parse
// because `source` actually changed.
//
// An earlier version put that decision here instead, filling a widget only
// while it was still empty. That cannot work: once auto-filled the widget is
// non-empty forever, so it populated on the first run and looked frozen
// after that. Content alone cannot distinguish "the user typed this" from
// "we typed this last run"; only Python's remembered `source` can.
//
// onExecuted (not onNodeCreated) matters for multiline STRING: those are DOM
// widgets whose `.value` setter routes through options.setValue, updating
// both the real <textarea> and the reactive widget store - but only once the
// widget has been registered into that store. By onExecuted it has been.
const FIELDS = ["visual", "speech", "sounds"];

app.registerExtension({
    name: "CCTech.LTXV23IDLoraPromptEditor",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "LTXV23IDLoraPromptEditor") return;

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            for (const field of FIELDS) {
                const values = message?.[field];
                if (!values || !values.length) continue;
                const widget = this.widgets?.find((w) => w.name === field);
                if (widget) widget.value = values[0];
            }

            app.graph.setDirtyCanvas(true, true);
        };
    },
});
