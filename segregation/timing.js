function render({ model, el }) {
  let started;
  let frame;
  const root = () => el.closest(".ada-segregation-view");
  // Plotly can lay out a figure before its notebook tab is visible. Use actual column widths.
  const slots = () => Array.from(root()?.querySelectorAll(".adaPlotSlot") ?? []);
  const observer = new ResizeObserver(() => {
    const widths = slots().map(slot => slot.clientWidth);
    const previous = model.get("widths");
    if (widths.length === 4 && widths.every(w => w > 0) && widths.some((w, i) => w !== previous[i])) {
      model.set("widths", widths);
      model.save_changes();
    }
  });
  function observePlots() {
    for (const slot of slots()) observer.observe(slot);
  }
  const attachFrame = requestAnimationFrame(observePlots);

  function begin(event) {
    const control = event.target.closest?.(".ada-query-trigger");
    if (!control || !root()?.contains(control)) return;
    started = performance.now();
    el.textContent = "";
  }

  function result() {
    cancelAnimationFrame(frame);
    const revision = model.get("revision");
    if (revision) observePlots();
    if (!revision) {
      el.textContent = "";
      started = undefined;
      return;
    }
    if (started === undefined) return;
    const start = started;
    const deadline = performance.now() + 15000;
    function painted() {
      if (started !== start || model.get("revision") !== revision) return;
      const plots = root()?.querySelectorAll(".js-plotly-plot") ?? [];
      // Plotly copies layout.meta into the rendered layout for each completed update.
      const matching = Array.from(plots).filter(plot => plot._fullLayout?.meta === revision);
      if (matching.length >= 2) {
        frame = requestAnimationFrame(() => {
          el.textContent = `Browser interaction to updated plots · ${((performance.now() - start) / 1000).toFixed(2)} s`;
          started = undefined;
        });
      } else if (performance.now() < deadline) {
        frame = requestAnimationFrame(painted);
      }
    }
    frame = requestAnimationFrame(painted);
  }

  const events = ["pointerup", "click", "keydown"];
  for (const event of events) document.addEventListener(event, begin, true);
  model.on("change:revision", result);
  return () => {
    observer.disconnect();
    cancelAnimationFrame(attachFrame);
    cancelAnimationFrame(frame);
    for (const event of events) document.removeEventListener(event, begin, true);
    model.off("change:revision", result);
  };
}

export default { render };
