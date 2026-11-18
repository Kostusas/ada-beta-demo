function render({ model, el }) {
  let started;
  let frame;
  const root = () => el.closest(".ada-segregation-view");

  function begin(event) {
    const control = event.target.closest?.(".ada-query-trigger");
    if (!control || !root()?.contains(control)) return;
    started = performance.now();
    el.textContent = "";
  }

  function result() {
    cancelAnimationFrame(frame);
    const revision = model.get("revision");
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
    cancelAnimationFrame(frame);
    for (const event of events) document.removeEventListener(event, begin, true);
    model.off("change:revision", result);
  };
}

export default { render };
