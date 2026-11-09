/** A load/dialog click must not become a pending start after the button re-enables. */
export function bindRunTrigger(button, state, start) {
  let armed = null;
  const arm = (event) => {
    armed = null;
    const current = state();
    if (!event.isTrusted || button.disabled || current.phase !== 'idle' || !current.graph) return;
    if (event.type === 'keydown' && (event.repeat || !['Enter', ' '].includes(event.key))) return;
    if (event.type === 'pointerdown' && event.button !== 0) return;
    armed = current.graph;
  };
  button.addEventListener('pointerdown', arm);
  button.addEventListener('keydown', arm);
  button.addEventListener('blur', () => {
    armed = null;
  });
  button.addEventListener('click', (event) => {
    const current = state(),
      requested = armed;
    armed = null;
    if (!event.isTrusted || button.disabled || current.phase !== 'idle' || !current.graph) return;
    const accessibilityClick = event.detail === 0 && button.matches(':focus');
    if (requested !== current.graph && !accessibilityClick) return;
    start();
  });
}
