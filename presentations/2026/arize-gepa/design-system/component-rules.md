# Component rules

Use components for:

- charts;
- data-driven visuals;
- custom SVG diagrams;
- animation or interaction;
- reusable media wrappers;
- small reusable presentational units.

Avoid components for:

- ordinary slide prose;
- one-off card grids expressible in `slides.md`;
- copy the presenter will edit frequently;
- slide-level wrappers that could be a class or layout.

Components should:

- fill parent-defined dimensions;
- use tokens from the global stylesheet or `design-system/tokens.css`;
- avoid hardcoded theme colors unless adding documented tokens;
- expose semantic props, not visual micro-control props;
- keep data-derived labels internally;
- leave rhetorical framing in `slides.md`;
- use one coordinate system for connector-heavy SVG diagrams.
