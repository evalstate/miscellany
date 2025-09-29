npx @marp-team/marp-cli@latest --themeSet ./theme/freud.css ./theme/structure.css ./theme/schema.css ./theme/evalstate-extensions.css -- --html ./presentation.md

node <<'NODE'
const fs = require('fs');
const path = require('path');
const target = path.join(__dirname, 'presentation.html');

if (!fs.existsSync(target)) {
  console.error('presentation.html was not generated.');
  process.exit(1);
}

const decodeHtml = (value) =>
  value
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'");

const pattern = /&lt;iframe class="web-embed"[\s\S]*?&lt;\/iframe&gt;/;

let html = fs.readFileSync(target, 'utf8');
const match = html.match(pattern);

if (!match) {
  console.warn('Iframe placeholder not found in presentation.html.');
  process.exit(0);
}

html = html.replace(pattern, decodeHtml(match[0]));
fs.writeFileSync(target, html);
NODE
