const katex = require('katex'); const fs = require('fs');
const items = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const out = items.map(([tex, display]) => katex.renderToString(tex, {displayMode: display, throwOnError: true, strict: 'ignore', trust: false}));
fs.writeFileSync(process.argv[3], JSON.stringify(out));
