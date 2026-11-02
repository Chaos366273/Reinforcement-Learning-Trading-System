# PathRise UI design samples

Static, dependency-free design samples for the PPO trading platform. They use dummy data only and do not touch the training code.

Open `index.html` in a browser (no build step), or serve the folder:

```bash
cd ui-concepts && python3 -m http.server 8080
```

| File | What it shows |
| --- | --- |
| `index.html` | Gallery with live previews of each concept |
| `concept-a-desk.html` | Desk: refined version of the current shell, with loading, empty and error states |
| `concept-b-cockpit.html` | Cockpit: top-nav layout, decision ring, gating pipeline, clickable 30-stock universe |
| `concept-c-explainer.html` | Decision Explainer: plain-English reasons, SHAP waterfall, safety checks, stop levels |
| `shared.css` | Tokens (dark default, `data-theme="light"`), components, responsive rules |
| `shared.js` | Rail, Clawbot dock (resizable, `/` tools, `@` agents), charts, dummy data |

URL flags useful for review: `?theme=light`, `?mode=pro`, `?state=loading|empty|error`, `?dock=open`, `?still` (disables motion for screenshots).

Fonts (Inter, JetBrains Mono) load from Google Fonts and fall back to system fonts offline.
