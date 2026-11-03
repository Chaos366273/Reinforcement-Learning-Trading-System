# PathRise UI design samples

Static, dependency-free PathRise site. `index.html` is the product. `gallery.html` keeps the first three visual samples. Dummy and recorded data only. Training code is unchanged.

Open `index.html` in a browser (no build step), or serve the folder:

```bash
cd ui-concepts && python3 -m http.server 8080
```

| File | What it shows |
| --- | --- |
| `index.html` | Product landing: Don't trade. Train. |
| `login.html` `signup.html` `onboarding.html` | Auth and first-run setup |
| `desk.html` | Signed-in Cockpit desk |
| `training.html` `arena.html` `report.html` | Bot builder, live training replay, report card |
| `fixtures/quick-train.jsonl` | Recorded training events that drive the Arena |
| `gallery.html` | The first three visual samples |
| `concept-a-desk.html` | Desk: refined version of the current shell, with loading, empty and error states |
| `concept-b-cockpit.html` | Cockpit: top-nav layout, decision ring, gating pipeline, clickable 30-stock universe |
| `concept-c-explainer.html` | Decision Explainer: plain-English reasons, SHAP waterfall, safety checks, stop levels |
| `shared.css` | Tokens (dark default, `data-theme="light"`), components, responsive rules |
| `shared.js` | Rail, Clawbot dock (resizable, `/` tools, `@` agents), charts, dummy data |

URL flags useful for review: `?theme=light`, `?mode=pro`, `?state=loading|empty|error`, `?dock=open`, `?still` (disables motion for screenshots).

Fonts (Inter, JetBrains Mono) load from Google Fonts and fall back to system fonts offline.
