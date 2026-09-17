"""Fold the viewer into one HTML file that opens by double-clicking.

Two constraints shape this, and both come from `file://`.

**A page opened from disk cannot fetch its neighbours.** Every file is a
separate origin, so `fetch("./data/rig.json")` fails. The assets are therefore
embedded, gzipped and base64'd, and `js/data.js` looks in
``globalThis.__FLYLOOP_ASSETS`` before reaching for the network.

**A module script cannot import from `file://` either.** An *inline* module
with no imports is allowed, so everything is concatenated into one: three.js,
OrbitControls and the three application modules, with their `import` and
`export` statements removed. three.module.js makes that easy by ending in a
single `export { A, B, ... }` whose shorthand is already valid object-literal
syntax -- it becomes `const THREE = { A, B, ... }` with one substitution.

What is lost: the stimulus panel. There is no server behind a file, so the demo
replays recorded episodes and nothing more. `wireStimulus` returns early when
the markup is absent.
"""

from __future__ import annotations

import base64
import gzip
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: Episodes worth shipping: the measured coupling, which does nothing visible,
#: and the amplified one, which does. Shipping only the second would be a
#: demonstration of a result this project does not have.
DEMO_EPISODES = ("gain1", "gain50000h4")


def strip_module(js: str) -> str:
    """Remove the import and export statements from one of our own modules."""
    js = re.sub(r"^import\s+[^;]*?;\s*$", "", js, flags=re.MULTILINE | re.DOTALL)
    js = re.sub(r"^export\s+(?=(const|function|class|async))", "", js, flags=re.MULTILINE)
    js = re.sub(r"^export\s*\{[^}]*\};\s*$", "", js, flags=re.MULTILINE)
    return js


def three_as_object(js: str) -> str:
    """Turn three.js's single trailing `export { ... }` into a THREE namespace.

    Wrapped in a function rather than spilled into the top level: three exports
    a class called `Controls`, and so does the OrbitControls import below it.
    Flat concatenation makes that a redeclaration and the whole module fails to
    parse.
    """
    marker = "\nexport {"
    if marker not in js:
        raise SystemExit("three.module.js does not end in the expected export block")
    head, _, tail = js.rpartition(marker)
    return f"const THREE = (() => {{\n{head}\nreturn {{{tail}\n}})();"


def orbit_controls(js: str) -> str:
    """Bind OrbitControls' imports to the THREE object instead of a module."""
    # The specifier is whatever the vendored copy uses -- the bare name when it
    # came through an importmap, a relative path when it did not.
    match = re.search(r"^import\s*\{([^}]*)\}\s*from\s*['\"][^'\"]+['\"];", js, re.MULTILINE)
    if not match:
        raise SystemExit("OrbitControls.js does not open with a named import as expected")
    names = ", ".join(n.strip() for n in match.group(1).split(",") if n.strip())
    js = js[: match.start()] + f"const {{ {names} }} = THREE;" + js[match.end() :]
    js = re.sub(r"^export\s*\{[^}]*\};\s*$", "", js, flags=re.MULTILINE)
    return f"const OrbitControls = (() => {{\n{js}\nreturn OrbitControls;\n}})();"


def pack(path: Path) -> str:
    return base64.b64encode(gzip.compress(path.read_bytes(), 9)).decode()


def build(data: Path, out: Path) -> Path:
    assets: dict[str, str] = {}
    for name in ("rig.json", "meshes.bin", "atlas.json", "atlas.bin"):
        assets[name] = pack(data / name)

    index = {"format": "flyloop-index/1", "episodes": []}
    for name in DEMO_EPISODES:
        folder = data / "episodes" / name
        if not folder.is_dir():
            raise SystemExit(f"no such episode: {folder}")
        manifest = json.loads((folder / "episode.json").read_text())
        for part in ("episode.json", "episode.bin"):
            assets[f"episodes/{name}/{part}"] = pack(folder / part)
        index["episodes"].append(
            {
                "name": name,
                "path": f"episodes/{name}/",
                "gain": manifest.get("gain"),
                "coupling_hops": manifest.get("coupling_hops", 1),
                "coupling_share": manifest.get("coupling_share"),
                "frames": manifest.get("frames"),
                "body": manifest.get("body"),
            }
        )
    assets["episodes.json"] = base64.b64encode(
        gzip.compress(json.dumps(index).encode(), 9)
    ).decode()

    html = (HERE / "index.html").read_text()
    # Out: the importmap, the external module, and the stimulus panel.
    html = re.sub(r"<script type=\"importmap\">.*?</script>", "", html, flags=re.DOTALL)
    html = re.sub(r"<script type=\"module\" src=[^>]*></script>", "", html)
    html = re.sub(r"<section id=\"stim-panel\".*?</section>\n", "", html, flags=re.DOTALL)
    html = re.sub(r"<button id=\"stim-toggle\".*?</button>\n", "", html)
    html = html.replace(
        '<link rel="stylesheet" href="./style.css">',
        "<style>\n" + (HERE / "style.css").read_text() + "\n</style>",
    )
    html = html.replace(
        "<title>flyloop episode player</title>",
        "<title>flyloop episode player (single file)</title>",
    )

    bundle = "\n".join(
        [
            _ASSET_PRELUDE,
            "const __RAW = " + json.dumps(assets) + ";",
            _ASSET_DECODE,
            three_as_object((HERE / "vendor" / "three.module.js").read_text()),
            orbit_controls((HERE / "vendor" / "OrbitControls.js").read_text()),
            strip_module((HERE / "js" / "fk.js").read_text()),
            strip_module((HERE / "js" / "data.js").read_text()),
            strip_module((HERE / "js" / "boot.js").read_text()),
        ]
    )
    html = html.replace("</body>", f'<script type="module">\n{bundle}\n</script>\n</body>')
    out.write_text(html)
    return out


_ASSET_PRELUDE = """
// Everything below is one inline module. Nothing is fetched and nothing is
// imported, because a page opened from disk can do neither.
"""

_ASSET_DECODE = """
async function __inflate(b64) {
  const bin = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  if (typeof DecompressionStream === "undefined") {
    throw new Error("this browser has no DecompressionStream; serve the app instead");
  }
  const stream = new Blob([bin]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}
globalThis.__FLYLOOP_ASSETS = {};
for (const [name, b64] of Object.entries(__RAW)) {
  globalThis.__FLYLOOP_ASSETS[name] = await __inflate(b64);
}
"""


if __name__ == "__main__":
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "flyloop-demo.html"
    built = build(data, out)
    print(f"wrote {built}  ({built.stat().st_size / 1048576:.1f} MB)")
