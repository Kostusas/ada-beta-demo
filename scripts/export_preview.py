"""Export nine published-resolution bait slices for an offline presentation."""
import base64
import io
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from figures import heatmaps, load_run, predict_slice


def main():
    run, p, saved = load_run(ROOT / "examples/published-slice")
    frames = []
    for bait in [1, 50, 100, 150, 200, 250, 300, 350, 400]:
        plane = saved if bait == run["bait"] else predict_slice(p, bait)
        fig = heatmaps(p, plane, bait, run["spacing_kb"])
        raster = io.BytesIO()
        fig.savefig(raster, format="png", dpi=110)
        plt.close(fig)
        png = io.BytesIO()
        Image.open(raster).convert("RGB").quantize(colors=192).save(png, format="PNG")
        frames.append({"bait": bait, "image": base64.b64encode(png.getvalue()).decode()})
    page = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Three-point chromosome contacts</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f5f5ef;color:#182b28;font:17px/1.55 system-ui,sans-serif}
main{max-width:1100px;margin:4vh auto;padding:0 24px}h1{font:clamp(2rem,5vw,3.5rem)/1.1 Georgia,serif;margin:0}
p{max-width:78ch}figure{background:white;margin:24px 0 12px;border:1px solid #d8ded6;border-radius:6px}
img{width:100%;vertical-align:middle}label{font-weight:650}input{accent-color:#007b72;width:min(480px,95%);vertical-align:middle}
footer{margin:1.8rem 0;font-size:.9rem}a{color:#09645c}
</style><main><h1>Three-point chromosome contacts</h1>
<p>Choose a bait on the circular chromosome. The left map shows how often pairs of regions
meet in condensin simulations. The right map predicts where two other regions meet the chosen bait.</p>
<figure><img id="maps" alt="Pair contact frequencies and the predicted three-point probability at the selected bait"></figure>
<label for="slider">Bait bin <output id="bait">200</output></label><br>
<input id="slider" type="range" min="0" max="8" value="4" step="1" aria-describedby="scale">
<p id="scale">404 bins at 10 kb spacing. The color scales saturate at 0.25 for pairs and 0.005 for triplets.
The dark lines through the bait mark bins excluded by the model.</p>
<footer>Harju, Messelink and Broedersz (2025) · <a href="https://doi.org/10.1103/PhysRevE.111.014403">Publication</a>
 · <a href="https://github.com/Kostusas/ada-beta-demo">Code and ADA benchmark</a></footer>
</main><script>
const frames = FRAMES;
const slider = document.querySelector('#slider');
function update() {
  const frame = frames[Number(slider.value)];
  document.querySelector('#bait').value = frame.bait;
  document.querySelector('#maps').src = 'data:image/png;base64,' + frame.image;
  slider.setAttribute('aria-valuetext', 'Bait bin ' + frame.bait);
}
slider.addEventListener('input', update);
update();
</script></html>"""
    (ROOT / "examples/explorer.html").write_text(page.replace("FRAMES", json.dumps(frames)))


if __name__ == "__main__":
    main()
