import os
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          os.environ.get("MODEL_FILE", "logistic_1.pkl"))
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURES = list(model.feature_names_in_)
CLASSES = [str(c) for c in model.classes_]

# Small emoji per input, purely cosmetic
ICONS = {
    "Major_Category": "🎓", "Year_of_Study": "📅", "Pre_Semester_GPA": "📊",
    "Weekly_GenAI_Hours": "🤖", "Primary_Use_Case": "🧩",
    "Prompt_Engineering_Skill": "✍️", "Paid_Subscription": "💳",
    "Traditional_Study_Hours": "📚", "Perceived_AI_Dependency": "🔗",
    "Institutional_Policy": "🏛️", "Skill_Retention_Score": "🧠",
}

HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Logistic Regression Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --c1: #6a11cb; --c2: #2575fc; --c3: #ff6a95;
    --glass: rgba(255,255,255,.14); --border: rgba(255,255,255,.28);
    --text: #fff; --muted: rgba(255,255,255,.75);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Poppins', system-ui, sans-serif; color: var(--text);
    min-height: 100vh; padding: 2.5rem 1rem 3rem;
    background: linear-gradient(-45deg, #6a11cb, #2575fc, #11998e, #ff6a95);
    background-size: 400% 400%; animation: bgShift 16s ease infinite;
    overflow-x: hidden;
  }
  @keyframes bgShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }

  /* floating blobs */
  .blob { position: fixed; border-radius: 50%; filter: blur(60px); opacity: .35; z-index: 0; animation: float 14s ease-in-out infinite; }
  .b1 { width: 280px; height: 280px; background: #ff6a95; top: -60px; left: -60px; }
  .b2 { width: 340px; height: 340px; background: #00e5ff; bottom: -100px; right: -80px; animation-delay: -5s; }
  .b3 { width: 200px; height: 200px; background: #ffd54f; top: 45%; left: 70%; animation-delay: -9s; }
  @keyframes float { 0%,100%{transform:translateY(0) scale(1)} 50%{transform:translateY(-40px) scale(1.12)} }

  .wrap { position: relative; z-index: 1; max-width: 880px; margin: 0 auto; }
  header { text-align: center; margin-bottom: 2rem; animation: dropIn .9s cubic-bezier(.2,.9,.3,1.2) both; }
  header h1 { font-size: clamp(1.8rem, 5vw, 2.8rem); font-weight: 700; letter-spacing: -.5px; }
  header h1 span { display: inline-block; animation: wave 2.6s ease-in-out infinite; }
  header p { color: var(--muted); margin-top: .5rem; }
  @keyframes dropIn { from { opacity: 0; transform: translateY(-30px); } to { opacity: 1; transform: none; } }
  @keyframes wave { 0%,100%{transform:rotate(-8deg)} 50%{transform:rotate(10deg)} }

  .card {
    background: var(--glass); border: 1px solid var(--border);
    backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
    border-radius: 24px; padding: 2rem; box-shadow: 0 20px 50px rgba(0,0,0,.25);
    animation: rise .9s .15s ease both;
  }
  @keyframes rise { from { opacity: 0; transform: translateY(40px); } to { opacity: 1; transform: none; } }

  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1.1rem; }
  .field { opacity: 0; animation: pop .6s ease forwards; }
  @keyframes pop { from { opacity: 0; transform: translateY(18px) scale(.96); } to { opacity: 1; transform: none; } }
  .field label { display: block; font-size: .8rem; font-weight: 500; margin-bottom: .35rem; color: var(--muted); }
  .field input {
    width: 100%; padding: .8rem 1rem; border-radius: 14px; font: inherit; color: #fff;
    background: rgba(255,255,255,.12); border: 1.5px solid transparent; outline: none;
    transition: border-color .25s, background .25s, transform .25s, box-shadow .25s;
  }
  .field input::placeholder { color: rgba(255,255,255,.5); }
  .field input:hover { background: rgba(255,255,255,.18); }
  .field input:focus { border-color: #fff; background: rgba(255,255,255,.22); transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0,0,0,.2); }

  .actions { display: flex; gap: .8rem; justify-content: center; margin-top: 1.8rem; flex-wrap: wrap; }
  button {
    font: inherit; font-weight: 600; border: none; cursor: pointer; border-radius: 999px;
    padding: .9rem 2.4rem; transition: transform .2s, box-shadow .2s;
  }
  #go { color: #fff; background: linear-gradient(90deg, #ff6a95, #ff9a44); box-shadow: 0 10px 25px rgba(255,106,149,.45); position: relative; overflow: hidden; }
  #go::after { content: ""; position: absolute; inset: 0; background: linear-gradient(120deg, transparent 30%, rgba(255,255,255,.45) 50%, transparent 70%); transform: translateX(-100%); animation: shine 3s infinite; }
  @keyframes shine { to { transform: translateX(100%); } }
  #go:hover { transform: translateY(-3px) scale(1.04); }
  #go:active { transform: scale(.97); }
  #reset { background: rgba(255,255,255,.18); color: #fff; border: 1px solid var(--border); }
  #reset:hover { background: rgba(255,255,255,.3); transform: translateY(-2px); }
  button:disabled { opacity: .7; cursor: wait; }

  .spinner { display: none; width: 20px; height: 20px; margin-left: .6rem; vertical-align: middle;
    border: 3px solid rgba(255,255,255,.4); border-top-color: #fff; border-radius: 50%; animation: spin .7s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .loading .spinner { display: inline-block; }

  #result { display: none; margin-top: 2rem; text-align: center; }
  #result.show { display: block; animation: rise .7s ease both; }
  .badge {
    display: inline-block; padding: .6rem 1.8rem; border-radius: 999px; font-size: 1.6rem; font-weight: 700;
    background: #fff; color: #333; animation: bounce .8s ease both, glow 2s ease-in-out infinite;
  }
  .badge.High { color: #11998e; } .badge.Low { color: #e53935; } .badge.Medium { color: #f59f00; }
  @keyframes bounce { 0%{transform:scale(.3);opacity:0} 55%{transform:scale(1.18)} 100%{transform:scale(1);opacity:1} }
  @keyframes glow { 0%,100%{box-shadow:0 0 0 rgba(255,255,255,0)} 50%{box-shadow:0 0 30px rgba(255,255,255,.7)} }
  .label { color: var(--muted); font-size: .9rem; margin-bottom: .7rem; }

  .bars { margin: 1.6rem auto 0; max-width: 520px; text-align: left; }
  .bar-row { margin-bottom: .9rem; }
  .bar-top { display: flex; justify-content: space-between; font-size: .85rem; margin-bottom: .3rem; }
  .track { height: 12px; border-radius: 99px; background: rgba(255,255,255,.2); overflow: hidden; }
  .fill { height: 100%; width: 0; border-radius: 99px; transition: width 1.2s cubic-bezier(.2,.8,.2,1); background: linear-gradient(90deg, #fff, #a0f0ff); }
  .fill.High { background: linear-gradient(90deg, #11998e, #38ef7d); }
  .fill.Low { background: linear-gradient(90deg, #e53935, #ff8a80); }
  .fill.Medium { background: linear-gradient(90deg, #f59f00, #ffe066); }

  .error { margin-top: 1.5rem; text-align: center; background: rgba(229,57,53,.55); padding: .9rem; border-radius: 14px; display: none; animation: shake .5s; }
  @keyframes shake { 0%,100%{transform:translateX(0)} 25%{transform:translateX(-8px)} 75%{transform:translateX(8px)} }
  footer { text-align: center; margin-top: 1.5rem; font-size: .8rem; color: var(--muted); }
  @media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; } }
</style>
</head>
<body>
  <div class="blob b1"></div><div class="blob b2"></div><div class="blob b3"></div>
  <div class="wrap">
    <header>
      <h1><span>🔮</span> Logistic Regression Predictor</h1>
      <p>Enter the values below and get an instant High / Medium / Low prediction.</p>
    </header>

    <div class="card">
      <form id="form" autocomplete="off">
        <div class="grid">
          {% for f in features %}
          <div class="field" style="animation-delay: {{ 0.25 + loop.index0 * 0.07 }}s">
            <label for="{{ f }}">{{ icons.get(f, '•') }} {{ f.replace('_', ' ') }}</label>
            <input type="number" step="any" id="{{ f }}" name="{{ f }}" placeholder="Enter value" required>
          </div>
          {% endfor %}
        </div>
        <div class="actions">
          <button type="submit" id="go">Predict<span class="spinner"></span></button>
          <button type="button" id="reset">Reset</button>
        </div>
      </form>

      <div class="error" id="error"></div>

      <div id="result">
        <div class="label">Predicted class</div>
        <div class="badge" id="badge"></div>
        <div class="bars" id="bars"></div>
      </div>
    </div>
    <footer>Built with Flask &amp; scikit-learn · Deployed on Render</footer>
  </div>

<script>
  const form = document.getElementById('form');
  const btn = document.getElementById('go');
  const result = document.getElementById('result');
  const badge = document.getElementById('badge');
  const bars = document.getElementById('bars');
  const errBox = document.getElementById('error');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    errBox.style.display = 'none';
    result.classList.remove('show');
    const data = {};
    new FormData(form).forEach((v, k) => data[k] = parseFloat(v));

    btn.disabled = true; btn.classList.add('loading');
    try {
      const res = await fetch('/predict', {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)
      });
      const out = await res.json();
      if (!res.ok) throw new Error(out.error || 'Something went wrong');
      show(out);
    } catch (err) {
      errBox.textContent = err.message;
      errBox.style.display = 'block';
      errBox.style.animation = 'none'; void errBox.offsetWidth; errBox.style.animation = 'shake .5s';
    } finally {
      btn.disabled = false; btn.classList.remove('loading');
    }
  });

  function show(out) {
    badge.textContent = out.prediction;
    badge.className = 'badge ' + out.prediction;
    bars.innerHTML = '';
    Object.entries(out.probabilities).sort((a, b) => b[1] - a[1]).forEach(([cls, p]) => {
      const row = document.createElement('div');
      row.className = 'bar-row';
      row.innerHTML = `<div class="bar-top"><span>${cls}</span><span class="pct">0%</span></div>
                       <div class="track"><div class="fill ${cls}"></div></div>`;
      bars.appendChild(row);
      const fill = row.querySelector('.fill'), pct = row.querySelector('.pct');
      requestAnimationFrame(() => requestAnimationFrame(() => fill.style.width = (p * 100) + '%'));
      // count-up animation
      const t0 = performance.now(), dur = 1200;
      (function tick(t) {
        const k = Math.min((t - t0) / dur, 1);
        pct.textContent = (p * 100 * k).toFixed(1) + '%';
        if (k < 1) requestAnimationFrame(tick);
      })(t0);
    });
    result.classList.add('show');
    result.scrollIntoView({behavior: 'smooth', block: 'center'});
  }

  document.getElementById('reset').addEventListener('click', () => {
    form.reset(); result.classList.remove('show'); errBox.style.display = 'none';
  });
</script>
</body>
</html>
"""


def run_prediction(data):
    row = {f: float(data[f]) for f in FEATURES}
    df = pd.DataFrame([row], columns=FEATURES)
    pred = str(model.predict(df)[0])
    probs = model.predict_proba(df)[0]
    return pred, {c: float(p) for c, p in zip(CLASSES, probs)}


@app.route("/")
def home():
    return render_template_string(HTML, features=FEATURES, icons=ICONS)


@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json(silent=True) or {}
    missing = [f for f in FEATURES if f not in data or data[f] is None]
    if missing:
        return jsonify({"error": "Missing values: " + ", ".join(missing)}), 400
    try:
        pred, probs = run_prediction(data)
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid value: {e}"}), 400
    return jsonify({"prediction": pred, "probabilities": probs})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
