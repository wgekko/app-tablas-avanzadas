import streamlit as st
import base64
from pathlib import Path
import streamlit.components.v1 as components

st.set_page_config(page_title="Analisis Mercado Laboral",
                   page_icon=":material/model_training:",
                   layout="wide",
                   initial_sidebar_state="collapsed")

# Ocultar sidebar
st.markdown("""
<style>
[data-testid="stSidebar"] {
    display: none;
}
[data-testid="stAppViewContainer"] {
    margin-left: 0px;
}
#MainMenu, header, footer, .stDeployButton {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# =========================================
# FUNCION IMAGEN BASE64
# =========================================
#IMG_PATH = "img/alex-knight.jpg"
IMG_PATH = "img/background.png"

def get_base64_of_image(path: str):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

background_base64 = get_base64_of_image(IMG_PATH)

# =========================================
#  ▼▼ PRIMER CÓDIGO (GLIDE REVEAL SECRET CODE)
# =========================================

pass_code = """
<style>
* { box-sizing: border-box; }
body {
  min-height: 100vh;
  display: grid;
  place-items: center;
  font-family: "SF Pro Text", "SF Pro Icons", "AOS Icons", "Helvetica Neue", Helvetica, Arial, sans-serif, system-ui;
  background: url('img/alex-knight-Ys-DBJeX0nE-unsplash.jpg') no-repeat center center fixed;
  background-size: cover;
  gap: 2rem;
}

/* Eliminamos el patrón de líneas para que no tape la imagen */
body::before {
  content: none;
}
.reveal-body {
  min-height: 100vh;
  display: grid;
  place-items: center;
  /* Fondo transparente para que se vea la imagen completa */
  background: transparent;
  font-family: Arial, sans-serif;
}

.reveal-section {
  display: grid;
  gap: 2rem;
  align-items: center;
  justify-content: center;
}

.reveal-section p {
  margin: 0;
  font-size: 1.5rem;
  background: linear-gradient(#eee, #777);
  color: transparent;
  background-clip: text;
}

.code {
  font-size: 1.8rem;
  display: flex;
  flex-wrap: nowrap;
  color: white;
  border-radius: 0.6rem;
  background: rgba(0,0,0,0.6); /* semi-transparente para mejor contraste */
  justify-content: center;
  box-shadow: 0 1px rgba(255,255,255,.25) inset;
  padding: 0.5rem 0.8rem;
}

.code:hover { cursor: grab; }

.digit {
  display: flex;
  height: 100%;
  padding: 1.8rem 0.4rem;
}

.digit span {
  scale: calc(var(--active, 0) + 0.5);
  filter: blur(calc((1 - var(--active, 0)) * 0.5rem));
  transition: scale 1s, filter 1s;
}

ul { padding:0; margin:0; }
.digit:first-of-type { padding-left: 1rem; }
.digit:last-of-type { padding-right: 1rem; }

:root {
  --lerp-0: 1;
  --lerp-1: calc(sin(50deg));
  --lerp-2: calc(sin(45deg));
  --lerp-3: calc(sin(35deg));
  --lerp-4: calc(sin(25deg));
  --lerp-5: calc(sin(15deg));
}

.digit:is(:hover, :focus-visible) { --active: var(--lerp-0); }
.digit:is(:hover, :focus-visible) + .digit,
.digit:has(+ .digit:is(:hover, :focus-visible)) { --active: var(--lerp-1); }

.digit:is(:hover, :focus-visible) + .digit + .digit,
.digit:has(+ .digit + .digit:is(:hover, :focus-visible)) { --active: var(--lerp-2); }

.digit:is(:hover, :focus-visible) + .digit + .digit + .digit,
.digit:has(+ .digit + .digit + .digit:is(:hover, :focus-visible)) { --active: var(--lerp-3); }

.digit:is(:hover, :focus-visible) + .digit + .digit + .digit + .digit,
.digit:has(+ .digit + .digit + .digit + .digit:is(:hover, :focus-visible)) { --active: var(--lerp-4); }

.digit:is(:hover, :focus-visible) + .digit + .digit + .digit + .digit + .digit,
.digit:has(+ .digit + .digit + .digit + .digit + .digit:is(:hover, :focus-visible)) { --active: var(--lerp-5); }
</style>

<div class="reveal-body">
<section class="reveal-section">
  <p>## ACCESS GRANTED END ##</p>
  <ul class="code">
    <li tabindex="0" class="digit"><span>O</span></li>
    <li tabindex="0" class="digit"><span>U</span></li>
    <li tabindex="0" class="digit"><span>T</span></li>
    <li tabindex="0" class="digit"><span>-</span></li>
    <li tabindex="0" class="digit"><span>S</span></li>
    <li tabindex="0" class="digit"><span>Y</span></li>
    <li tabindex="0" class="digit"><span>S</span></li>
    <li tabindex="0" class="digit"><span>T</span></li>
    <li tabindex="0" class="digit"><span>E</span></li>
    <li tabindex="0" class="digit"><span>M</span></li>
  </ul>
</section>
</div>
"""
components.html(pass_code, height=220, scrolling=False)

# =========================================
#  ▼▼ SEGUNDO CÓDIGO (TU ANIMACIÓN SCRAMBLE)
# =========================================

st.write(" ")  # separador visual

html_code = f"""
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8" />
<style>
body {{    
    background-image: 
        linear-gradient(to bottom, rgba(0,0,0,0.5) 100%, rgba(0,0,0,0.5) 100%),
        linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)), /* capa semitransparente */
        url("data:image/jpg;base64,{background_base64}") !important;
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    color: #0f0;
    font-family: 'Courier New', monospace;
}}
</style>
</head>

<body>
<div id="text" style="font-size:38px;text-align:center;margin-top:160px;"></div>

<script>
class TextScramble {{
    constructor(el) {{
        this.el = el;
        this.chars = "!<>-_\\/[]—=+*^?#";
        this.update = this.update.bind(this);
    }}

    setText(newText) {{
        const oldText = this.el.innerText;
        const length = Math.max(oldText.length, newText.length);
        const promise = new Promise(resolve => this.resolve = resolve);
        this.queue = [];

    for (let i=0;i<length;i++) {{
        const from = oldText[i] || "";
        const to = newText[i] || "";
        const start = Math.floor(Math.random()*40);
        const end   = start + Math.floor(Math.random()*40);
        this.queue.push({{from,to,start,end}});
    }}

    cancelAnimationFrame(this.frameRequest);
    this.frame = 0;
    this.update();
    return promise;
  }}

  update() {{
    let output="";
    let complete=0;

    for (let i=0;i<this.queue.length;i++) {{
      let {{from,to,start,end,char}} = this.queue[i];

      if (this.frame >= end) {{
        complete++;
        output += to;
      }} else if (this.frame >= start) {{
        if (!char || Math.random()<0.28) {{
          char = this.randomChar();
          this.queue[i].char = char;
        }}
        output += `<span style="color:#0f0;">${{char}}</span>`;
      }} else {{
        output += from;
      }}
    }}

    this.el.innerHTML = output;

    if (complete === this.queue.length) {{
      this.resolve();
    }} else {{
      this.frameRequest = requestAnimationFrame(this.update);
      this.frame++;
    }}
  }}

  randomChar() {{
    return this.chars[Math.floor(Math.random()*this.chars.length)];
  }}
}}

const fx = new TextScramble(document.getElementById("text"));
fx.setText("Salio del sistema");
</script>

</body>
</html>
"""
components.html(html_code, height=400, scrolling=False)
