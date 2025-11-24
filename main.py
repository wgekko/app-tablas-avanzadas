import streamlit as st
import base64
from pathlib import Path
from streamlit.components.v1 import html
import streamlit.components.v1 as components
import unicodedata

st.set_page_config(page_title="Analisis Mercado Laboral", page_icon=":material/model_training:",layout="wide", initial_sidebar_state="collapsed")

hide_sidebar_style = """
    <style>
        /* Oculta la barra lateral completa */
        [data-testid="stSidebar"] {
            display: none;
        }
        /* Ajusta el área principal para usar todo el ancho */
        [data-testid="stAppViewContainer"] {
            margin-left: 0px;
        }
    </style>
"""
st.markdown(hide_sidebar_style, unsafe_allow_html=True)


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
            .stDeployButton {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)




pass_code = """
<style>
* { box-sizing: border-box; }

body {
  min-height: 100vh;
  display: grid;
  place-items: center;
  font-family: "SF Pro Text", "SF Pro Icons", "AOS Icons", "Helvetica Neue", Helvetica, Arial, sans-serif, system-ui;
  background: hsl(0 0% 0%);
  gap: 2rem;
}

body::before {
	--line: hsl(0 0% 95% / 0.25);
	content: "";
	height: 100vh;
	width: 100vw;
	position: fixed;
	background:
		linear-gradient(90deg, var(--line) 1px, transparent 1px 10vmin) 0 -5vmin / 10vmin 10vmin,
		linear-gradient(var(--line) 1px, transparent 1px 10vmin) 0 -5vmin / 10vmin 10vmin;
	mask: linear-gradient(-5deg, transparent 10%, white);
	top: 0;
	z-index: -1;
}

.reveal-body {
  min-height: 100vh;
  display: grid;
  place-items: center;
  background: black;
  font-family: Arial, sans-serif;
}

.reveal-section {
  display: grid;
  gap: 2rem; /* MÁS PEQUEÑO */
  align-items: center;
  justify-content: center;
}

.reveal-section p {
  margin: 0;
  font-size: 1.5rem; /* TÍTULO MÁS PEQUEÑO */
  background: linear-gradient(#eee, #777);
  color: transparent;
  background-clip: text;
}

.code {
  font-size: 1.8rem; /* TAMAÑO DE LAS LETRAS REDUCIDO */
  display: flex;
  flex-wrap: nowrap;
  color: white;
  border-radius: 0.6rem; /* MÁS PEQUEÑO */
  background: #0d0d0d;
  justify-content: center;
  box-shadow: 0 1px rgba(255,255,255,.25) inset;
  padding: 0.5rem 0.8rem; /* CAJA MÁS PEQUEÑA */
}

.code:hover { cursor: grab; }

.digit {
  display: flex;
  height: 100%;
  padding: 1.8rem 0.4rem; /* CAJA INTERNA MÁS PEQUEÑA */
}

.digit span {
  scale: calc(var(--active, 0) + 0.5);
  filter: blur(calc((1 - var(--active, 0)) * 0.5rem)); /* menos blur */
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
  <p>#- Acceder a la clave -#</p>
  <ul class="code">
    <li tabindex="0" class="digit"><span>E</span></li>
    <li tabindex="0" class="digit"><span>M</span></li>
    <li tabindex="0" class="digit"><span>P</span></li>
    <li tabindex="0" class="digit"><span>L</span></li>
    <li tabindex="0" class="digit"><span>E</span></li>
    <li tabindex="0" class="digit"><span>O</span></li>
  </ul>
</section>
</div>
"""
components.html(pass_code, height=220, scrolling=False)

# -----------------------------------
# FUNCION PARA CONVERTIR IMAGEN A BASE64
# -----------------------------------
# Ruta de la imagen de fondo
IMG_PATH = "img/alex-knight.jpg"

def get_base64_of_image(path: str):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Convertimos la imagen a base64
#background_base64 = get_base64_of_image(IMG_PATH)
background_base64 = get_base64_of_image("img/ciudad-nocturna.jpg")
#background_base64 = get_base64_of_image("img/alex-knight-Ys-DBJeX0nE-unsplash.jpg")

st.audio("audio/Harry_Gregson-Williams_-_Muir_Races_to_Work_Saundtrek_k_filmu_SHpionskie_igry_Spy_Game_(mp3.pm).mp3", format="audio/mp3",loop=True, autoplay=True)

html_code = f"""
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8" />
<style>
body {{
  
  background-image: url("data:image/jpg;base64,{background_base64}") !important;
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
  
  //background-color: #111;
  color: #0f0;
  font-family: 'Courier New', monospace;
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  margin: 0;
}}

.text-container {{
  font-size: 2.5rem;
  font-weight: bold;
  min-height: 150px;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  text-shadow: 0 0 8px currentColor;
}}

.scrambling {{
  text-shadow: 0 0 12px currentColor;
}}

.controls {{
  position: absolute;
  top: 20px;
  left: 20px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  background-color: rgba(0,0,0,0.7);
  padding: 15px;
  border: 1px solid #0f0;
  max-width: 300px;
  box-shadow: 0 0 10px rgba(0, 255, 0, 0.3);
}}

.control-group {{
  display: flex;
  flex-direction: column;
  margin-bottom: 10px;
}}

.control-group label {{
  margin-bottom: 5px;
  font-size: 0.8rem;
}}

.btn, select, input[type="text"] {{
  padding: 8px 12px;
  background-color: #333;
  color: #0f0;
  border: 1px solid #0f0;
  cursor: pointer;
  font-family: 'Courier New', monospace;
}}

.btn:hover {{
  box-shadow: 0 0 10px rgba(0, 255, 0, 0.5);
}}

.custom-text {{
  width: 100%;
  resize: none;
  height: 60px;
  background-color: #333;
  color: #0f0;
  border: 1px solid #0f0;
  font-family: 'Courier New', monospace;
  margin-bottom: 10px;
}}

.color-picker {{
  display: flex;
  gap: 5px;
}}

.color-input {{
  width: 50%;
}}
</style>
</head>

<body>
<audio id="hoverSound" preload="auto">
  <source src="audio/Harry_Gregson-Williams_-_Muir_Races_to_Work_Saundtrek_k_filmu_SHpionskie_igry_Spy_Game_(mp3.pm).mp3" type="audio/mpeg">
</audio>

<div class="controls">
    <div class="control-group">
      <label for="charSet">Character Set:</label>
      <select id="charSet" onchange="updateSettings()">
        <option value="tech1">Tech/Code Symbols 1</option>
        <option value="tech2">Tech/Code Symbols 2</option>
        <option value="math">Math & Binary</option>
        <option value="cryptic">Cryptic/Alien</option>
        <option value="mixed">Mixed Languages</option>
        <option value="alphabet">Alphabet</option>
        <option value="matrix1">Matrix 1</option>
        <option value="matrix2">Matrix 2</option>
        <option value="matrix3">Chinese Characters</option>
        <option value="matrix4">Japanese Kanji</option>
      </select>
    </div>
    
    <div class="control-group">
      <label for="customText">Custom Text:</label>
      <textarea id="customText" class="custom-text" placeholder="Enter custom text..."></textarea>
    </div>
    
    <div class="control-group">
      <label for="revealSpeed">Reveal Speed:</label>
      <input type="range" id="revealSpeed" min="1" max="10" value="10" onchange="updateSettings()">
    </div>
    
    <div class="control-group">
      <label for="changeFreq">Change Frequency:</label>
      <input type="range" id="changeFreq" min="1" max="100" value="28" onchange="updateSettings()">
    </div>
    
    <div class="control-group">
      <label for="glowIntensity">Glow Intensity</label>
      <input type="range" id="glowIntensity" min="0" max="20" value="8" onchange="updateGlow()">
    </div>
    
    <div class="control-group">
      <label>Colors:</label>
      <div class="color-picker">
        <input type="color" id="bgColor" value="#111111" class="color-input" onchange="updateColors()">
        <input type="color" id="textColor" value="#00ff00" class="color-input" onchange="updateColors()">
      </div>
    </div>
    
    <button class="btn" onclick="playAnimation()">Reveal Text</button>
</div>

<div class="text-container" id="text"></div>

<script>


class TextScramble {{
  constructor(el) {{
    this.el = el;
    this.charSets = {{
      tech1: '!<>-\\\/[]—=+*^?#',
      tech2: '<>-\\\/[]—=+*^?#$%&()~',
      math: '01︎10︎101︎01︎+=-×÷',
      cryptic: '¥¤§Ω∑∆√∞≈≠≤≥',
      mixed: 'あ㐀明る日¥£€$¢₽₹₿',
      alphabet: 'abcdefghijklmnopqrstuvwxyz',
      matrix1: 'ラドクリフマラソンわたしワタシんょンョたばこタバコとうきょうトウキョウ',
      matrix2: '日ﾊﾐﾋｰｳｼﾅﾓﾆｻﾜﾂｵﾘｱﾎﾃﾏｹﾒｴｶｷﾑﾕﾗｾﾈｽﾀﾇﾍ',
      matrix3: '字型大小女巧偉周年',
      matrix4: '九七二人入八力十下三千上口土夕大女子小山川五天中六円手文日月木水火犬王正出本右四左玉生田白目石立百年休先名字早気竹糸耳虫村男町花見貝赤足車学林空金雨青草音',

    }};
    this.chars = this.charSets.tech1;
    this.update = this.update.bind(this);
    this.revealSpeed = 1;
    this.changeFrequency = 0.28;
    this.highlightColor = '#00ff88';
    this.glowIntensity = 8;
    this.activeGlowIntensity = 12;
  }}

  setCharSet(setName) {{
    if (this.charSets[setName]) {{
      this.chars = this.charSets[setName];
      return true;
    }}
    return false;
  }}

  setRevealSpeed(speed) {{
    this.revealSpeed = 11 - speed;
  }}

  setChangeFrequency(freq) {{
    this.changeFrequency = freq / 100;
  }}

  setHighlightColor(color) {{
    this.highlightColor = color;
  }}

  setGlowIntensity(intensity) {{
    this.glowIntensity = intensity;
    this.activeGlowIntensity = intensity + 4;
    document.getElementById('text').style.textShadow = `0 0 ${{intensity}}px currentColor`;
  }}

  setText(newText) {{
    const oldText = this.el.innerText;
    const length = Math.max(oldText.length, newText.length);
    const promise = new Promise(resolve => this.resolve = resolve);
    this.queue = [];

    for (let i = 0; i < length; i++) {{
      const from = oldText[i] || '';
      const to = newText[i] || '';
      const start = Math.floor(Math.random() * (40 / this.revealSpeed));
      const end = start + Math.floor(Math.random() * (40 / this.revealSpeed));
      this.queue.push({{ from, to, start, end }});
    }}

    cancelAnimationFrame(this.frameRequest);
    this.frame = 0;
    this.update();
    return promise;
  }}

  update() {{
    let output = '';
    let complete = 0;

    for (let i = 0, n = this.queue.length; i < n; i++) {{
      let {{ from, to, start, end, char }} = this.queue[i];

      if (this.frame >= end) {{
        complete++;
        output += to;
      }} else if (this.frame >= start) {{
        if (!char || Math.random() < this.changeFrequency) {{
          char = this.randomChar();
          this.queue[i].char = char;
        }}
        output += `<span class="scrambling" style="color: ${{this.highlightColor}}; text-shadow: 0 0 ${{this.activeGlowIntensity}}px currentColor;">${{char}}</span>`;
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
    return this.chars[Math.floor(Math.random() * this.chars.length)];
  }}
}}

const phrases = [
  'Encrypte la Clave',
  'Secret Message Revealed',
  'Access Granted',
  'System Online',
  'Loading Complete'
];

let counter = 0;
const el = document.getElementById('text');
const fx = new TextScramble(el);

function updateSettings() {{
  const charSet = document.getElementById('charSet').value;
  const revealSpeed = parseInt(document.getElementById('revealSpeed').value);
  const changeFreq = parseInt(document.getElementById('changeFreq').value);

  fx.setCharSet(charSet);
  fx.setRevealSpeed(revealSpeed);
  fx.setChangeFrequency(changeFreq);
}}

function updateColors() {{
  const bgColor = document.getElementById('bgColor').value;
  const textColor = document.getElementById('textColor').value;

  document.body.style.backgroundColor = bgColor;
  document.body.style.color = textColor;
  fx.setHighlightColor(shiftColor(textColor, 40));
}}

function updateGlow() {{
  const glowIntensity = parseInt(document.getElementById('glowIntensity').value);
  fx.setGlowIntensity(glowIntensity);
}}

function shiftColor(hex, lightnessDelta) {{
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);

  const shift = (c) => {{
    const newVal = Math.min(255, c + lightnessDelta);
    return newVal.toString(16).padStart(2, '0');
  }};

  return `#${{shift(r)}}${{shift(g)}}${{shift(b)}}`;
}}

///////////////////////////////////////////////////////
// ★★ MODIFICACIÓN PEDIDA ★★
// Convierte el texto ingresado totalmente al charset
///////////////////////////////////////////////////////
function playAnimation() {{
  const customText = document.getElementById('customText').value.trim();
  const charSet = fx.chars;

  let text;

  if (customText) {{
    text = customText
        .split("")
        .map((_, i) => charSet[i % charSet.length])
        .join("");
  }} else {{
    text = phrases[counter];
  }}

  fx.setText(text).then(() => {{
    setTimeout(() => {{
      if (!customText) {{
        counter = (counter + 1) % phrases.length;
      }}
    }}, 2000);
  }});
}}

updateColors();
updateSettings();
updateGlow();
setTimeout(playAnimation, 1000);
</script>

</body>
</html>
"""

st.components.v1.html(html_code, height=600, scrolling=False)


# --- Inyectar CSS desde archivo ---
with open("assets/style.css","r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


with st.container():

    # --- Estado de sesión ---
    if "ingreso" not in st.session_state:
        st.session_state.ingreso = False

    if "clave" not in st.session_state:
        st.session_state.clave = ""

    # --- Función para verificar la clave ---
    def verificar_clave():
        clave_ingresada = unicodedata.normalize("NFC", st.session_state.clave)
        claves_validas = [
            "!<>-\/", "<>-\/[", "01︎10︎", "¥¤§Ω∑∆", "あ㐀明る日¥",
            "abcdef", "ラドクリフマ", "日ﾊﾐﾋｰｳ", "字型大小女巧", "九七二人入八",
        ]
        claves_validas = [unicodedata.normalize("NFC", c) for c in claves_validas]

        if clave_ingresada in claves_validas:
            st.session_state.ingreso = True
        else:
            st.error(":material/error: Clave incorrecta")

    # --- Contenedor principal ---
    with st.container():

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            if not st.session_state.ingreso:
                st.subheader(":material/password: Ingreso clave encryptada")

                st.text_input(
                    "Ingrese clave encryptada:",
                    type="password",
                    key="clave"
                )

                st.button(
                    "Ingresar",
                    key="ingresar",
                    use_container_width=True,
                    on_click=verificar_clave
                )

            # --- Menú solo si el acceso es correcto ---
            if st.session_state.ingreso:
                with st.container(border=True):
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

                    with col1:
                        if st.button("Empleo", key="acceso", use_container_width=True):
                            st.switch_page("pages/1_empleo.py")

                    with col2:
                        if st.button("Desocupación", key="acceso1", use_container_width=True):
                            st.switch_page("pages/2_desocupacion.py")

                    with col3:
                        if st.button("Subocupación", key="acceso2", use_container_width=True):
                            st.switch_page("pages/3_subocupacion.py") 

                    with col4:
                        if st.button("Informalidad Laboral", key="acceso3", use_container_width=True):
                            st.switch_page("pages/4_informalidad.py")


