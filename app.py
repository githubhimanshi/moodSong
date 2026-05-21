"""
MoodTune Backend — Claude Vision Emotion Detection
pip install fastapi uvicorn python-multipart pillow numpy anthropic
uvicorn app:app --reload --port 8000
Set your API key:  export ANTHROPIC_API_KEY=sk-ant-...
"""

import io, base64, os, struct, json
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("index.html")


# ── Anthropic client ───────────────────────────────────────────────────────

_anthropic_client = None

def get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        try:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            _anthropic_client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise RuntimeError("anthropic package not installed")
    return _anthropic_client


# ── Claude Vision emotion detection ───────────────────────────────────────

def detect_emotion_claude(img: Image.Image) -> dict:
    try:
        client = get_anthropic()
        buf = io.BytesIO()
        img_rgb = img.convert("RGB")
        w, h = img_rgb.size
        if max(w, h) > 1024:
            ratio = 1024 / max(w, h)
            img_rgb = img_rgb.resize((int(w*ratio), int(h*ratio)), Image.LANCZOS)
        img_rgb.save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        prompt = """Analyze this image and detect the primary human emotion or mood conveyed.
Consider: facial expression, body language, lighting, color tone, composition, and overall atmosphere.
Respond ONLY with a JSON object, no markdown, no explanation:
{
  "label": "<one of: happy, sad, angry, fear, surprise, neutral, contempt, melancholic, anxious, peaceful, excited, nostalgic, mysterious>",
  "score": <float 0.0-1.0, your confidence>,
  "scene": "<2-3 word scene description>",
  "reasoning": "<one sentence explaining why>"
}"""

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                {"type": "text", "text": prompt}
            ]}],
        )

        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        result = json.loads(raw.strip())

        label     = result.get("label", "neutral").lower()
        score     = float(result.get("score", 0.75))
        scene     = result.get("scene", "cinematic portrait")
        reasoning = result.get("reasoning", "")
        print(f"  [Claude Vision] label={label}, score={score:.2f}, scene={scene}")
        print(f"  [Claude Vision] reason={reasoning}")
        return {"label": label, "score": min(score, 0.99), "scene": scene, "reasoning": reasoning, "source": "claude"}

    except Exception as e:
        print(f"  [Claude Vision] FAILED: {e} — falling back to pixel analysis")
        result = detect_emotion_offline(img)
        result["source"] = "pixel_fallback"
        return result


def detect_emotion_offline(img: Image.Image) -> dict:
    arr        = np.array(img.convert("RGB")).astype(float)
    brightness = np.mean(arr) / 255.0
    r_mean, g_mean, b_mean = [np.mean(arr[:,:,i])/255 for i in range(3)]
    max_c, min_c = np.max(arr/255, axis=2), np.min(arr/255, axis=2)
    saturation   = float(np.mean(max_c - min_c))
    contrast     = float(np.std(arr) / 128.0)
    warmth       = (r_mean - b_mean + 1) / 2
    channel_spread = max(abs(r_mean-g_mean), abs(g_mean-b_mean), abs(r_mean-b_mean))
    is_greyscale   = channel_spread < 0.04 and saturation < 0.08
    h = arr.shape[0]
    lum = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
    bottom_heavy = max(0.0, (np.mean(lum[h//2:,:]) - np.mean(lum[:h//2,:])) / 255.0)

    scores = {
        "happy":       brightness*0.4 + saturation*0.4 + warmth*0.2,
        "sad":         (1-brightness)*0.35 + (1-min(saturation*8,1))*0.40 + bottom_heavy*0.25,
        "melancholic": (1-abs(brightness-0.55))*0.3 + (1-min(saturation*6,1))*0.45 + (1-warmth)*0.25,
        "angry":       contrast*0.4 + warmth*0.35 + saturation*0.25,
        "fear":        (1-brightness)*0.4 + contrast*0.35 + (1-min(saturation*4,1))*0.25,
        "surprise":    saturation*0.45 + contrast*0.3 + brightness*0.25,
        "neutral":     (1-abs(brightness-0.5))*0.4 + min(saturation*3,1)*0.4 + (1-contrast)*0.2,
        "mysterious":  (1-brightness)*0.35 + (1-min(saturation*5,1))*0.35 + contrast*0.30,
    }
    if is_greyscale: scores["neutral"] *= 0.15
    best  = max(scores, key=scores.get)
    score = round(scores[best], 3)
    scene = ("monochrome melancholic" if is_greyscale and brightness<0.6
             else "vibrant joyful" if brightness>0.65 and saturation>0.25
             else "dark mysterious" if brightness<0.35
             else "warm passionate" if warmth>0.65 and saturation>0.3
             else "desaturated somber" if saturation<0.1 else "neutral cinematic")
    print(f"  → Detected: {best} ({int(score*100)}%) | scene: {scene}")
    return {"label": best, "score": min(score,0.99), "scene": scene, "reasoning": "", "source": "pixel"}


# ── Color helpers ──────────────────────────────────────────────────────────

def dominant_colors(img, count=5):
    from collections import Counter
    arr = (np.array(img.resize((100,100)).convert("RGB")) // 32) * 32
    top = Counter([tuple(r) for r in arr.reshape(-1,3)]).most_common(count)
    return [[int(v) for v in c[0]] for c in top]

def colors_to_mood(colors):
    warm=cold=bright=dark=0
    for r,g,b in colors:
        if r>150 and g<120: warm+=1
        elif b>150 and r<120: cold+=1
        lum=0.299*r+0.587*g+0.114*b
        if lum>160: bright+=1
        elif lum<80: dark+=1
    out=[]
    if warm>=2: out.append("warm passionate")
    if cold>=2: out.append("cool melancholic")
    if dark>=2: out.append("mysterious dark")
    if bright>=2: out.append("vibrant uplifting")
    return out or ["neutral cinematic"]

def build_prompts(emotion_label, scene, color_mood):
    emap = {
        "happy":      ("upbeat pop",120,"acoustic guitar, bright synths"),
        "sad":        ("lo-fi sad",68,"soft piano, cello"),
        "melancholic":("ambient melancholic",72,"piano, strings"),
        "angry":      ("intense rock",142,"electric guitar, heavy drums"),
        "fear":       ("dark ambient horror",55,"eerie pads, dissonance"),
        "neutral":    ("cinematic ambient",90,"orchestral pads"),
        "surprise":   ("cinematic pop",128,"bright brass"),
        "contempt":   ("noir jazz",84,"saxophone, moody piano"),
        "anxious":    ("tense electronica",112,"arpeggios, dark synths"),
        "peaceful":   ("ambient peaceful",60,"soft pads, acoustic guitar"),
        "excited":    ("euphoric dance",138,"synths, punchy drums"),
        "nostalgic":  ("lo-fi nostalgic",80,"warm piano, vinyl crackle"),
        "mysterious": ("dark cinematic",76,"cello, textured pads"),
    }
    genre, bpm, instr = emap.get(emotion_label.lower(), emap["neutral"])
    base = f"{genre}, {instr}, {scene}, {', '.join(color_mood)}"
    return [
        (f"{base}, intimate solo",  bpm, "solo",    emotion_label),
        (f"{base}, full band",      bpm, "full",    emotion_label),
        (f"{base}, minimal sparse", bpm, "minimal", emotion_label),
    ]


# ══════════════════════════════════════════════════════════════════════════
#  AUDIO UTILITIES
# ══════════════════════════════════════════════════════════════════════════

SR = 44100

def _to_wav(audio):
    audio = np.clip(audio, -1, 1)
    mx = np.max(np.abs(audio))
    if mx > 0: audio = audio / mx * 0.88
    fade = min(int(SR*1.5), len(audio)//4)
    audio[:fade]  *= np.linspace(0, 1, fade)
    audio[-fade:] *= np.linspace(1, 0, fade)
    pcm = (audio * 32767).astype(np.int16)
    ds  = len(pcm) * 2
    w   = struct.pack('<4sI4s', b'RIFF', 36+ds, b'WAVE')
    w  += struct.pack('<4sIHHIIHH', b'fmt ', 16, 1, 1, SR, SR*2, 2, 16)
    w  += struct.pack('<4sI', b'data', ds)
    w  += pcm.tobytes()
    return w

def _f(midi):  return 440.0 * (2.0 ** ((midi - 69) / 12.0))
def _env(n, a, d, s, r):
    e = np.ones(n)
    ai = min(int(a*n), n); di = min(int(d*n), n-ai)
    ri = min(int(r*n), n-ai-di); si = n-ai-di-ri
    if si < 0: si = 0; ri = n-ai-di
    if ai: e[:ai]                     = np.linspace(0, 1, ai)
    if di: e[ai:ai+di]                = np.linspace(1, s, di)
    if si: e[ai+di:ai+di+si]          = s
    if ri: e[ai+di+si:ai+di+si+ri]    = np.linspace(s, 0, ri)
    return e

def _sine(f, t): return np.sin(2*np.pi*f*t)
def _saw(f, t):  return 2*(f*t - np.floor(f*t + 0.5))
def _tri(f, t):  return 2*np.abs(2*(f*t - np.floor(f*t + 0.5))) - 1

def _reverb(a, ms=80, dec=0.4):
    d = int(SR*ms/1000); out = a.copy()
    if d < len(a):   out[d:]   += a[:-d]   * dec
    if 2*d < len(a): out[2*d:] += a[:-2*d] * dec*dec
    if 3*d < len(a): out[3*d:] += a[:-3*d] * dec**3
    return out

def _hit(out, sig, pos):
    if pos < 0 or pos >= len(out): return
    e = min(pos + len(sig), len(out))
    if e > pos: out[pos:e] += sig[:e-pos]

# Variation transpositions: 3 distinct keys per emotion family
_VAR_TRANSPOSE = [0, 5, 7]   # root / IV / V — always musically related

# Variation durations: each song is a different length so WAV sizes differ
# and the musical phrase structure lands differently
_VAR_DURATION  = [20, 23, 18]   # seconds: standard / extended / compact

# Variation oscillator choices: changes timbral character completely
_VAR_OSC       = [_sine, _tri, _saw]   # pure / hollow / bright

# ══════════════════════════════════════════════════════════════════════════
#  GENERATORS  (style controls layers; variation controls key + pattern)
# ══════════════════════════════════════════════════════════════════════════

# ── HAPPY ─────────────────────────────────────────────────────────────────

def gen_happy(bpm=120, duration=20, style="full", variation=0):
    """
    solo    → melody + light chords, no drums
    full    → chords + bass + melody + drums  (brightest)
    minimal → bass + chords only, half-time feel
    variation 0/1/2 → C / F / G major, sine / triangle / saw timbre
    """
    duration = _VAR_DURATION[variation % 3]
    N    = SR * duration
    out  = np.zeros(N)
    beat = int(SR * 60 / bpm)
    tp   = _VAR_TRANSPOSE[variation % 3]
    osc  = _VAR_OSC[variation % 3]

    # Rhythmic feel per variation
    swing_ratios = [0.5, 0.55, 0.625]          # straight / light swing / shuffle
    sw = swing_ratios[variation % 3]

    scale  = [60+tp, 62+tp, 64+tp, 65+tp, 67+tp, 69+tp, 71+tp, 72+tp]
    chords = [
        [60+tp, 64+tp, 67+tp],
        [65+tp, 69+tp, 72+tp],
        [67+tp, 71+tp, 74+tp],
        [60+tp, 64+tp, 67+tp],
    ]

    use_chords = style in ("full", "minimal")
    use_bass   = style in ("full", "minimal")
    use_melody = style in ("solo", "full")
    use_drums  = style == "full"

    cl = beat * 4

    # Chords
    if use_chords:
        for ci, st in enumerate(range(0, N, cl)):
            ch = chords[ci % 4]
            en = min(st + cl, N); sn = en - st
            sg = np.linspace(0, sn/SR, sn); seg = np.zeros(sn)
            for m in ch: seg += osc(_f(m), sg) * 0.20
            seg += osc(_f(ch[0]+12), sg) * 0.15
            amp = 0.9 if style == "full" else 0.6
            out[st:en] += seg * _env(sn, 0.02, 0.1, 0.8, 0.1) * amp

    # Bass (bouncy on full, steady on minimal)
    if use_bass:
        bass_roots = [chords[i % 4][0] - 12 for i in range(8)]
        step = beat // 2 if style == "full" else beat
        for bi, st in enumerate(range(0, N, step)):
            if st >= N: break
            m = bass_roots[bi % 8]
            nn = min(step, N-st); nt = np.linspace(0, nn/SR, nn)
            _hit(out, _sine(_f(m), nt) * _env(nn, 0.01, 0.1, 0.7, 0.2) * 0.28, st)

    # Melody
    if use_melody:
        mel_patterns = [
            [0, 2, 4, 5, 4, 2, 0, 2],   # ascending-descending
            [4, 5, 4, 2, 0, 2, 4, 5],   # starts on 3rd
            [0, 2, 4, 7, 4, 2, 4, 0],   # wider leap
        ]
        mel = mel_patterns[variation % 3]
        mb  = int(beat * sw * 2)
        for mi, st in enumerate(range(0, N, mb)):
            if st >= N: break
            m  = scale[mel[mi % 8]]
            nn = min(mb, N-st); nt = np.linspace(0, nn/SR, nn)
            _hit(out, _tri(_f(m+12), nt) * _env(nn, 0.01, 0.05, 0.75, 0.2) * 0.22, st)

    # Drums
    if use_drums:
        k_t = np.linspace(0, 0.15, int(SR*0.15))
        k_f = np.linspace(180, 35, len(k_t))
        kick  = np.sin(2*np.pi*k_f*k_t) * np.exp(-k_t*25) * 0.9
        s_t   = np.linspace(0, 0.12, int(SR*0.12))
        snare = (np.sin(2*np.pi*220*s_t)*np.exp(-s_t*30)*0.5
                 + np.random.randn(len(s_t))*np.exp(-s_t*20)*0.5) * 0.7
        hh    = np.random.randn(int(SR*0.04)) * np.exp(-np.linspace(0,1,int(SR*0.04))*50) * 0.35
        for b in range(0, N, beat):
            bn = (b // beat) % 4
            _hit(out, kick,  b)
            _hit(out, hh,    b)
            _hit(out, hh,    b + int(beat * sw))
            if bn in [1, 3]: _hit(out, snare, b)

    return out


# ── SAD ───────────────────────────────────────────────────────────────────

def gen_sad(bpm=60, duration=20, style="solo", variation=0):
    """
    solo    → slow arpeggiated piano + weeping melody, no bass/drums
    full    → piano + melody + bass drone + soft brush drums
    minimal → bass drone + sparse chord pads only
    variation 0/1/2 → Am / Dm / Em, sine / triangle / saw timbre
    """
    duration = _VAR_DURATION[variation % 3]
    N    = SR * duration
    out  = np.zeros(N)
    beat = int(SR * 60 / bpm)
    tp   = _VAR_TRANSPOSE[variation % 3]
    osc  = _VAR_OSC[variation % 3]

    chord_sets = [
        [[57+tp,60+tp,64+tp],[62+tp,65+tp,69+tp],[55+tp,59+tp,62+tp],[57+tp,60+tp,64+tp]],
        [[57+tp,60+tp,64+tp],[60+tp,63+tp,67+tp],[55+tp,58+tp,62+tp],[57+tp,60+tp,64+tp]],
        [[52+tp,55+tp,59+tp],[57+tp,60+tp,64+tp],[55+tp,59+tp,62+tp],[52+tp,55+tp,59+tp]],
    ]
    chords = chord_sets[variation % 3]

    weep_patterns = [
        [69+tp,67+tp,65+tp,64+tp,62+tp,60+tp,62+tp,64+tp],
        [64+tp,62+tp,60+tp,59+tp,57+tp,55+tp,57+tp,59+tp],
        [67+tp,65+tp,64+tp,62+tp,60+tp,59+tp,60+tp,62+tp],
    ]
    weep = weep_patterns[variation % 3]

    use_chords = style in ("solo", "full")
    use_melody = style in ("solo", "full")
    use_bass   = style in ("full", "minimal")
    use_drums  = style == "full"

    cl = beat * 4

    # Arpeggiated piano chords
    if use_chords:
        for ci, st in enumerate(range(0, N, cl)):
            ch = chords[ci % 4]
            for ni, m in enumerate(ch):
                ns = st + ni * beat
                if ns >= N: continue
                nn = min(int(beat*2.5), N-ns)
                nt = np.linspace(0, nn/SR, nn)
                wave = osc(_f(m), nt) + osc(_f(m+12), nt) * 0.3
                seg  = _reverb(wave * _env(nn, 0.01, 0.15, 0.4, 0.6) * 0.30, ms=120, dec=0.5)
                _hit(out, seg, ns)

    # Weeping melody with vibrato
    if use_melody:
        mb  = int(beat * 1.5)
        vib_rates = [4.5, 5.0, 4.0]
        vr = vib_rates[variation % 3]
        for mi, st in enumerate(range(0, N, mb)):
            if st >= N: break
            m  = weep[mi % 8]
            nn = min(mb, N-st); nt = np.linspace(0, nn/SR, nn)
            vib  = 1.0 + 0.005 * np.sin(2*np.pi*vr*nt)
            wave = np.sin(2*np.pi*_f(m)*vib*nt)
            seg  = _reverb(wave * _env(nn, 0.1, 0.2, 0.5, 0.5) * 0.18, ms=200, dec=0.55)
            _hit(out, seg, st)

    # Bass drone
    if use_bass:
        bass_roots = [[45+tp,50+tp,43+tp,45+tp],
                      [40+tp,45+tp,38+tp,40+tp],
                      [52+tp,47+tp,50+tp,52+tp]]
        bass = bass_roots[variation % 3]
        for bi, st in enumerate(range(0, N, beat*4)):
            en = min(st+beat*4, N); sn = en-st
            sg = np.linspace(0, sn/SR, sn)
            m  = bass[bi % 4]
            wave = _sine(_f(m), sg)*0.6 + _sine(_f(m+7), sg)*0.2
            out[st:en] += wave * _env(sn, 0.2, 0.3, 0.6, 0.4) * 0.22

    # Soft brushed drums
    if use_drums:
        brush_t = np.linspace(0, 0.15, int(SR*0.15))
        brush   = np.random.randn(len(brush_t)) * np.exp(-brush_t*20) * 0.30
        for b in range(0, N, beat):
            if (b // beat) % 4 in [1, 3]: _hit(out, brush, b)

    return out


# ── FEAR ──────────────────────────────────────────────────────────────────

def gen_fear(bpm=55, duration=20, style="full", variation=0):
    """
    solo    → dissonant pads + slow heartbeat, no stabs/scrape
    full    → all layers (pads, stabs, heartbeat, scrape noise)
    minimal → heartbeat + deep drone only
    variation 0/1/2 → different dissonance clusters / heartbeat tempo
    """
    duration = _VAR_DURATION[variation % 3]
    N    = SR * duration
    out  = np.zeros(N)
    beat = int(SR * 60 / bpm)
    tp   = _VAR_TRANSPOSE[variation % 3]
    t    = np.linspace(0, duration, N)

    cluster_sets = [
        [[35+tp,41+tp],[34+tp,40+tp],[36+tp,42+tp,45+tp],[33+tp,39+tp,46+tp]],
        [[32+tp,38+tp],[33+tp,39+tp],[34+tp,40+tp,44+tp],[31+tp,37+tp,43+tp]],
        [[36+tp,42+tp],[35+tp,41+tp],[37+tp,43+tp,46+tp],[34+tp,40+tp,47+tp]],
    ]
    clusters = cluster_sets[variation % 3]

    use_pads     = style in ("solo", "full", "minimal")
    use_stabs    = style == "full"
    use_heartbeat= style in ("solo", "full", "minimal")
    use_scrape   = style == "full"

    cl = beat * 4

    # Dissonant pad clusters
    if use_pads:
        for ci, st in enumerate(range(0, N, cl)):
            cl_ = clusters[ci % 4]
            en  = min(st+cl, N); sn = en-st
            sg  = np.linspace(0, sn/SR, sn); seg = np.zeros(sn)
            beat_freq = [7.3, 6.1, 8.7][variation % 3]
            for m in cl_:
                seg += _sine(_f(m),    sg) * 0.25
                seg += _sine(_f(m)*1.008, sg) * 0.20
                seg += _sine(_f(m)*0.992, sg) * 0.18
            lfo = 0.4 + 0.6*np.abs(np.sin(2*np.pi*beat_freq*sg))
            seg = _reverb(seg * _env(sn, 0.3, 0.4, 0.7, 0.4) * lfo, ms=300, dec=0.7)
            out[st:en] += seg

    # Jump-scare stabs
    if use_stabs:
        np.random.seed(13 + variation)
        stab_t = [int(x) for x in sorted(np.random.choice(N-SR, 8, replace=False))]
        stab_n = [72+tp,73+tp,74+tp,71+tp,75+tp,70+tp,76+tp,69+tp]
        for i, st in enumerate(stab_t):
            nn  = int(SR*0.08); nt = np.linspace(0, 0.08, nn)
            wave= _saw(_f(stab_n[i%8]), nt)
            seg = _reverb(wave * _env(nn, 0.001, 0.05, 0.3, 0.6) * 0.55, ms=60, dec=0.5)
            _hit(out, seg, st)

    # Heartbeat (rate varies by variation)
    if use_heartbeat:
        lub_gap  = [0.18, 0.22, 0.15][variation % 3]
        dub_gap  = [0.65, 0.80, 0.55][variation % 3]
        np.random.seed(7 + variation)
        thud_t  = np.linspace(0, 0.3, int(SR*0.3))
        thud    = np.sin(2*np.pi*np.linspace(60,25,len(thud_t))*thud_t) * np.exp(-thud_t*15)
        pos = 0
        while pos < N - len(thud)*2:
            _hit(out, thud,      pos); pos += int(SR * lub_gap)
            _hit(out, thud*0.6,  pos); pos += int(SR * (dub_gap + np.random.rand()*0.4))

    # High scraping noise
    if use_scrape:
        scrape = np.random.randn(N) * 0.06
        scrape *= np.sin(2*np.pi*np.linspace(3000,1200,N)/SR*np.arange(N))
        scrape *= 0.5 + 0.5*np.sin(2*np.pi*0.3*t)
        out += scrape

    return out


# ── ANGRY ─────────────────────────────────────────────────────────────────

def gen_angry(bpm=145, duration=20, style="full", variation=0):
    """
    solo    → distorted guitar riff only, no drums/bass
    full    → power chords + distorted bass + aggressive drums
    minimal → bass riff + sparse kick, half density
    variation 0/1/2 → E / A / D power-chord roots, different rhythm patterns
    """
    duration = _VAR_DURATION[variation % 3]
    N    = SR * duration
    out  = np.zeros(N)
    beat = int(SR * 60 / bpm)
    tp   = _VAR_TRANSPOSE[variation % 3]
    osc  = _VAR_OSC[variation % 3]

    root_sets = [
        [40+tp, 45+tp, 38+tp, 40+tp],
        [45+tp, 40+tp, 43+tp, 45+tp],
        [38+tp, 43+tp, 41+tp, 38+tp],
    ]
    roots = root_sets[variation % 3]

    # Rhythm patterns: list of beat-offsets within 2-beat cell where hits land
    hit_patterns = [
        [0],                   # straight on the bar
        [0, beat//3],          # syncopated triplet feel
        [0, beat*3//4],        # dotted feel
    ]
    hit_offsets = hit_patterns[variation % 3]

    use_guitar = style in ("solo", "full")
    use_bass   = style in ("full", "minimal")
    use_drums  = style in ("full", "minimal")

    cl = beat * 2

    # Distorted power chords
    if use_guitar:
        for ci, st in enumerate(range(0, N, cl)):
            r  = roots[ci % 4]
            en = min(st+cl, N); sn = en-st
            sg = np.linspace(0, sn/SR, sn)
            wave = (_saw(_f(r),    sg) * 0.4
                  + _saw(_f(r+7),  sg) * 0.35
                  + _saw(_f(r+12), sg) * 0.25)
            wave = np.tanh(wave * 3.5) * 0.5
            for off in hit_offsets:
                pos = st + off
                if pos + sn//len(hit_offsets) < N:
                    seg_len = sn // len(hit_offsets)
                    seg = wave[:seg_len] * _env(seg_len, 0.005, 0.05, 0.95, 0.05)
                    _hit(out, seg, pos)

    # Distorted bass
    if use_bass:
        density = beat * 2 if style == "minimal" else cl
        for ci, st in enumerate(range(0, N, density)):
            r  = roots[ci % 4] - 12
            en = min(st+density, N); sn = en-st
            sg = np.linspace(0, sn/SR, sn)
            wave = np.tanh(_saw(_f(r), sg) * 2) * 0.5
            out[st:en] += wave * _env(sn, 0.01, 0.05, 0.9, 0.1) * 0.38

    # Drums
    if use_drums:
        k_t   = np.linspace(0, 0.12, int(SR*0.12))
        k_f   = np.linspace(200, 30, len(k_t))
        kick  = np.sin(2*np.pi*k_f*k_t) * np.exp(-k_t*30) * 1.0
        s_t   = np.linspace(0, 0.1, int(SR*0.1))
        snare = (np.sin(2*np.pi*280*s_t)*np.exp(-s_t*35)*0.6
                 + np.random.randn(len(s_t))*np.exp(-s_t*25)*0.8) * 0.9
        hh    = np.random.randn(int(SR*0.025)) * np.exp(-np.linspace(0,1,int(SR*0.025))*80) * 0.4

        drum_density = 2 if style == "minimal" else 1  # skip alternate beats for minimal
        for bi, b in enumerate(range(0, N, beat)):
            if style == "minimal" and bi % drum_density != 0: continue
            bn = (b // beat) % 4
            _hit(out, kick, b)
            if bn == 0: _hit(out, kick, b + beat//4)
            if bn in [1, 3]:
                _hit(out, snare*1.2, b)
                _hit(out, snare*0.8, b + beat//4)
            _hit(out, hh, b)
            _hit(out, hh, b + beat//2)

    return out


# ── PEACEFUL ──────────────────────────────────────────────────────────────

def gen_peaceful(bpm=58, duration=20, style="solo", variation=0):
    """
    solo    → floating melody only over breath drone
    full    → pads + melody + breath drone
    minimal → pads + drone, no melody
    variation 0/1/2 → G / C / D pentatonic, sine / tri / saw timbre
    """
    duration = _VAR_DURATION[variation % 3]
    N    = SR * duration
    out  = np.zeros(N)
    beat = int(SR * 60 / bpm)
    tp   = _VAR_TRANSPOSE[variation % 3]
    osc  = _VAR_OSC[variation % 3]
    t    = np.linspace(0, duration, N)

    pad_sets = [
        [[55+tp,59+tp,62+tp],[57+tp,62+tp,66+tp],[52+tp,57+tp,62+tp],[55+tp,59+tp,62+tp]],
        [[52+tp,55+tp,59+tp],[55+tp,60+tp,64+tp],[50+tp,55+tp,59+tp],[52+tp,55+tp,59+tp]],
        [[57+tp,60+tp,64+tp],[60+tp,64+tp,67+tp],[55+tp,59+tp,62+tp],[57+tp,60+tp,64+tp]],
    ]
    pads = pad_sets[variation % 3]

    mel_sets = [
        [64+tp,67+tp,69+tp,71+tp,67+tp,64+tp,62+tp,59+tp],
        [67+tp,69+tp,71+tp,74+tp,71+tp,69+tp,67+tp,64+tp],
        [62+tp,64+tp,67+tp,69+tp,67+tp,64+tp,62+tp,59+tp],
    ]
    mel = mel_sets[variation % 3]

    # Bass root for drone
    drone_root = [43+tp, 40+tp, 45+tp][variation % 3]

    use_pads   = style in ("full", "minimal")
    use_melody = style in ("solo", "full")
    use_drone  = True   # always present — anchors the piece

    cl = beat * 6

    # Pads
    if use_pads:
        for ci, st in enumerate(range(0, N, cl)):
            ch  = pads[ci % 4]
            en  = min(st+cl, N); sn = en-st
            sg  = np.linspace(0, sn/SR, sn); seg = np.zeros(sn)
            for m in ch:
                seg += osc(_f(m),       sg) * 0.22
                seg += osc(_f(m)*1.003, sg) * 0.12
            seg = _reverb(seg * _env(sn, 0.5, 0.3, 0.7, 0.6), ms=250, dec=0.65)
            out[st:en] += seg

    # Melody
    if use_melody:
        mb       = int(beat * 2)
        vib_rate = [3.5, 4.0, 3.0][variation % 3]
        for mi, st in enumerate(range(0, N, mb)):
            if st >= N: break
            m  = mel[mi % 8]
            nn = min(mb, N-st); nt = np.linspace(0, nn/SR, nn)
            vib  = 1.0 + 0.003*np.sin(2*np.pi*vib_rate*nt)
            wave = osc(_f(m)*vib, nt) if osc != _sine else np.sin(2*np.pi*_f(m)*vib*nt)
            seg  = _reverb(wave * _env(nn, 0.4, 0.4, 0.5, 0.6) * 0.15, ms=300, dec=0.6)
            _hit(out, seg, st)

    # Breathing drone
    if use_drone:
        breath_rate = [0.15, 0.12, 0.18][variation % 3]
        breath = 0.3 + 0.4*np.sin(2*np.pi*breath_rate*t)
        out   += _sine(_f(drone_root), t) * 0.15 * breath

    return out


# ── EXCITED ───────────────────────────────────────────────────────────────

def gen_excited(bpm=140, duration=20, style="full", variation=0):
    """
    solo    → arpeggio melody only, no drums/bass
    full    → stabs + arpeggio + bass + 4-on-floor drums
    minimal → bass + kick, stripped back
    variation 0/1/2 → D / G / A, different arpeggio patterns
    """
    duration = _VAR_DURATION[variation % 3]
    N    = SR * duration
    out  = np.zeros(N)
    beat = int(SR * 60 / bpm)
    tp   = _VAR_TRANSPOSE[variation % 3]
    osc  = _VAR_OSC[variation % 3]

    chord_sets = [
        [[62+tp,66+tp,69+tp],[67+tp,71+tp,74+tp],[69+tp,73+tp,76+tp],[62+tp,66+tp,69+tp]],
        [[67+tp,71+tp,74+tp],[72+tp,76+tp,79+tp],[74+tp,78+tp,81+tp],[67+tp,71+tp,74+tp]],
        [[69+tp,73+tp,76+tp],[74+tp,78+tp,81+tp],[76+tp,80+tp,83+tp],[69+tp,73+tp,76+tp]],
    ]
    chords = chord_sets[variation % 3]

    arp_patterns = [
        [74+tp,76+tp,78+tp,81+tp,78+tp,76+tp,74+tp,71+tp],
        [76+tp,78+tp,81+tp,83+tp,81+tp,78+tp,76+tp,74+tp],
        [71+tp,74+tp,76+tp,78+tp,76+tp,74+tp,71+tp,69+tp],
    ]
    arp = arp_patterns[variation % 3]

    bass_sets = [
        [50+tp,55+tp,57+tp,50+tp],
        [55+tp,60+tp,62+tp,55+tp],
        [57+tp,62+tp,64+tp,57+tp],
    ]
    bass_r = bass_sets[variation % 3]

    use_chords = style in ("full",)
    use_arp    = style in ("solo", "full")
    use_bass   = style in ("full", "minimal")
    use_drums  = style in ("full", "minimal")

    cl = beat * 2

    # Stab chords
    if use_chords:
        for ci, st in enumerate(range(0, N, cl)):
            ch  = chords[ci % 4]
            en  = min(st+cl, N); sn = en-st
            sg  = np.linspace(0, sn/SR, sn); seg = np.zeros(sn)
            for m in ch:
                seg += osc(_f(m),    sg) * 0.22
                seg += osc(_f(m+12), sg) * 0.12
            out[st:en] += seg * _env(sn, 0.005, 0.08, 0.85, 0.08)

    # Arpeggio
    if use_arp:
        ab = beat // 2
        for ai, st in enumerate(range(0, N, ab)):
            if st >= N: break
            m  = arp[ai % 8]
            nn = max(ab - int(SR*0.01), 100); nn = min(nn, N-st)
            nt = np.linspace(0, nn/SR, nn)
            _hit(out, osc(_f(m), nt) * _env(nn, 0.005, 0.05, 0.8, 0.15) * 0.28, st)

    # Syncopated bass
    if use_bass:
        for bi, st in enumerate(range(0, N, beat*4)):
            r = bass_r[bi % 4] - 12
            skip_beats = [1] if style == "minimal" else [1]  # syncopation
            for si in range(16):
                if si % 4 in skip_beats: continue
                pos = st + si * beat//4
                if pos >= N: break
                nn = min(beat//4 - 100, N-pos); nt = np.linspace(0, nn/SR, nn)
                _hit(out, _saw(_f(r), nt) * _env(nn, 0.005, 0.1, 0.8, 0.1) * 0.30, pos)

    # Drums
    if use_drums:
        k_t   = np.linspace(0, 0.1, int(SR*0.1))
        k_f   = np.linspace(150, 35, len(k_t))
        kick  = np.sin(2*np.pi*k_f*k_t) * np.exp(-k_t*35) * 0.95
        clap_t= np.linspace(0, 0.08, int(SR*0.08))
        clap  = np.random.randn(len(clap_t)) * np.exp(-clap_t*40) * 0.8
        hh    = np.random.randn(int(SR*0.03)) * np.exp(-np.linspace(0,1,int(SR*0.03))*70) * 0.3
        for b in range(0, N, beat):
            bn = (b // beat) % 4
            _hit(out, kick, b)
            if bn in [1, 3]: _hit(out, clap, b)
            if style == "full":
                _hit(out, hh, b)
                _hit(out, hh, b + beat//2)

    return out


# ── NOSTALGIC ─────────────────────────────────────────────────────────────

def gen_nostalgic(bpm=80, duration=20, style="solo", variation=0):
    """
    solo    → warm piano arpeggios + melody, no crackle/bass/drums
    full    → piano + melody + walking bass + brushed drums + vinyl crackle
    minimal → walking bass + chords + crackle, no melody/drums
    variation 0/1/2 → C / F / G maj7, different walking bass lines
    """
    duration = _VAR_DURATION[variation % 3]
    N    = SR * duration
    out  = np.zeros(N)
    beat = int(SR * 60 / bpm)
    tp   = _VAR_TRANSPOSE[variation % 3]
    osc  = _VAR_OSC[variation % 3]

    chord_sets = [
        [[60+tp,64+tp,67+tp,74+tp],[65+tp,69+tp,72+tp,76+tp],
         [55+tp,59+tp,62+tp,69+tp],[60+tp,64+tp,67+tp,69+tp]],
        [[65+tp,69+tp,72+tp,79+tp],[60+tp,64+tp,67+tp,74+tp],
         [60+tp,63+tp,67+tp,74+tp],[65+tp,69+tp,72+tp,76+tp]],
        [[67+tp,71+tp,74+tp,81+tp],[62+tp,66+tp,69+tp,76+tp],
         [62+tp,65+tp,69+tp,76+tp],[67+tp,71+tp,74+tp,79+tp]],
    ]
    chords = chord_sets[variation % 3]

    mel_sets = [
        [72+tp,71+tp,69+tp,67+tp,69+tp,71+tp,72+tp,74+tp],
        [77+tp,76+tp,74+tp,72+tp,74+tp,76+tp,77+tp,79+tp],
        [79+tp,78+tp,76+tp,74+tp,76+tp,78+tp,79+tp,81+tp],
    ]
    mel = mel_sets[variation % 3]

    bass_sets = [
        [48+tp,50+tp,52+tp,50+tp,53+tp,52+tp,50+tp,48+tp],
        [53+tp,55+tp,57+tp,55+tp,58+tp,57+tp,55+tp,53+tp],
        [55+tp,57+tp,59+tp,57+tp,60+tp,59+tp,57+tp,55+tp],
    ]
    bass_w = bass_sets[variation % 3]

    use_chords  = style in ("solo", "full", "minimal")
    use_melody  = style in ("solo", "full")
    use_bass    = style in ("full", "minimal")
    use_drums   = style == "full"
    use_crackle = style in ("full", "minimal")

    cl = beat * 4

    # Piano arpeggios
    if use_chords:
        for ci, st in enumerate(range(0, N, cl)):
            ch = chords[ci % 4]
            for ni, m in enumerate(ch):
                delay = ni * int(SR*0.04); ns = min(delay, cl)
                nn = min(cl - ns, N-st-ns)
                if nn < 100: continue
                nt   = np.linspace(0, nn/SR, nn)
                wave = osc(_f(m), nt)*0.6 + osc(_f(m*2), nt)*0.2
                seg  = _reverb(wave * _env(nn, 0.01, 0.2, 0.45, 0.5) * 0.24, ms=80, dec=0.35)
                _hit(out, seg, st + ns)

    # Warm melody
    if use_melody:
        mb = int(beat * 1.2)
        for mi, st in enumerate(range(0, N, mb)):
            if st >= N: break
            m  = mel[mi % 8]
            nn = min(int(beat*0.9), N-st)
            nt = np.linspace(0, nn/SR, nn)
            _hit(out, osc(_f(m), nt) * _env(nn, 0.05, 0.15, 0.55, 0.45) * 0.20, st)

    # Walking bass
    if use_bass:
        for bi, st in enumerate(range(0, N, beat)):
            if st >= N: break
            m  = bass_w[bi % 8]
            nn = min(beat - int(SR*0.02), N-st)
            nt = np.linspace(0, nn/SR, nn)
            _hit(out, _sine(_f(m), nt) * _env(nn, 0.02, 0.1, 0.6, 0.25) * 0.25, st)

    # Brushed drums
    if use_drums:
        brush_t = np.linspace(0, 0.15, int(SR*0.15))
        brush   = np.random.randn(len(brush_t)) * np.exp(-brush_t*20) * 0.35
        for b in range(0, N, beat):
            if (b // beat) % 4 in [1, 3]: _hit(out, brush, b)

    # Vinyl crackle (amount varies)
    if use_crackle:
        crackle_amp = [0.018, 0.030, 0.012][variation % 3]
        crackle     = np.random.randn(N) * crackle_amp
        crackle[::3] *= 2.5
        out += crackle

    return out


# ── MYSTERIOUS ────────────────────────────────────────────────────────────

def gen_mysterious(bpm=70, duration=20, style="minimal", variation=0):
    """
    solo    → cello drone + sparse water drops only
    full    → dark pads + water drops + cello drone
    minimal → dark pads + drone only, no drops
    variation 0/1/2 → Phrygian / Locrian / Chromatic drop scales
    """
    duration = _VAR_DURATION[variation % 3]
    N    = SR * duration
    out  = np.zeros(N)
    beat = int(SR * 60 / bpm)
    tp   = _VAR_TRANSPOSE[variation % 3]
    osc  = _VAR_OSC[variation % 3]
    t    = np.linspace(0, duration, N)

    chord_sets = [
        [[40+tp,47+tp,53+tp],[41+tp,48+tp,55+tp],[38+tp,45+tp,52+tp],[40+tp,47+tp,53+tp]],
        [[38+tp,45+tp,51+tp],[39+tp,46+tp,53+tp],[36+tp,43+tp,50+tp],[38+tp,45+tp,51+tp]],
        [[41+tp,48+tp,54+tp],[42+tp,49+tp,56+tp],[39+tp,46+tp,53+tp],[41+tp,48+tp,54+tp]],
    ]
    chords = chord_sets[variation % 3]

    # Three modal scales for drops
    drop_scales = [
        [40+tp,41+tp,43+tp,45+tp,47+tp,48+tp,50+tp],   # Phrygian
        [40+tp,41+tp,43+tp,44+tp,46+tp,48+tp,50+tp],   # Locrian
        [40+tp,41+tp,42+tp,44+tp,46+tp,47+tp,50+tp],   # Chromatic-ish
    ]
    scale = drop_scales[variation % 3]

    use_pads  = style in ("full", "minimal")
    use_drops = style in ("solo", "full")
    use_drone = True

    cl = beat * 6

    # Dark detuned pads
    if use_pads:
        for ci, st in enumerate(range(0, N, cl)):
            ch  = chords[ci % 4]
            en  = min(st+cl, N); sn = en-st
            sg  = np.linspace(0, sn/SR, sn); seg = np.zeros(sn)
            for m in ch:
                seg += osc(_f(m),       sg) * 0.20
                seg += osc(_f(m)*1.005, sg) * 0.10
            seg = _reverb(seg * _env(sn, 0.6, 0.4, 0.5, 0.7), ms=400, dec=0.75)
            out[st:en] += seg

    # Sparse water-drop notes
    if use_drops:
        np.random.seed(3 + variation)
        n_drops = [14, 10, 18][variation % 3]
        drop_t  = [int(x) for x in sorted(np.random.choice(N-SR, n_drops, replace=False))]
        for i, st in enumerate(drop_t):
            m  = scale[i % 7] + 24
            nn = int(SR*0.8); nt = np.linspace(0, 0.8, nn)
            wave = _sine(_f(m), nt)
            seg  = _reverb(wave * _env(nn, 0.005, 0.1, 0.0, 0.9) * 0.22, ms=500, dec=0.8)
            _hit(out, seg, st)

    # Cello drone with vibrato
    if use_drone:
        drone_root = [40+tp, 38+tp, 41+tp][variation % 3]
        vib_rate   = [3.2, 2.8, 3.6][variation % 3]
        vib  = 1.0 + 0.006*np.sin(2*np.pi*vib_rate*t)
        bass = (np.sin(2*np.pi*_f(drone_root)*vib*t)
              + np.sin(2*np.pi*_f(drone_root+12)*vib*t) * 0.3)
        swell = 0.3 + 0.4*np.sin(2*np.pi*0.1*t)
        out  += _reverb(bass * swell * 0.20, ms=300, dec=0.6)

    return out


# ── ANXIOUS ───────────────────────────────────────────────────────────────

def gen_anxious(bpm=112, duration=20, style="full", variation=0):
    """
    solo    → fast arpeggio only, no drums/bass/stabs
    full    → staccato stabs + 16th arpeggio + bass + nervous drums
    minimal → bass + sparse stabs, no arpeggio/drums
    variation 0/1/2 → Dm / Em / Bm, different arpeggio speeds
    """
    duration = _VAR_DURATION[variation % 3]
    N    = SR * duration
    out  = np.zeros(N)
    beat = int(SR * 60 / bpm)
    tp   = _VAR_TRANSPOSE[variation % 3]
    osc  = _VAR_OSC[variation % 3]

    scale_sets = [
        [50+tp,52+tp,53+tp,55+tp,57+tp,58+tp,60+tp,62+tp],
        [52+tp,53+tp,55+tp,57+tp,58+tp,60+tp,62+tp,64+tp],
        [47+tp,48+tp,50+tp,52+tp,53+tp,55+tp,57+tp,59+tp],
    ]
    scale = scale_sets[variation % 3]

    chord_sets = [
        [[50+tp,53+tp,57+tp],[53+tp,57+tp,60+tp],[52+tp,55+tp,58+tp],[50+tp,53+tp,57+tp]],
        [[52+tp,55+tp,59+tp],[55+tp,59+tp,62+tp],[53+tp,57+tp,60+tp],[52+tp,55+tp,59+tp]],
        [[47+tp,50+tp,54+tp],[50+tp,54+tp,57+tp],[48+tp,52+tp,55+tp],[47+tp,50+tp,54+tp]],
    ]
    chords = chord_sets[variation % 3]

    bass_sets = [
        [50+tp,50+tp,53+tp,50+tp,55+tp,53+tp,52+tp,50+tp],
        [52+tp,52+tp,55+tp,52+tp,57+tp,55+tp,53+tp,52+tp],
        [47+tp,47+tp,50+tp,47+tp,52+tp,50+tp,48+tp,47+tp],
    ]
    bp = bass_sets[variation % 3]

    # Arpeggio subdivision: 16th / triplet / 32nd
    arp_divs  = [4, 3, 6]
    arp_div   = arp_divs[variation % 3]

    use_stabs  = style in ("full", "minimal")
    use_arp    = style in ("solo", "full")
    use_bass   = style in ("full", "minimal")
    use_drums  = style == "full"

    cl = beat * 2

    # Staccato stabs
    if use_stabs:
        for ci, st in enumerate(range(0, N, cl)):
            ch  = chords[ci % 4]
            en  = min(st+cl, N); sn = en-st
            sg  = np.linspace(0, sn/SR, sn); seg = np.zeros(sn)
            for m in ch: seg += osc(_f(m), sg) * 0.20
            out[st:en] += seg * _env(sn, 0.005, 0.1, 0.4, 0.15)

    # Fast arpeggio
    if use_arp:
        ab = beat // arp_div
        for ai, st in enumerate(range(0, N, ab)):
            if st >= N: break
            m  = scale[ai % 8] + (12 if ai % 3 == 0 else 0)
            nn = max(ab - int(SR*0.01), 100); nn = min(nn, N-st)
            nt = np.linspace(0, nn/SR, nn)
            _hit(out, osc(_f(m), nt) * _env(nn, 0.005, 0.05, 0.6, 0.15) * 0.18, st)

    # Jumpy bass
    if use_bass:
        bb = beat // 2
        for bi, st in enumerate(range(0, N, bb)):
            if st >= N: break
            m  = bp[bi % 8] - 12
            nn = min(bb - int(SR*0.02), N-st); nt = np.linspace(0, nn/SR, nn)
            wave = _saw(_f(m), nt)*0.4 + _sine(_f(m), nt)*0.6
            _hit(out, wave * _env(nn, 0.01, 0.08, 0.65, 0.12) * 0.28, st)

    # Nervous drums
    if use_drums:
        k_t   = np.linspace(0, 0.08, int(SR*0.08))
        k_f   = np.linspace(160, 30, len(k_t))
        kick  = np.sin(2*np.pi*k_f*k_t) * np.exp(-k_t*35) * 0.75
        s_t   = np.linspace(0, 0.07, int(SR*0.07))
        snare = np.random.randn(len(s_t)) * np.exp(-s_t*40) * 0.5
        hh    = np.random.randn(int(SR*0.02)) * np.exp(-np.linspace(0,1,int(SR*0.02))*80) * 0.3
        np.random.seed(99 + variation)
        for b in range(0, N, beat):
            bn = (b // beat) % 4
            if bn in [0, 2]: _hit(out, kick, b)
            if bn in [1, 3]:
                _hit(out, snare, b)
                _hit(out, snare*0.4, b + beat//3)
            for h in range(4):
                if np.random.random() < 0.75: _hit(out, hh, b + h*beat//4)

    return out


# ── Dispatch ───────────────────────────────────────────────────────────────

_GENERATORS = {
    "happy":       gen_happy,
    "sad":         gen_sad,
    "fear":        gen_fear,
    "angry":       gen_angry,
    "peaceful":    gen_peaceful,
    "excited":     gen_excited,
    "nostalgic":   gen_nostalgic,
    "mysterious":  gen_mysterious,
    "anxious":     gen_anxious,
    "melancholic": gen_sad,
    "surprise":    gen_excited,
    "contempt":    gen_mysterious,
    "neutral":     gen_peaceful,
}

_BASE_BPM = {
    "happy":120, "sad":60, "fear":55, "angry":145, "peaceful":58,
    "excited":140, "nostalgic":80, "mysterious":70, "anxious":112,
    "melancholic":65, "surprise":130, "contempt":75, "neutral":72,
}

# Style multipliers: full = reference tempo, solo = slightly slower, minimal = slower still
_STYLE_MULT = {"solo": 0.92, "full": 1.0, "minimal": 0.82}

# Each variation index maps to one style — guarantees all three are used
_VARIATION_STYLES = ["solo", "full", "minimal"]


def generate_song(prompt_text, key_name, bpm_hint, style, index, emotion="neutral"):
    print(f"  Generating song {index+1} — emotion={emotion}, style={style}, variation={index}")
    emo       = emotion.lower()
    gen       = _GENERATORS.get(emo, gen_peaceful)
    base_bpm  = _BASE_BPM.get(emo, bpm_hint)
    bpm       = int(base_bpm * _STYLE_MULT.get(style, 1.0))
    # variation = index ensures each of the 3 songs is in a different key/pattern
    audio     = gen(bpm=bpm, duration=20, style=style, variation=index)
    wav       = _to_wav(audio)
    print(f"  ✓ Song {index+1} done ({emo}/{style}/var{index}) — {round(len(wav)/1024)}KB")
    return base64.b64encode(wav).decode()


# ── Main endpoint ──────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    mime    = file.content_type or ""

    if mime.startswith("video"):
        try:
            import cv2, tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(content); tmp = f.name
            cap = cv2.VideoCapture(tmp)
            ret, frame = cap.read(); cap.release(); os.unlink(tmp)
            if not ret: raise Exception("no frame")
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        except Exception as e:
            raise HTTPException(400, f"Video needs opencv: pip install opencv-python ({e})")
    else:
        try:
            img = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception:
            raise HTTPException(400, "Invalid image file")

    print("\n========== MOODTUNE ==========")
    print("[1] Detecting emotion with Claude Vision...")
    face = detect_emotion_claude(img)

    print("[2] Analyzing colors...")
    colors = dominant_colors(img)
    cmood  = colors_to_mood(colors)

    emotion_label = face["label"]
    scene = face.get("scene", "cinematic portrait")
    print(f"    emotion={emotion_label}, scene={scene}, source={face.get('source')}")

    print("[3] Building music prompts...")
    prompts = build_prompts(emotion_label, scene, cmood)

    print("[4] Generating 3 songs (different keys, layers, patterns)...")
    songs = []
    for i, (prompt_text, bpm, style, emo) in enumerate(prompts):
        # style comes from build_prompts; variation=i gives a different key each time
        audio_b64 = generate_song(prompt_text, "auto", bpm, _VARIATION_STYLES[i], i, emotion=emo)
        songs.append({
            "id":    i + 1,
            "prompt": prompt_text,
            "style": _VARIATION_STYLES[i],
            "audio_b64": audio_b64,
        })

    print("==============================\n")
    return JSONResponse({
        "emotions": {"face": face, "scene": scene, "color_mood": cmood, "palette_rgb": colors},
        "songs": songs,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
