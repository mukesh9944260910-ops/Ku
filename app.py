import asyncio, hashlib, io, json, os, re, subprocess, tempfile, time, wave

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import edge_tts
except ImportError:
    edge_tts = None

st.set_page_config(page_title="SRT → Tamil Dub", page_icon="🎙️", layout="centered")
SR = 24000
API = "https://generativelanguage.googleapis.com/v1beta/models/"
VOICES = {
    "Valluvar (India, male)": "ta-IN-ValluvarNeural", "Pallavi (India, female)": "ta-IN-PallaviNeural",
    "Kumar (Sri Lanka, male)": "ta-LK-KumarNeural", "Saranya (Sri Lanka, female)": "ta-LK-SaranyaNeural",
    "Anbu (Singapore, male)": "ta-SG-AnbuNeural", "Venba (Singapore, female)": "ta-SG-VenbaNeural",
    "Surya (Malaysia, male)": "ta-MY-SuryaNeural", "Kani (Malaysia, female)": "ta-MY-KaniNeural",
}
SAMPLE = "மனித வரலாற்றின் பெரும்பகுதியில்... ஒரு இடத்திலிருந்து இன்னொரு இடத்திற்குச் செல்வது ஒரு சவாலாக இருந்தது."


# ------------------------------------------------------------------ helpers
def sh(*a):
    subprocess.run(a, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def duration(path):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                          "-of", "default=nw=1:nk=1", path]))


def parse_srt(txt):
    txt = txt.replace("\r", "").lstrip("\ufeff")
    out = []
    t = r"(\d+):(\d+):(\d+)[,.](\d+)"
    for blk in re.split(r"\n\s*\n", txt.strip()):
        lines = [l for l in blk.split("\n") if l.strip()]
        for j, l in enumerate(lines):
            m = re.match(t + r"\s*-->\s*" + t, l.strip())
            if m:
                g = m.groups()
                f = lambda h, mi, s, ms: int(h) * 3600 + int(mi) * 60 + int(s) + int(ms.ljust(3, "0")[:3]) / 1000
                text = " ".join(x.strip() for x in lines[j + 1:])
                if text:
                    out.append([f(*g[:4]), f(*g[4:]), re.sub(r"<[^>]+>", "", text)])
                break
    return out


def merge_sentences(rows, max_len=12.0, gap=0.8):
    out = []
    for s, e, t in rows:
        if out and (out[-1][2][-1:] not in ".?!।" and s - out[-1][1] < gap and e - out[-1][0] < max_len):
            out[-1][1], out[-1][2] = e, out[-1][2] + " " + t
        else:
            out.append([s, e, t])
    return out


def call(model, body, key, note=None, tries=8):
    for i in range(tries):
        r = requests.post(f"{API}{model}:generateContent",
                          headers={"x-goog-api-key": key, "Content-Type": "application/json"}, json=body, timeout=300)
        if r.ok:
            return r.json()
        if r.status_code == 429 or r.status_code >= 500:
            wait = min(20 * (i + 1), 90)
            m = re.search(r'"retryDelay":\s*"(\d+)', r.text)
            if m:
                wait = min(int(m.group(1)) + 3, 120)
            if note is not None:
                note.info(f"Gemini busy ({r.status_code}). {wait}s wait ({i + 1}/{tries})...")
            time.sleep(wait)
            continue
        m = re.search(r"use models/([\w.\-]+)", r.text)
        if r.status_code == 404 and m and m.group(1) != model:
            return call(m.group(1), body, key, note, tries)
        raise RuntimeError(f"Gemini error {r.status_code}: {r.text[:300]}")
    raise RuntimeError("Gemini failed after retries")


def gtext(prompt, key, model, note=None):
    j = call(model, {"contents": [{"parts": [{"text": prompt}]}]}, key, note)
    return j["candidates"][0]["content"]["parts"][0]["text"].strip()


def translate(df, dur, key, model, bar, note, batch=25):
    starts = list(df["start"]) + [dur]
    for b in range(0, len(df), batch):
        items = [{"id": int(i), "start": round(df.iloc[i]["start"], 1),
                  "max_words": max(int((starts[i + 1] - starts[i] - 0.1) * 2.0), 2), "english": df.iloc[i]["english"]}
                 for i in range(b, min(b + batch, len(df)))]
        p = ("Translate these English documentary narration lines into natural spoken Tamil (not literal), same meaning "
             "and emotion. Each Tamil line must have at most max_words words so it fits its time slot. Keep '...' pauses. "
             'Return JSON array only: [{"id":n,"tamil":"..."}].\n' + json.dumps(items, ensure_ascii=False))
        j = call(model, {"contents": [{"parts": [{"text": p}]}], "generationConfig": {"responseMimeType": "application/json"}}, key, note)
        txt = j["candidates"][0]["content"]["parts"][0]["text"].replace("```json", "").replace("```", "")
        for a in json.loads(txt):
            df.at[df.index[int(a["id"])], "tamil"] = a["tamil"]
        bar.progress(min((b + batch) / len(df), 1.0), text="Translating...")
    return df


def synth(text, voice, rate, pitch, wd):
    mp3, raw = os.path.join(wd, "e.mp3"), os.path.join(wd, "e.raw")
    last = None
    for _ in range(3):
        try:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(edge_tts.Communicate(text, voice, rate=f"{int(rate):+d}%", pitch=f"{int(pitch):+d}Hz").save(mp3))
            finally:
                loop.close()
            sh("ffmpeg", "-y", "-i", mp3, "-f", "s16le", "-ac", "1", "-ar", str(SR), raw)
            return np.frombuffer(open(raw, "rb").read(), dtype=np.int16)
        except Exception as e:
            last = e
            time.sleep(2)
    raise RuntimeError(f"edge-tts failed: {last}")


def trim(pcm, thr=350, keep=int(0.03 * SR)):
    idx = np.where(np.abs(pcm.astype(np.int32)) > thr)[0]
    return pcm if len(idx) == 0 else pcm[max(idx[0] - keep, 0): idx[-1] + keep]


def atempo(pcm, rate, wd):
    a, b = os.path.join(wd, "t_in.raw"), os.path.join(wd, "t_out.raw")
    open(a, "wb").write(pcm.tobytes())
    sh("ffmpeg", "-y", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", a, "-filter:a", f"atempo={rate:.4f}", "-f", "s16le", b)
    return np.frombuffer(open(b, "rb").read(), dtype=np.int16)


def wav_bytes(pcm):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
    return buf.getvalue()


# ------------------------------------------------------------------ UI
st.title("🎙️ SRT → Tamil Dub")
st.caption("English SRT (or Tamil SRT) upload pannunga. Ovvoru line-um adhoda timestamp-la exact-a Tamil voice-la set aagum. Free, key thevai illa.")
ss = st.session_state
for k, v in dict(df=None, dir=tempfile.mkdtemp(), video=None, vsig=None, vdur=0.0, wav=None, mp3=None, mp4=None,
                 report=None, clips={}, ssig=None).items():
    ss.setdefault(k, v)

with st.sidebar:
    st.header("Voice")
    voice = VOICES[st.selectbox("Voice", list(VOICES))]
    base_rate = st.slider("Base speed", -30, 30, 0, 5, format="%d%%")
    base_pitch = st.slider("Pitch", -30, 30, 0, 5, format="%dHz")
    st.caption("Ovvoru row-layum 'rate' and 'pitch' column-la thani-yaa maathalam (emphasis-kku).")
    st.header("Timing")
    max_extra = st.slider("Max auto speed-up", 0, 60, 35, 5, format="%d%%",
                          help="Line neelama iruntha ivvalavu varai thaanaave vegamaakum.")
    strict = st.checkbox("Strict fit (never overlap)", value=True)
    st.header("Gemini (optional)")
    try:
        dk = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        dk = ""
    key = st.text_input("Gemini API key", value=dk, type="password", help="English SRT-a Tamil-aakka mattum. Tamil SRT irundha thevai illa.")
    text_model = st.text_input("Text model", "gemini-3.8-flash")
    if st.button("▶ Preview voice"):
        try:
            p = trim(synth(SAMPLE, voice, base_rate, base_pitch, ss["dir"]))
            st.audio(wav_bytes(p), format="audio/wav")
        except Exception as e:
            st.error(str(e))

st.subheader("1️⃣ Upload")
srt = st.file_uploader("SRT file", type=["srt"])
kind = st.radio("SRT language", ["English (translate to Tamil)", "Already Tamil"], horizontal=True)
merge = st.checkbox("Join short captions into full sentences (English SRT)", value=True)
vid = st.file_uploader("Video (optional: MP4 output + exact duration)", type=["mp4", "mov", "mkv", "webm"])

if vid is not None and ss["vsig"] != f"{vid.name}-{vid.size}":
    pth = os.path.join(ss["dir"], "in_" + vid.name.replace(" ", "_"))
    open(pth, "wb").write(vid.getbuffer())
    ss.update(vsig=f"{vid.name}-{vid.size}", video=pth, vdur=duration(pth), wav=None, mp3=None, mp4=None)

if srt is not None:
    sig = f"{srt.name}-{srt.size}-{kind}-{merge}"
    if ss["ssig"] != sig:
        rows = parse_srt(srt.getvalue().decode("utf-8", errors="ignore"))
        if kind.startswith("English") and merge:
            rows = merge_sentences(rows)
        is_ta = kind.startswith("Already")
        ss["df"] = pd.DataFrame({"start": [round(r[0], 2) for r in rows], "end": [round(r[1], 2) for r in rows],
                                 "english": ["" if is_ta else r[2] for r in rows],
                                 "tamil": [r[2] if is_ta else "" for r in rows],
                                 "rate": 0, "pitch": 0})
        ss.update(ssig=sig, clips={}, wav=None, mp3=None, mp4=None, report=None)

if ss["df"] is not None:
    df = ss["df"]
    total = ss["vdur"] or (float(df["end"].max()) + 1.0)
    st.caption(f"{len(df)} lines · timeline {total:.1f}s")

    if kind.startswith("English"):
        if st.button("2️⃣ Translate to Tamil (Gemini)", disabled=not key):
            bar, note = st.progress(0.0), st.empty()
            try:
                ss["df"] = translate(df.copy(), total, key, text_model, bar, note)
                note.success("Translated.")
                st.rerun()
            except Exception as e:
                st.error(str(e))
        if not key:
            st.info("Gemini key illa-na: Tamil script-a vera edathula (Claude) ezhudhi, 'Already Tamil' SRT-a upload pannunga.")

    st.subheader("Review / edit")
    ss["df"] = st.data_editor(ss["df"], use_container_width=True, hide_index=True, num_rows="fixed",
                              disabled=["start", "end", "english"], key="ed",
                              column_config={"rate": st.column_config.NumberColumn("rate %", min_value=-50, max_value=80, step=5),
                                             "pitch": st.column_config.NumberColumn("pitch Hz", min_value=-50, max_value=50, step=5)})

    if st.button("3️⃣ Generate Tamil audio", type="primary"):
        df, wd = ss["df"].copy(), ss["dir"]
        if (df["tamil"].astype(str).str.strip() == "").any():
            st.error("Sila lines-la Tamil text illa. Mudhal-la translate pannunga.")
            st.stop()
        out = np.zeros(int(total * SR) + SR, dtype=np.int16)
        bar, report = st.progress(0.0), []
        try:
            for i in range(len(df)):
                g = df.iloc[i]
                nxt = float(df.iloc[i + 1]["start"]) if i + 1 < len(df) else total
                slot = nxt - float(g["start"]) - 0.10
                target = slot * 0.97
                r0, p0 = base_rate + int(g["rate"]), base_pitch + int(g["pitch"])
                text = str(g["tamil"])
                k = hashlib.md5(f"{text}|{voice}|{r0}|{p0}|{slot:.2f}|{max_extra}|{strict}".encode()).hexdigest()
                if k not in ss["clips"]:
                    clip = trim(synth(text, voice, r0, p0, wd))
                    d = len(clip) / SR
                    if d > target and max_extra > 0:            # 1) native TTS speed-up (sounds natural)
                        need = (d / target - 1) * 100
                        extra = min(need + 3, max_extra)
                        clip = trim(synth(text, voice, int(round((1 + r0 / 100) * (1 + extra / 100) * 100 - 100)), p0, wd))
                    d = len(clip) / SR
                    if d > target:                              # 2) small extra tempo change
                        clip = trim(atempo(clip, min(d / target, 1.25), wd))
                    d = len(clip) / SR
                    cut = False
                    if strict and d > slot:                     # 3) last resort: soft fade-cut
                        clip = clip[: int(slot * SR)].copy()
                        f = int(0.08 * SR)
                        clip[-f:] = (clip[-f:] * np.linspace(1, 0, f)).astype(np.int16)
                        cut = True
                    ss["clips"][k] = (clip, cut)
                clip, cut = ss["clips"][k]
                d = len(clip) / SR
                report.append({"#": i + 1, "start": float(g["start"]), "slot_s": round(slot, 2), "audio_s": round(d, 2),
                               "status": "✂️ cut" if cut else ("⚠️ overlap" if d > slot else "✅ fits")})
                o = int(float(g["start"]) * SR)
                out[o:o + len(clip)] = clip[: len(out) - o]
                bar.progress((i + 1) / len(df), text=f"Voice {i + 1}/{len(df)}")
            wp = os.path.join(wd, "dub.wav")
            ss["wav"] = wav_bytes(out[: int(total * SR)]); open(wp, "wb").write(ss["wav"])
            sh("ffmpeg", "-y", "-i", wp, "-b:a", "192k", os.path.join(wd, "dub.mp3"))
            ss["mp3"] = open(os.path.join(wd, "dub.mp3"), "rb").read()
            ss["mp4"] = None
            if ss["video"]:
                mp = os.path.join(wd, "video_tamil.mp4")
                sh("ffmpeg", "-y", "-i", ss["video"], "-i", wp, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                   "-c:a", "aac", "-b:a", "160k", "-shortest", mp)
                ss["mp4"] = open(mp, "rb").read()
            ss["report"] = pd.DataFrame(report)
            st.rerun()
        except Exception as e:
            st.error(str(e))

if ss["wav"]:
    st.subheader("4️⃣ Result")
    r = ss["report"]
    bad = int((r["status"] != "✅ fits").sum())
    if bad == 0:
        st.success(f"Ellaa {len(r)} lines-um exact slot-kulle fit. Audio length = {total:.1f}s")
    else:
        st.warning(f"{bad} line(s) tight. Table-la ✂️/⚠️ lines-la Tamil kurachi marubadi Generate pannunga (maathuna lines mattum regenerate aagum).")
    st.audio(ss["wav"], format="audio/wav")
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Audio WAV", ss["wav"], "tamil_dub.wav", "audio/wav")
    c2.download_button("⬇️ Audio MP3", ss["mp3"], "tamil_dub.mp3", "audio/mpeg")
    if ss["mp4"]:
        st.download_button("⬇️ Video + Tamil audio", ss["mp4"], "video_tamil.mp4", "video/mp4")
    with st.expander("Timing report"):
        st.dataframe(r, use_container_width=True, hide_index=True)
    st.caption("CapCut: video import → original audio mute → indha audio-va 0:00-la vainga.")
