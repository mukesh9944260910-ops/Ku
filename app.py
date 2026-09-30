import base64, hashlib, io, json, os, subprocess, tempfile, time, wave

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="English → Tamil Video Dubber", page_icon="🎙️", layout="centered")

API = "https://generativelanguage.googleapis.com/v1beta/models/"
SR = 24000
CHUNK = 120
VOICES = {
    "Charon (deep, calm)": "Charon", "Orus (firm, mature)": "Orus", "Fenrir (excited)": "Fenrir",
    "Algenib (gravelly)": "Algenib", "Alnilam (firm)": "Alnilam", "Iapetus (clear)": "Iapetus",
    "Umbriel (easy-going)": "Umbriel", "Rasalgethi (informative)": "Rasalgethi", "Sadaltager (knowledgeable)": "Sadaltager",
    "Gacrux (mature)": "Gacrux", "Enceladus (breathy)": "Enceladus", "Schedar (even)": "Schedar",
    "Puck (upbeat)": "Puck", "Achird (friendly)": "Achird", "Zubenelgenubi (casual)": "Zubenelgenubi",
    "Kore (firm, female)": "Kore", "Zephyr (bright, female)": "Zephyr", "Leda (youthful, female)": "Leda",
    "Aoede (breezy, female)": "Aoede", "Sulafat (warm, female)": "Sulafat", "Vindemiatrix (gentle, female)": "Vindemiatrix",
    "Achernar (soft, female)": "Achernar",
}
PRESETS = {
    "Calm documentary": "calm, warm, curious documentary narrator, steady pace",
    "Dramatic storyteller": "dramatic, suspenseful storyteller, expressive, strong emphasis on key words",
    "Energetic explainer": "energetic, upbeat YouTube explainer, quick and lively",
    "Serious / news": "serious, authoritative, neutral news-style narrator",
    "Friendly & casual": "friendly, casual, conversational, smiling tone",
    "Copy original only": "",
}
SAMPLE = "மனித வரலாற்றின் பெரும்பகுதியில்... ஒரு இடத்திலிருந்து இன்னொரு இடத்திற்குச் செல்வது ஒரு சவாலாக இருந்தது."


# ------------------------------------------------------------------ helpers
def sh(*a):
    subprocess.run(a, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def duration(path):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                          "-of", "default=nw=1:nk=1", path]))


def call(model, body, key, note=None, tries=5):
    for i in range(tries):
        r = requests.post(f"{API}{model}:generateContent",
                          headers={"x-goog-api-key": key, "Content-Type": "application/json"}, json=body, timeout=300)
        if r.ok:
            return r.json()
        if r.status_code == 429 or r.status_code >= 500:
            wait = 20 * (i + 1)
            if note is not None:
                note.info(f"API busy ({r.status_code}). {wait}s-la retry pannum...")
            time.sleep(wait)
            continue
        raise RuntimeError(f"API error {r.status_code}: {r.text[:300]}")
    raise RuntimeError("API failed after retries")


def prompt_for(visuals):
    return (
        "You are dubbing an English narrated video into Tamil. "
        + ("You get the video (audio + visuals). Use what is on screen so the Tamil fits the visuals. " if visuals
           else "You get the narrator audio. ")
        + "Transcribe the English narration and write the Tamil dub script. Return a JSON array only. "
        'Each item = one sentence or short phrase: {"start":sec,"end":sec,"english":"...","tamil":"...","style":"..."}. '
        "start/end = seconds from the beginning of THIS clip, accurate to 0.3s, start exactly when the narrator starts speaking. "
        '"tamil" = natural spoken documentary Tamil (not literal), same meaning and emphasis, and short enough to be spoken '
        "within (end-start) seconds: at most (end-start)*2.0 words. Keep '...' where the narrator pauses. "
        '"style" = short delivery direction copying the original narrator tone (emotion, pace, energy, emphasis words), '
        "e.g. \"calm, curious, slow\". Skip music-only parts."
    )


def analyze(video, dur, key, text_model, visuals, bar, note):
    segs, n, wd = [], int(np.ceil(dur / CHUNK)), os.path.dirname(video)
    for c in range(n):
        off = c * CHUNK
        if visuals:
            f, mime = os.path.join(wd, f"c{c}.mp4"), "video/mp4"
            sh("ffmpeg", "-y", "-ss", str(off), "-t", str(CHUNK), "-i", video, "-vf", "scale=-2:360,fps=2",
               "-c:v", "libx264", "-crf", "34", "-preset", "veryfast", "-c:a", "aac", "-ac", "1", "-ar", "16000", "-b:a", "48k", f)
        else:
            f, mime = os.path.join(wd, f"c{c}.wav"), "audio/wav"
            sh("ffmpeg", "-y", "-ss", str(off), "-t", str(CHUNK), "-i", video, "-vn", "-ac", "1", "-ar", "16000", f)
        data = base64.b64encode(open(f, "rb").read()).decode()
        j = call(text_model, {"contents": [{"parts": [{"inlineData": {"mimeType": mime, "data": data}},
                                                       {"text": prompt_for(visuals)}]}],
                              "generationConfig": {"responseMimeType": "application/json"}}, key, note)
        txt = j["candidates"][0]["content"]["parts"][0]["text"].replace("```json", "").replace("```", "")
        for a in json.loads(txt):
            s = float(a["start"]) + off
            if segs and s < segs[-1]["start"] + 0.2:
                continue
            segs.append({"start": round(s, 2), "end": round(float(a["end"]) + off, 2), "english": a["english"],
                         "tamil": a["tamil"], "style": a.get("style", "calm")})
        os.remove(f)
        bar.progress((c + 1) / n, text=f"Analyzing {c + 1}/{n}")
    return segs


def tts(text, direction, key, voice, tts_model, note):
    j = call(tts_model, {
        "contents": [{"parts": [{"text": f"Say the following in Tamil. Voice direction: {direction}. "
                                         f"Pause naturally at ellipses. Sound human, never robotic:\n{text}"}]}],
        "generationConfig": {"responseModalities": ["AUDIO"],
                             "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}}}, key, note)
    pcm = base64.b64decode(j["candidates"][0]["content"]["parts"][0]["inlineData"]["data"])
    return np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype=np.int16)


def trim(pcm, thr=350, keep=int(0.03 * SR)):
    """Remove leading/trailing silence so speech starts exactly at the segment start."""
    idx = np.where(np.abs(pcm.astype(np.int32)) > thr)[0]
    if len(idx) == 0:
        return pcm
    return pcm[max(idx[0] - keep, 0): idx[-1] + keep]


def shorten(text, target, key, text_model, note):
    j = call(text_model, {"contents": [{"parts": [{"text":
             f"Shorten this spoken Tamil narration so it takes at most {target:.1f} seconds to say "
             f"(max {max(int(target * 2.0), 2)} words). Keep the core meaning and tone. Return only the Tamil text.\n{text}"}]}]},
             key, note)
    return j["candidates"][0]["content"]["parts"][0]["text"].strip()


def speed(pcm, rate, wd):
    a, b = os.path.join(wd, "t_in.raw"), os.path.join(wd, "t_out.raw")
    open(a, "wb").write(pcm.tobytes())
    filt = ",".join(["atempo=2.0"] * int(rate // 2 > 0 and rate >= 2) + [f"atempo={rate / (2 if rate >= 2 else 1):.4f}"])
    sh("ffmpeg", "-y", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", a, "-filter:a", filt, "-f", "s16le", b)
    return np.frombuffer(open(b, "rb").read(), dtype=np.int16)


def wav_bytes(pcm):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
    return buf.getvalue()


# ------------------------------------------------------------------ UI
st.title("🎙️ English → Tamil Video Dubber")
st.caption("Video upload pannunga → English kettu Tamil script → ovvoru segment-um video timeline-ku exact-a set aagum. "
           "Video-va touch pannala. Audio mattum download pannikalam.")

ss = st.session_state
for k, v in dict(segs=None, clips={}, dir=tempfile.mkdtemp(), video=None, dur=0.0, wav=None, mp3=None, mp4=None,
                 report=None, sig=None).items():
    ss.setdefault(k, v)

with st.sidebar:
    st.header("Settings")
    try:
        default_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        default_key = ""
    key = st.text_input("Gemini API key", value=default_key, type="password")
    st.subheader("Voice")
    voice_label = st.selectbox("Voice", list(VOICES))
    voice = VOICES[voice_label]
    preset = st.selectbox("Style preset", list(PRESETS))
    custom = st.text_area("Extra style instruction (optional)",
                          placeholder="e.g. slightly slower, deep voice, emphasise numbers and years")
    copy_tone = st.checkbox("Copy original narrator's tone per segment", value=True,
                            help="Analyze step-la Gemini original narrator-oda emotion/pace-a 'style' column-la eluthum.")
    st.subheader("Timing")
    strict = st.checkbox("Strict fit (never overlap next segment)", value=True,
                         help="Neelama iruntha: mudhal text shorten, apparam speed-up, kadaisiya soft fade-cut.")
    max_rate = st.slider("Max speed-up", 1.0, 1.6, 1.3, 0.05,
                         help="1.3x varai natural-a irukkum. Athukku mela voice vegama theriyum.")
    visuals = st.checkbox("Use video visuals while scripting", value=True)
    make_mp4 = st.checkbox("Also make MP4 (video + Tamil audio)", value=False)
    text_model = st.text_input("Text model", "gemini-2.5-flash")
    tts_model = st.text_input("TTS model", "gemini-2.5-flash-preview-tts")


def direction(seg_style=""):
    parts = [PRESETS[preset]]
    if copy_tone and seg_style:
        parts.append(f"match the original narrator: {seg_style}")
    if custom.strip():
        parts.append(custom.strip())
    return "; ".join(p for p in parts if p) or "natural documentary narrator"


with st.sidebar:
    if st.button("▶ Preview voice", disabled=not key):
        try:
            pcm = trim(tts(SAMPLE, direction("calm, curious"), key, voice, tts_model, st.empty()))
            st.audio(wav_bytes(pcm), format="audio/wav")
        except Exception as e:
            st.error(str(e))

up = st.file_uploader("Video (mp4 / mov / mkv / webm)", type=["mp4", "mov", "mkv", "webm"])
if up is not None:
    sig = f"{up.name}-{up.size}"
    if ss["sig"] != sig:
        path = os.path.join(ss["dir"], "input_" + up.name.replace(" ", "_"))
        with open(path, "wb") as f:
            f.write(up.getbuffer())
        ss.update(sig=sig, video=path, dur=duration(path), segs=None, clips={}, wav=None, mp3=None, mp4=None, report=None)
    st.video(up)
    st.caption(f"Duration: {ss['dur']:.1f}s")
    if not key:
        st.warning("Sidebar-la API key podunga.")
    if st.button("1️⃣ Analyze & translate", type="primary", disabled=not key):
        bar, note = st.progress(0.0), st.empty()
        try:
            segs = analyze(ss["video"], ss["dur"], key, text_model, visuals, bar, note)
            ss.update(segs=pd.DataFrame(segs), clips={}, wav=None, mp3=None, mp4=None, report=None)
            note.success(f"{len(segs)} segments ready.")
        except Exception as e:
            st.error(str(e))

if ss["segs"] is not None:
    st.subheader("2️⃣ Review")
    st.caption("Tamil and style columns edit pannalam. Marubadi Generate pannum pothu maathuna lines mattum regenerate aagum.")
    ss["segs"] = st.data_editor(ss["segs"], use_container_width=True, num_rows="fixed", hide_index=True,
                                disabled=["start", "end", "english"], key="editor")

    if st.button("3️⃣ Generate Tamil audio", type="primary", disabled=not key):
        df, dur, wd = ss["segs"].copy(), ss["dur"], ss["dir"]
        out = np.zeros(int(dur * SR) + SR, dtype=np.int16)
        bar, note, report = st.progress(0.0), st.empty(), []
        try:
            for i in range(len(df)):
                g = df.iloc[i]
                nxt = float(df.iloc[i + 1]["start"]) if i + 1 < len(df) else dur
                slot = nxt - float(g["start"]) - 0.10
                target = slot * 0.97
                d_txt = direction(str(g["style"]))
                k = hashlib.md5(f'{g["tamil"]}|{d_txt}|{voice}|{slot:.2f}|{max_rate}|{strict}'.encode()).hexdigest()
                if k not in ss["clips"]:
                    text = str(g["tamil"])
                    clip = trim(tts(text, d_txt, key, voice, tts_model, note))
                    for _ in range(3):                          # 1) shorten the text
                        if len(clip) / SR <= target:
                            break
                        text = shorten(text, target, key, text_model, note)
                        df.at[df.index[i], "tamil"] = text
                        clip = trim(tts(text, d_txt, key, voice, tts_model, note))
                    d = len(clip) / SR
                    if d > target:                              # 2) speed up (pitch preserved)
                        clip = trim(speed(clip, min(max_rate, d / target), wd))
                    d = len(clip) / SR
                    cut = False
                    if strict and d > slot:                     # 3) last resort: soft fade-cut, flagged
                        clip = clip[: int(slot * SR)].copy()
                        f = int(0.08 * SR)
                        clip[-f:] = (clip[-f:] * np.linspace(1, 0, f)).astype(np.int16)
                        cut = True
                    ss["clips"][k] = (clip, cut, text)
                clip, cut, text = ss["clips"][k]
                df.at[df.index[i], "tamil"] = text
                d = len(clip) / SR
                status = "✂️ cut (text kurainga)" if cut else ("⚠️ overlap" if d > slot else "✅ fits")
                report.append({"#": i + 1, "start": float(g["start"]), "slot_s": round(slot, 2),
                               "audio_s": round(d, 2), "status": status})
                o = int(float(g["start"]) * SR)
                out[o:o + len(clip)] = clip[: len(out) - o]
                bar.progress((i + 1) / len(df), text=f"Voice {i + 1}/{len(df)}")
            final = out[: int(dur * SR)]
            wd_wav = os.path.join(wd, "dub.wav")
            ss["wav"] = wav_bytes(final); open(wd_wav, "wb").write(ss["wav"])
            sh("ffmpeg", "-y", "-i", wd_wav, "-b:a", "192k", os.path.join(wd, "dub.mp3"))
            ss["mp3"] = open(os.path.join(wd, "dub.mp3"), "rb").read()
            if make_mp4:
                mp4p = os.path.join(wd, "video_tamil.mp4")
                sh("ffmpeg", "-y", "-i", ss["video"], "-i", wd_wav, "-map", "0:v", "-map", "1:a",
                   "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", mp4p)
                ss["mp4"] = open(mp4p, "rb").read()
            ss["segs"], ss["report"] = df, pd.DataFrame(report)
            note.success("Done!")
            st.rerun()
        except Exception as e:
            st.error(str(e))

if ss["wav"]:
    st.subheader("4️⃣ Result")
    r = ss["report"]
    bad = int((r["status"] != "✅ fits").sum())
    st.success(f"Audio length = video length ({ss['dur']:.1f}s). {len(r) - bad}/{len(r)} lines perfectly fit.") if bad == 0 \
        else st.warning(f"{bad} line(s) tight-a irukku. Table-la ✂️/⚠️ lines-la Tamil text-a kurachi marubadi Generate pannunga.")
    st.audio(ss["wav"], format="audio/wav")
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Audio WAV", ss["wav"], "tamil_dub.wav", "audio/wav")
    c2.download_button("⬇️ Audio MP3 (small)", ss["mp3"], "tamil_dub.mp3", "audio/mpeg")
    if ss["mp4"]:
        st.download_button("⬇️ Video + Tamil audio", ss["mp4"], "video_tamil.mp4", "video/mp4")
    with st.expander("Timing report"):
        st.dataframe(r, use_container_width=True, hide_index=True)
    st.caption("CapCut-la: video import → original audio mute → indha audio-va 0:00-la vainga. Full length same-a irukkum.")
