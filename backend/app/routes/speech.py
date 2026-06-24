import os
import uuid
import tempfile
import re

import numpy as np
import librosa
import soundfile as sf
import whisper
import torch

from app.services.phoneme.data import PHONEME_DATA
from fastapi import APIRouter, UploadFile, File, Form, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session as DbSession
from rapidfuzz import fuzz
from silero_vad import get_speech_timestamps, load_silero_vad
from g2p_en import G2p
from faster_whisper import WhisperModel

from app.services.phoneme.scoring import score_phonemes
from app.database import SessionLocal
from app.models.patient import Patient
from app.models.session import Session as TherapySession
g2p = G2p()

router = APIRouter(prefix="/speech", tags=["Speech Therapy"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------

try:
    whisper_model = WhisperModel(
        "base.en",
        device="cpu",
        compute_type="int8"
    )

    print("✅ Faster-Whisper loaded")

except Exception as e:
    whisper_model = None
    print("❌ Faster-Whisper error:", e)

try:
    vad_model = load_silero_vad()
    print("✅ Silero VAD loaded")
except Exception as e:
    vad_model = None
    print("❌ VAD error:", e)


# ---------------------------------------------------
# SAVE AUDIO
# ---------------------------------------------------
# def save_audio(file: UploadFile):
#     path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.wav")
#     with open(path, "wb") as f:
#         f.write(file.file.read())
#     return path
def save_audio(file: UploadFile):

    print("Filename:", file.filename)
    print("Content-Type:", file.content_type)

    ext = os.path.splitext(file.filename)[1]

    if not ext:
        ext = ".wav"

    path = os.path.join(
        tempfile.gettempdir(),
        f"{uuid.uuid4()}{ext}"
    )

    with open(path, "wb") as f:
        f.write(file.file.read())

    print("Saved as:", path)

    return path

# ---------------------------------------------------
# NORMALIZE TEXT
# ---------------------------------------------------
def normalize_text(text):
    return re.sub(r'[^a-z ]', '', text.lower()).strip()


WORD_MAPPER_DISPLAY_MAP = {
    "AA": "a",
    "AE": "Ai",
    "AH": "u",
    "AO": "aw",
    "AW": "au",
    "AY": "Ai",
    "B": "b",
    "CH": "ch",
    "D": "d",
    "DH": "th",
    "EH": "e",
    "ER": "er",
    "EY": "ay",
    "F": "f",
    "G": "g",
    "HH": "h",
    "IH": "i",
    "IY": "ee",
    "JH": "j",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ng",
    "OW": "o",
    "OY": "oy",
    "P": "p",
    "R": "r",
    "SH": "sh",
    "S": "s",
    "T": "t",
    "TH": "th",
    "UH": "oo",
    "UW": "oo",
    "V": "v",
    "W": "w",
    "Y": "y",
    "Z": "z",
    "ZH": "zh",
}

COMMON_WORD_MAPPER_OVERRIDES = {
    "apple": ["Ai", "p", "p", "l"],
    "ball": ["b", "aw", "l"],
    "teacher": ["t", "ee", "ch", "er"],
    "umbrella": ["u", "m", "b", "r", "e", "l", "u"],
    "cat": ["k", "a", "t"],
    "dog": ["d", "o", "g"],
    "sun": ["s", "u", "n"],
    "ship": ["sh", "i", "p"],
    "boat": ["b", "oa", "t"],
}


def get_display_phoneme_label(token):
    return WORD_MAPPER_DISPLAY_MAP.get(token.upper(), token.upper())


def get_word_mapper_display(word, tokens):
    normalized = word.lower().strip()

    if normalized in COMMON_WORD_MAPPER_OVERRIDES:
        return COMMON_WORD_MAPPER_OVERRIDES[normalized]

    labels = []
    i = 0

    while i < len(tokens):
        token = tokens[i].upper()

        if token == "AH" and i + 1 < len(tokens) and tokens[i + 1].upper() == "L":
            labels.append("ul")
            i += 2
            continue

        if token == "AH" and i + 1 < len(tokens) and tokens[i + 1].upper() == "R":
            labels.append("ur")
            i += 2
            continue

        if token == "AE" and i + 1 < len(tokens) and tokens[i + 1].upper() == "L":
            labels.append("al")
            i += 2
            continue

        if token in {"EY", "AY", "AI"}:
            labels.append("Ai")
            i += 1
            continue

        if token in {"OW", "OE", "OH"}:
            labels.append("o")
            i += 1
            continue

        if token in {"IY", "EE"}:
            labels.append("ee")
            i += 1
            continue

        if token in {"UW", "UH", "OO"}:
            labels.append("oo")
            i += 1
            continue

        labels.append(get_display_phoneme_label(token))
        i += 1

    return labels


def get_display_phoneme_list(tokens, word=None):
    if word is not None:
        return get_word_mapper_display(word, tokens)
    return [get_display_phoneme_label(t) for t in tokens]


# ---------------------------------------------------
# VAD SPLIT (Silero)
# ---------------------------------------------------
def vad_split(y, sr):
    if vad_model is None:
        return [y]

    y_tensor = torch.tensor(y)

    timestamps = get_speech_timestamps(
        y_tensor,
        vad_model,
        sampling_rate=sr
    )

    segments = []

    for seg in timestamps:
        start = seg["start"]
        end = seg["end"]
        segments.append(y[start:end])

    return segments if segments else [y]


# ---------------------------------------------------
# SMART CHILD SEGMENT SELECTION
# ---------------------------------------------------
def select_child_segment(y, sr):

    segments = vad_split(y, sr)

    best_audio = None
    best_score = -999

    for audio in segments:

        if len(audio) < 1600:
            continue

        # energy
        energy = np.mean(librosa.feature.rms(y=audio))

        # pitch
        pitches, _ = librosa.piptrack(y=audio, sr=sr)
        pitch_vals = pitches[pitches > 0]
        pitch = np.mean(pitch_vals) if len(pitch_vals) else 0

        duration = len(audio) / sr

        # 🎯 CHILD HEURISTIC
        # Prefer strong, voiced segments and avoid tiny noise bursts.
        score = (energy * 100) + (pitch * 0.35) - (duration * 0.5)

        if score > best_score:
            best_score = score
            best_audio = audio

    if best_audio is None:
        return y

    print(f"🎯 Selected segment score: {best_score:.2f}")

    return best_audio


# ---------------------------------------------------
# TRANSCRIPTION
# ---------------------------------------------------
def transcribe(y, sr, prompt=None):
    if whisper_model is None:
        return ""

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        sf.write(tmp_path, y, sr)
        print("Transcribing file:", tmp_path)
        segments, info = whisper_model.transcribe(
            tmp_path,
            language="en",
            beam_size=5
        )

        text = " ".join(
            segment.text
            for segment in segments
        )
        return normalize_text(text)
    except Exception as e:
        print("transcription error:", e)
        return ""
    finally:
        # if os.path.exists(tmp_path):
            # os.remove(tmp_path)
        pass
# ---------------------------------------------------
# BASIC PHONEME
# ---------------------------------------------------

def get_basic_phonemes(word):

    phonemes = g2p(word)

    clean = []

    for p in phonemes:

        p = p.replace("0", "").replace("1", "").replace("2", "")

        if p.isalpha():
            clean.append(p)

    return clean

# ---------------------------------------------------
# compare phoneme
# ---------------------------------------------------
def compare_phonemes(expected, spoken):

    matches = []

    correct = 0

    total = max(len(expected), 1)

    for i in range(len(expected)):

        exp = expected[i]

        got = spoken[i] if i < len(spoken) else None

        is_correct = exp == got

        if is_correct:
            correct += 1

        matches.append({
            "expected": exp,
            "detected": got,
            "correct": is_correct
        })

    accuracy = int((correct / total) * 100)

    return {
        "matches": matches,
        "accuracy": accuracy
    }

# ---------------------------------------------------
# BEST WORD MATCH
# ---------------------------------------------------
def extract_spoken_part(transcript, target):

    words = transcript.split()
    if not words:
        return ""

    target = target.lower()

    best_match = ""
    best_score = 0

    for word in words:
        # Compare progressively smaller prefixes
        for i in range(1, len(word) + 1):
            part = word[:i]

            score = fuzz.ratio(part, target[:len(part)])

            if score > best_score:
                best_score = score
                best_match = part

    # 🔥 HARD RULE: do NOT exceed target length
    if len(best_match) > len(target):
        best_match = best_match[:len(target)]

    return best_match
# ---------------------------------------------------
# SCORE
# ---------------------------------------------------
def compute_score(target, spoken):

    if spoken == "":
        return 0

    target = target.lower()
    spoken = spoken.lower()

    # Base similarity
    similarity = fuzz.ratio(target, spoken)

    # Length penalty (VERY IMPORTANT)
    length_ratio = len(spoken) / len(target)

    # Penalize short pronunciations
    if length_ratio < 0.5:
        return int(similarity * length_ratio)

    # Medium partial
    if length_ratio < 0.8:
        return int(similarity * 0.8)

    # Full word attempt
    return int(similarity)


# ---------------------------------------------------
# FEEDBACK
# ---------------------------------------------------
def generate_feedback(score, target, spoken):
    if spoken == "":
        return "No speech detected. Try again slowly.", 1

    if score >= 90:
        return "Excellent pronunciation! 🎉", 5

    if score >= 70:
        return f"Very good! Improve ending of '{target}'.", 4

    if score >= 50:
        return f"You said '{spoken}'. Try full word '{target}'.", 3

    return f"Break it: {target[:2]}...{target}", 2


# ---------------------------------------------------
# AUDIO FEATURES
# ---------------------------------------------------
def extract_features(y, sr):
    y = librosa.util.normalize(y)
    y, _ = librosa.effects.trim(y)

    duration = librosa.get_duration(y=y, sr=sr)
    loudness = float(np.mean(librosa.feature.rms(y=y)))

    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitches[pitches > 0]
    pitch = float(np.mean(pitch_vals)) if len(pitch_vals) else 0

    return {
        "duration": round(duration, 2),
        "loudness": round(loudness, 4),
        "pitch": round(pitch, 2)
    }


# ---------------------------------------------------
# MAIN API
def extract_first_sound(text):

    if text == "":
        return ""

    words = text.split()

    if len(words) == 0:
        return ""

    first_word = words[0]

    return first_word[0]

@router.post("/therapy")
async def therapy(
    file: UploadFile = File(...),
    patient_name: str = Form(...),
    patient_age: int | None = Form(None),
    target_word: str = Form(...),
    therapy_mode: str = Form(...),
    db: DbSession = Depends(get_db)
):
    try:

        # -------------------
        # SAVE AUDIO
        # -------------------
        path = save_audio(file)

        try:
            info = sf.info(path)
            print("AUDIO INFO:", info)
        except Exception as e:
            print("SOUNDFILE ERROR:", e)

        y, sr = librosa.load(
            path,
            sr=16000
        )

        # normalize audio
        y = librosa.util.normalize(y)

        # trim silence
        y, _ = librosa.effects.trim(
            y,
            top_db=10
        )

        # -------------------
        # LIGHTWEIGHT CHILD SELECTION
        # -------------------
        y_child = select_child_segment(y, sr)

        print("Full audio duration:", len(y) / sr)
        print("Selected audio duration:", len(y_child) / sr)
        # fallback safety
        if y_child is None or len(y_child) < 300:
            print("⚠ Using full audio fallback")
            y_child = y

        # -------------------
        # TRANSCRIPTION
        # -------------------
        transcript_child = transcribe(
            y_child,
            sr,
            prompt=target_word
        )

        transcript_full = transcribe(
            y,
            sr,
            prompt=target_word
        )

        target = normalize_text(
            target_word
        )

        def match_score(text):
            if therapy_mode == "First Letter Match":
                return fuzz.ratio(
                    extract_first_sound(text),
                    extract_first_sound(target)
                )
            return fuzz.ratio(text, target)

        score_child = match_score(transcript_child)
        score_full = match_score(transcript_full)

        if transcript_child == "":
            transcript = transcript_full
            print("⚠ Child segment transcript empty → using full audio transcript")
        elif transcript_full == "":
            transcript = transcript_child
            print("⚠ Full audio transcript empty → using child segment transcript")
        elif score_full > score_child + 5:
            transcript = transcript_full
            print("⚠ Using full audio transcript as better match:", score_full, "vs", score_child)
        else:
            transcript = transcript_child
            print("✅ Using selected child segment transcript:", score_child, "vs", score_full)

        print("TRANSCRIPT:", transcript)
        # -------------------
        # TEXT NORMALIZATION
        # -------------------
        target = target

        if therapy_mode == "First Letter Match":

            spoken = extract_first_sound(
                transcript
            )

        else:

            spoken = transcript

        spoken = normalize_text(
            spoken
        )

        # -------------------
        # PHONEME ANALYSIS
        # -------------------

        expected_phonemes = get_basic_phonemes(
            target
        )

        spoken_phonemes = get_basic_phonemes(
            spoken
        )

        phoneme_result = score_phonemes(
            expected_phonemes,
            spoken_phonemes
        )
        
        # -------------------
        # SCORING
        # -------------------
        score = compute_score(
            target,
            spoken
        )

        # -------------------
        # FEEDBACK
        # -------------------
        feedback, stars = generate_feedback(
            score,
            target,
            spoken
        )

        # -------------------
        # FEATURES
        # -------------------
        metrics = extract_features(
            y_child,
            sr
        )

        # -------------------
        # RESPONSE
        # -------------------
        expected_display = get_display_phoneme_list(
            expected_phonemes,
            word=target_word
        )
        spoken_display = get_display_phoneme_list(
            spoken_phonemes,
            word=spoken
        )

        result = {

            "child_name": patient_name,

            "target_word": target_word,

            "spoken_word": (
                spoken
                if spoken
                else "No speech detected"
            ),

            "full_transcript": transcript,

            "accuracy": score,

            "phoneme_accuracy": phoneme_result["accuracy"],

            "phoneme_matches": phoneme_result["matches"],

            "expected_phonemes": expected_phonemes,

            "spoken_phonemes": spoken_phonemes,

            "expected_phonemes_display": expected_display,

            "spoken_phonemes_display": spoken_display,

            "duration": metrics["duration"],

            "loudness": metrics["loudness"],

            "pitch": metrics["pitch"],

            "feedback": feedback,

            "stars": stars
        }

        normalized_patient_name = (patient_name or "Child").strip() or "Child"
        patient = db.query(Patient).filter(
            func.lower(Patient.name) == normalized_patient_name.lower()
        ).first()

        if patient is None:
            patient = Patient(
                name=normalized_patient_name,
                age=patient_age,
                language="English"
            )
            db.add(patient)
            db.flush()
        elif patient_age is not None:
            patient.age = patient_age

        session = TherapySession(
            patient_id=patient.id,
            target_word=target_word,
            spoken_word=result["spoken_word"],
            accuracy=score,
            feedback=feedback,
            stars=stars,
            session_type=therapy_mode,
            audio_file=file.filename
        )
        db.add(session)
        db.commit()
        db.refresh(patient)
        db.refresh(session)

        progress = db.query(
            func.count(TherapySession.id),
            func.avg(TherapySession.accuracy)
        ).filter(TherapySession.patient_id == patient.id).one()

        result["patient_id"] = str(patient.id)
        result["session_id"] = str(session.id)
        result["child_age"] = patient.age
        result["total_sessions"] = progress[0] or 0
        result["average_accuracy"] = round(float(progress[1] or 0), 1)

        # -------------------
        # CLEANUP
        # -------------------
        if os.path.exists(path):
            os.remove(path)

        return result

    except Exception as e:

        print("THERAPY ERROR:", e)

        return {
            "error": str(e)
        }
    
@router.post("/phonemes/preview")
async def preview_phonemes(
    word: str = Form(...)
):

    return {
        "success": True,
        "data": {
            "phonemes": get_basic_phonemes(word)
        }
    }
    
@router.post("/phonemes")
async def get_phonemes(word: str = Form(...)):
    return {
        "success": True,
        "data": {
            "word": word,
            "phonemes": get_basic_phonemes(word)
        }
    }

@router.post("/compare")
async def compare_word(
    file: UploadFile = File(...),
    target_word: str = Form(...)
):

    result = await therapy(
        file=file,
        patient_name="Demo",
        target_word=target_word,
        therapy_mode="Full Word Match"
    )

    return {
        "success": True,
        "data": {
            "target_word": result["target_word"],
            "transcript": result["spoken_word"],
            "accuracy": result["accuracy"],
            "feedback": result["feedback"],
            "expected_phonemes": result["expected_phonemes"],
            "spoken_phonemes": result["spoken_phonemes"],
            "matches": result["phoneme_matches"]
        }
    }

@router.post("/compare_phoneme")
async def compare_phoneme(
    file: UploadFile = File(...),
    target_phoneme: str = Form(...)
):
    try:

        # Save audio
        path = save_audio(file)

        # Load audio
        y, sr = librosa.load(
            path,
            sr=16000
        )

        y = librosa.util.normalize(y)

        y, _ = librosa.effects.trim(
            y,
            top_db=10
        )

        # Use same child-segment logic
        y_child = select_child_segment(y, sr)

        # Transcribe
        spoken_word = transcribe(
            y_child,
            sr
        )

        # Convert transcript → phonemes
        spoken_phonemes = get_basic_phonemes(
            spoken_word
        )

        correct = (
            target_phoneme in spoken_phonemes
        )

        # cleanup
        if os.path.exists(path):
            os.remove(path)

        return {
            "success": True,
            "data": {
                "correct": correct,
                "transcript": spoken_word,
                "detected_phonemes": spoken_phonemes,
                "feedback": (
                    "Great Job!"
                    if correct
                    else f"Try emphasizing /{target_phoneme}/"
                )
            }
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }

