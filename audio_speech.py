"""CarePriority Gov - Speech & Audio Assistance Module
=====================================================
Provides:
1. Multilingual Voice Dictation (Web Speech API; te-IN, hi-IN, en-IN)
   - Transcribes clinician/patient speech into editable text for Chief Complaint
   - Clear fallback message when speech recognition is unsupported (Chrome/Edge required)
   - Safety rule: Dictation only pre-fills or assists; nurse always reviews and submits.

2. Read-Result-Aloud (speechSynthesis)
   - Audio announcement of triage tier, department, and key computed reasons
   - Strict privacy rule: NEVER reads patient names unless explicitly ticked by staff.
"""

import json
from typing import Dict, Any, Optional
import streamlit as st
import streamlit.components.v1 as components


def build_read_aloud_text(encounter: Dict[str, Any], include_name: bool = False) -> str:
    """Builds the text to be spoken by speech synthesis, strictly enforcing name privacy."""
    enc_id = encounter.get("encounter_id", "ENC-0000")
    short_token = enc_id.split("-")[-1] if "-" in enc_id else enc_id[-4:]
    token_display = f"T-{short_token}"

    dept = encounter.get("department", "General Medicine")
    tier_name = encounter.get("tier_name") or ("Needs Nurse Review" if encounter.get("abstained") else f"Level {encounter.get('tier', 3)}")
    reasons = encounter.get("reasons", [])
    primary_reason = reasons[0] if reasons else "Routine clinical assessment"

    if include_name:
        name = encounter.get("patient_name") or "Walk-in patient"
        speech_text = f"CarePriority Alert: Patient {name}, Token {token_display}. Priority: {tier_name}. Department: {dept}. Primary clinical reason: {primary_reason}."
    else:
        # STRICT PRIVACY DEFAULT: No patient name is spoken aloud in the waiting hall
        speech_text = f"CarePriority Alert: Token {token_display}. Priority: {tier_name}. Department: {dept}. Primary clinical reason: {primary_reason}."

    return speech_text


def render_voice_dictation_widget() -> None:
    """Renders the Web Speech API voice dictation widget with te-IN, hi-IN, en-IN support."""
    html_code = """
    <div id="dictation-wrapper" style="
      background: linear-gradient(135deg, rgba(8, 22, 42, 0.85) 0%, rgba(5, 14, 31, 0.95) 100%);
      border: 1px solid rgba(102, 227, 196, 0.35);
      border-radius: 14px;
      padding: 16px 20px;
      margin-bottom: 14px;
      font-family: 'Inter', system-ui, sans-serif;
      color: #F1F5F9;
      box-shadow: 0 4px 18px rgba(0, 0, 0, 0.4);
    ">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 10px;">
        <div style="display: flex; align-items: center; gap: 8px;">
          <span style="font-size: 1.25rem;">🎙️</span>
          <span style="font-family: 'Sora', sans-serif; font-weight: 700; font-size: 0.92rem; color: #66E3C4;">
            Rapid Voice Dictation (AI Speech-to-Text)
          </span>
        </div>
        <div style="display: flex; gap: 6px; align-items: center;">
          <span style="font-size: 0.78rem; color: #94A3B8;">Dialect:</span>
          <select id="dialect-select" style="
            background: #081528;
            color: #66E3C4;
            border: 1px solid rgba(102, 227, 196, 0.4);
            border-radius: 8px;
            padding: 4px 8px;
            font-size: 0.82rem;
            cursor: pointer;
            outline: none;
          ">
            <option value="te-IN">తెలుగు (Telugu - te-IN)</option>
            <option value="hi-IN">हिन्दी (Hindi - hi-IN)</option>
            <option value="en-IN" selected>English (India - en-IN)</option>
          </select>
        </div>
      </div>

      <div id="unsupported-msg" style="display: none; background: rgba(255, 122, 0, 0.15); border: 1px solid #FF7A00; border-radius: 8px; padding: 10px 14px; font-size: 0.82rem; color: #FFB366; margin-bottom: 10px;">
        ⚠️ <strong>Browser Notice:</strong> The Web Speech API is supported in Google Chrome and Microsoft Edge. For unsupported browsers or offline environments, please type directly into the text box below.
      </div>

      <div style="display: flex; gap: 10px; margin-bottom: 10px; flex-wrap: wrap;">
        <button id="record-btn" onclick="toggleDictation()" style="
          background: linear-gradient(180deg, #098A98 0%, #065B66 100%);
          color: #FFFFFF;
          border: 1px solid #66E3C4;
          border-radius: 10px;
          padding: 7px 18px;
          font-weight: 700;
          font-size: 0.84rem;
          cursor: pointer;
          display: flex;
          align-items: center;
          gap: 6px;
          box-shadow: 0 4px 12px rgba(8, 127, 140, 0.3);
          transition: all 180ms ease;
        ">
          <span id="btn-icon">🎤</span> <span id="btn-label">Start Speaking</span>
        </button>

        <button id="copy-btn" onclick="copyTranscript()" style="
          background: rgba(255, 255, 255, 0.08);
          color: #E2E8F0;
          border: 1px solid rgba(255, 255, 255, 0.15);
          border-radius: 10px;
          padding: 7px 16px;
          font-size: 0.84rem;
          cursor: pointer;
          display: flex;
          align-items: center;
          gap: 6px;
        ">
          📋 Copy to Clipboard
        </button>

        <button onclick="clearTranscript()" style="
          background: transparent;
          color: #94A3B8;
          border: 1px solid rgba(255, 255, 255, 0.08);
          border-radius: 10px;
          padding: 7px 14px;
          font-size: 0.82rem;
          cursor: pointer;
        ">
          ✕ Clear
        </button>
      </div>

      <textarea id="transcript-box" rows="3" placeholder="Dictated symptoms in Telugu, Hindi, or English will transcribe here in real-time. Clinician may edit freely..." style="
        width: 100%;
        box-sizing: border-box;
        background: rgba(5, 14, 31, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 8px;
        padding: 10px 12px;
        color: #FFFFFF;
        font-size: 0.88rem;
        font-family: inherit;
        resize: vertical;
        outline: none;
      "></textarea>

      <div style="font-size: 0.76rem; color: #94A3B8; margin-top: 6px; display: flex; justify-content: space-between; align-items: center;">
        <span>🛡️ <strong>Safety Rule:</strong> Dictation provides a draft for clinician review. The nurse always edits and submits.</span>
        <span id="copy-status" style="color: #66E3C4; font-weight: 600;"></span>
      </div>
    </div>

    <script>
      let recognition = null;
      let isRecording = false;
      const transcriptBox = document.getElementById('transcript-box');
      const recordBtn = document.getElementById('record-btn');
      const btnIcon = document.getElementById('btn-icon');
      const btnLabel = document.getElementById('btn-label');
      const dialectSelect = document.getElementById('dialect-select');
      const unsupportedMsg = document.getElementById('unsupported-msg');
      const copyStatus = document.getElementById('copy-status');

      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

      if (!SpeechRecognition) {
        unsupportedMsg.style.display = 'block';
        recordBtn.disabled = true;
        recordBtn.style.opacity = '0.5';
        recordBtn.style.cursor = 'not-allowed';
      }

      function toggleDictation() {
        if (!SpeechRecognition) return;

        if (isRecording) {
          recognition.stop();
          return;
        }

        try {
          recognition = new SpeechRecognition();
          recognition.lang = dialectSelect.value;
          recognition.continuous = true;
          recognition.interimResults = true;

          recognition.onstart = () => {
            isRecording = true;
            recordBtn.style.background = 'linear-gradient(180deg, #FF2A55 0%, #A00025 100%)';
            recordBtn.style.borderColor = '#FF8099';
            btnIcon.textContent = '⏹️';
            btnLabel.textContent = 'Stop Dictation';
          };

          recognition.onresult = (event) => {
            let finalStr = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {
              finalStr += event.results[i][0].transcript;
            }
            if (transcriptBox.value.length > 0 && !transcriptBox.value.endsWith(' ')) {
              transcriptBox.value += ' ' + finalStr;
            } else {
              transcriptBox.value += finalStr;
            }
          };

          recognition.onerror = (event) => {
            console.warn('SpeechRecognition error:', event.error);
            stopRecordingUI();
          };

          recognition.onend = () => {
            stopRecordingUI();
          };

          recognition.start();
        } catch (err) {
          console.error(err);
          stopRecordingUI();
        }
      }

      function stopRecordingUI() {
        isRecording = false;
        recordBtn.style.background = 'linear-gradient(180deg, #098A98 0%, #065B66 100%)';
        recordBtn.style.borderColor = '#66E3C4';
        btnIcon.textContent = '🎤';
        btnLabel.textContent = 'Start Speaking';
      }

      function copyTranscript() {
        if (!transcriptBox.value.trim()) return;
        navigator.clipboard.writeText(transcriptBox.value.trim()).then(() => {
          copyStatus.textContent = '✓ Copied to clipboard!';
          setTimeout(() => { copyStatus.textContent = ''; }, 3000);
        }).catch(err => {
          transcriptBox.select();
          document.execCommand('copy');
          copyStatus.textContent = '✓ Copied!';
          setTimeout(() => { copyStatus.textContent = ''; }, 3000);
        });
      }

      function clearTranscript() {
        transcriptBox.value = '';
        copyStatus.textContent = '';
      }
    </script>
    """
    components.html(html_code, height=220)


def render_read_result_aloud_widget(encounter: Dict[str, Any], include_name: bool = False) -> None:
    """Renders the Web Speech API speechSynthesis read-aloud controller."""
    spoken_text = build_read_aloud_text(encounter, include_name=include_name)
    escaped_text = json.dumps(spoken_text)

    html_code = f"""
    <div id="read-aloud-widget" style="
      background: linear-gradient(135deg, rgba(8, 22, 42, 0.9) 0%, rgba(5, 14, 31, 0.95) 100%);
      border: 1px solid rgba(102, 227, 196, 0.3);
      border-radius: 12px;
      padding: 12px 18px;
      margin: 12px 0;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
      font-family: 'Inter', system-ui, sans-serif;
    ">
      <div style="display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 1.3rem;">🔊</span>
        <div>
          <div style="font-family: 'Sora', sans-serif; font-size: 0.88rem; font-weight: 700; color: #FFFFFF;">
            Audio Announcement &bull; Speech Synthesis
          </div>
          <div style="font-size: 0.74rem; color: #94A3B8;">
            { 'Privacy Safe: Patient name omitted from waiting hall broadcast.' if not include_name else '⚠️ Staff Alert: Patient name included in broadcast.' }
          </div>
        </div>
      </div>

      <div style="display: flex; gap: 8px; align-items: center;">
        <button id="speak-btn" onclick="startSpeech()" style="
          background: linear-gradient(180deg, #098A98 0%, #065B66 100%);
          color: #FFFFFF;
          border: 1px solid #66E3C4;
          border-radius: 8px;
          padding: 6px 14px;
          font-size: 0.82rem;
          font-weight: 700;
          cursor: pointer;
          display: inline-flex;
          align-items: center;
          gap: 6px;
        ">
          ▶️ Play Result
        </button>
        <button id="stop-btn" onclick="stopSpeech()" style="
          background: rgba(255, 255, 255, 0.08);
          color: #CBD5E1;
          border: 1px solid rgba(255, 255, 255, 0.15);
          border-radius: 8px;
          padding: 6px 12px;
          font-size: 0.82rem;
          cursor: pointer;
        ">
          ⏹️ Stop
        </button>
      </div>
    </div>

    <script>
      const speechText = {escaped_text};

      function startSpeech() {{
        if (!('speechSynthesis' in window)) {{
          alert('Web Speech Synthesis is not supported in this browser.');
          return;
        }}
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(speechText);
        utterance.rate = 0.95;
        utterance.pitch = 1.0;
        
        // Pick regional Indian English voice if available
        const voices = window.speechSynthesis.getVoices();
        const indianVoice = voices.find(v => v.lang === 'en-IN' || v.name.includes('India'));
        if (indianVoice) {{
          utterance.voice = indianVoice;
        }}

        window.speechSynthesis.speak(utterance);
      }}

      function stopSpeech() {{
        if ('speechSynthesis' in window) {{
          window.speechSynthesis.cancel();
        }}
      }}
    </script>
    """
    components.html(html_code, height=85)
