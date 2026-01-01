/**
 * speech.js - Speech Page Functionality
 * Handles voice generation and speech settings
 */

import { characterCache } from './characterCache.js';
import { handleDbError } from './db.js';
import * as websocket from './websocket.js';

let isSpeechInitialized = false;
let voices = [];
let selectedVoiceName = null;
let createButtonTemplate = null;

/**
 * Initialize the speech page
 */
export function initSpeech() {
  if (isSpeechInitialized) return;

  console.log('Initializing speech page...');

  createNotificationContainer();
  setupEventListeners();
  cacheButtonTemplates();
  setupCacheEventHandlers();

  // Initialize state
  handleMethodChange();
  loadVoices();

  isSpeechInitialized = true;
}

/**
 * Cache button templates so icons can be restored
 */
function cacheButtonTemplates() {
  const createBtn = document.getElementById('speech-create-btn');
  if (createBtn && !createButtonTemplate) {
    createButtonTemplate = createBtn.innerHTML;
  }
}

/**
 * Ensure WebSocket is connected before DB requests
 */
async function ensureWebSocketConnected() {
  if (!websocket.isConnected()) {
    await websocket.connect();
  }
}

/**
 * Load voices from cache (initialize if needed)
 */
async function loadVoices() {
  try {
    await ensureWebSocketConnected();

    if (!characterCache.isInitialized) {
      const data = await characterCache.initialize();
      voices = data.voices || [];
    } else {
      voices = characterCache.getAllVoices();
    }

    renderVoiceList();
  } catch (error) {
    console.error('Error loading voices:', error);
    const errorMessage = handleDbError(error);
    showNotification('Error Loading Voices', errorMessage, 'error');
  }
}

/**
 * Setup cache event handlers for UI updates
 */
function setupCacheEventHandlers() {
  characterCache.on('voice:created', () => {
    voices = characterCache.getAllVoices();
    renderVoiceList();
  });

  characterCache.on('voice:updated', () => {
    voices = characterCache.getAllVoices();
    renderVoiceList();
  });

  characterCache.on('voice:deleted', () => {
    voices = characterCache.getAllVoices();
    if (selectedVoiceName && !characterCache.getVoice(selectedVoiceName)) {
      resetForm();
    }
    renderVoiceList();
  });
}

/**
 * Setup event listeners for the speech page
 */
function setupEventListeners() {
  // Method radio buttons
  const cloneRadio = document.getElementById('speech-method-clone');
  const profileRadio = document.getElementById('speech-method-profile');

  if (cloneRadio && profileRadio) {
    cloneRadio.addEventListener('change', handleMethodChange);
    profileRadio.addEventListener('change', handleMethodChange);
  }

  // Create/Save Voice button
  const createBtn = document.getElementById('speech-create-btn');
  if (createBtn) {
    createBtn.addEventListener('click', handleSaveVoice);
  }

  // Delete Voice button
  const deleteBtn = document.getElementById('speech-delete-btn');
  if (deleteBtn) {
    deleteBtn.addEventListener('click', handleDeleteVoice);
  }

  // New Voice button
  const newVoiceBtn = document.getElementById('speech-new-voice-btn');
  if (newVoiceBtn) {
    newVoiceBtn.addEventListener('click', () => {
      resetForm();
    });
  }
}

/**
 * Handle voice creation method change
 */
function handleMethodChange() {
  const cloneRadio = document.getElementById('speech-method-clone');
  const profileRadio = document.getElementById('speech-method-profile');

  // Inputs
  const speakerDesc = document.getElementById('speech-speaker-description');
  const audioPath = document.getElementById('speech-audio-path');
  const textPath = document.getElementById('speech-text-path');

  if (!cloneRadio || !profileRadio) return;

  if (cloneRadio.checked) {
    // Clone Method Active
    if (speakerDesc) {
      speakerDesc.disabled = true;
      speakerDesc.classList.add('disabled');
      speakerDesc.placeholder = 'Not used in Clone mode';
    }

    if (audioPath) {
      audioPath.disabled = false;
      audioPath.classList.remove('disabled');
    }

    if (textPath) {
      textPath.disabled = false;
      textPath.classList.remove('disabled');
    }

  } else if (profileRadio.checked) {
    // Profile Method Active
    if (speakerDesc) {
      speakerDesc.disabled = false;
      speakerDesc.classList.remove('disabled');
      speakerDesc.placeholder = "Describe the speaker's voice characteristics, tone, accent, and style...";
    }

    if (audioPath) {
      audioPath.disabled = true;
      audioPath.classList.add('disabled');
    }

    if (textPath) {
      textPath.disabled = true;
      textPath.classList.add('disabled');
    }
  }
}

/**
 * Render the voice list
 */
function renderVoiceList() {
  const list = document.getElementById('speech-voice-list');
  const empty = document.getElementById('speech-voice-empty');

  if (!list || !empty) return;

  list.innerHTML = '';

  const sortedVoices = [...voices].sort((a, b) => {
    return (a.voice || '').localeCompare(b.voice || '');
  });

  if (sortedVoices.length === 0) {
    empty.style.display = 'block';
    return;
  }

  empty.style.display = 'none';

  sortedVoices.forEach((voice) => {
    const item = document.createElement('div');
    item.className = 'voice-list-item';
    if (selectedVoiceName && voice.voice === selectedVoiceName) {
      item.classList.add('active');
    }

    const methodLabel = voice.method === 'clone' ? 'Clone' : 'Profile';

    item.innerHTML = `
      <div>
        <div class="voice-list-name">${voice.voice}</div>
        <div class="voice-list-meta">${methodLabel}</div>
      </div>
    `;

    item.addEventListener('click', () => {
      setSelectedVoice(voice);
    });

    list.appendChild(item);
  });
}

/**
 * Set the currently selected voice and populate form
 */
function setSelectedVoice(voice) {
  selectedVoiceName = voice.voice;

  const nameInput = document.getElementById('speech-voice-name');
  const cloneRadio = document.getElementById('speech-method-clone');
  const profileRadio = document.getElementById('speech-method-profile');
  const speakerDesc = document.getElementById('speech-speaker-description');
  const scenePrompt = document.getElementById('speech-scene-prompt');
  const audioPath = document.getElementById('speech-audio-path');
  const textPath = document.getElementById('speech-text-path');

  if (nameInput) nameInput.value = voice.voice || '';
  if (speakerDesc) speakerDesc.value = voice.speaker_desc || '';
  if (scenePrompt) scenePrompt.value = voice.scene_prompt || '';
  if (audioPath) audioPath.value = voice.audio_path || '';
  if (textPath) textPath.value = voice.text_path || '';

  const method = voice.method === 'clone' ? 'clone' : 'profile';
  if (cloneRadio && profileRadio) {
    cloneRadio.checked = method === 'clone';
    profileRadio.checked = method === 'profile';
  }

  handleMethodChange();
  updateFormButtons(true);
  renderVoiceList();
}

/**
 * Reset form for creating a new voice
 */
function resetForm() {
  selectedVoiceName = null;

  const nameInput = document.getElementById('speech-voice-name');
  const cloneRadio = document.getElementById('speech-method-clone');
  const profileRadio = document.getElementById('speech-method-profile');
  const speakerDesc = document.getElementById('speech-speaker-description');
  const scenePrompt = document.getElementById('speech-scene-prompt');
  const audioPath = document.getElementById('speech-audio-path');
  const textPath = document.getElementById('speech-text-path');

  if (nameInput) nameInput.value = '';
  if (speakerDesc) speakerDesc.value = '';
  if (scenePrompt) scenePrompt.value = '';
  if (audioPath) audioPath.value = '';
  if (textPath) textPath.value = '';

  if (cloneRadio && profileRadio) {
    cloneRadio.checked = true;
    profileRadio.checked = false;
  }

  handleMethodChange();
  updateFormButtons(false);
  renderVoiceList();
}

/**
 * Update form buttons based on editing state
 */
function updateFormButtons(isEditing) {
  const createBtn = document.getElementById('speech-create-btn');
  const deleteBtn = document.getElementById('speech-delete-btn');

  if (createBtn) {
    if (isEditing) {
      createBtn.textContent = 'Save Changes';
    } else if (createButtonTemplate) {
      createBtn.innerHTML = createButtonTemplate;
    } else {
      createBtn.textContent = 'Create Voice';
    }
  }

  if (deleteBtn) {
    deleteBtn.disabled = !isEditing;
  }
}

/**
 * Handle Create/Update Voice button click
 */
async function handleSaveVoice() {
  const nameInput = document.getElementById('speech-voice-name');
  const cloneRadio = document.getElementById('speech-method-clone');
  const speakerDesc = document.getElementById('speech-speaker-description');
  const scenePrompt = document.getElementById('speech-scene-prompt');
  const audioPath = document.getElementById('speech-audio-path');
  const textPath = document.getElementById('speech-text-path');
  const createBtn = document.getElementById('speech-create-btn');

  const voiceName = nameInput?.value?.trim();

  // Validation
  if (!voiceName) {
    showNotification('Validation Error', 'Please enter a name for this voice', 'error');
    return;
  }

  const method = cloneRadio?.checked ? 'clone' : 'profile';

  if (method === 'profile' && (!speakerDesc || !speakerDesc.value.trim())) {
    showNotification('Validation Error', 'Speaker description is required for Profile method', 'error');
    return;
  }

  if (method === 'clone' && ((!audioPath || !audioPath.value.trim()) || (!textPath || !textPath.value.trim()))) {
    showNotification('Validation Error', 'Audio path and text path are required for Clone method', 'error');
    return;
  }

  const voiceData = {
    voice: voiceName,
    method: method,
    speaker_desc: speakerDesc ? speakerDesc.value : '',
    scene_prompt: scenePrompt ? scenePrompt.value : '',
    audio_path: audioPath ? audioPath.value : '',
    text_path: textPath ? textPath.value : ''
  };

  // UI Feedback
  if (createBtn) {
    createBtn.disabled = true;
    createBtn.innerHTML = `<span class="loading-spinner-small"></span> Saving...`;
  }

  try {
    let data;
    const isEditing = Boolean(selectedVoiceName);

    if (isEditing) {
      const updates = {
        method: voiceData.method,
        speaker_desc: voiceData.speaker_desc,
        scene_prompt: voiceData.scene_prompt,
        audio_path: voiceData.audio_path,
        text_path: voiceData.text_path
      };

      if (selectedVoiceName !== voiceData.voice) {
        updates.new_voice = voiceData.voice;
      }

      data = await characterCache.updateVoice(selectedVoiceName, updates);
      selectedVoiceName = data.voice;

      showNotification(
        'Voice Updated',
        `Voice "${data.voice}" updated successfully!`,
        'success'
      );
    } else {
      data = await characterCache.createVoice(voiceData);
      selectedVoiceName = data.voice;

      showNotification(
        'Voice Created',
        `Voice "${data.voice}" created successfully!`,
        'success'
      );
    }

    setSelectedVoice(data);
  } catch (error) {
    console.error('Error saving voice:', error);
    const errorMessage = handleDbError(error);
    showNotification('Error Saving Voice', errorMessage, 'error');
  } finally {
    if (createBtn) {
      createBtn.disabled = false;
      updateFormButtons(Boolean(selectedVoiceName));
    }
  }
}

/**
 * Handle Delete Voice button click
 */
async function handleDeleteVoice() {
  if (!selectedVoiceName) {
    return;
  }

  if (!confirm(`Delete voice "${selectedVoiceName}"? This cannot be undone.`)) {
    return;
  }

  const deleteBtn = document.getElementById('speech-delete-btn');
  if (deleteBtn) {
    deleteBtn.disabled = true;
    deleteBtn.textContent = 'Deleting...';
  }

  try {
    await characterCache.deleteVoice(selectedVoiceName);
    showNotification('Voice Deleted', `Voice "${selectedVoiceName}" deleted.`, 'success');
    resetForm();
  } catch (error) {
    console.error('Error deleting voice:', error);
    const errorMessage = handleDbError(error);
    showNotification('Error Deleting Voice', errorMessage, 'error');
  } finally {
    if (deleteBtn) {
      deleteBtn.textContent = 'Delete Voice';
      deleteBtn.disabled = !selectedVoiceName;
    }
  }
}

/**
 * Create notification container if needed
 */
function createNotificationContainer() {
  if (document.getElementById('notification-container')) {
    return;
  }

  const container = document.createElement('div');
  container.id = 'notification-container';
  container.className = 'notification-container';
  document.body.appendChild(container);
}

/**
 * Show notification helper
 */
function showNotification(title, message, type = 'info') {
  const container = document.getElementById('notification-container');
  if (!container) return;

  const notification = document.createElement('div');
  notification.className = `notification notification-${type} slide-in`;

  let icon = '';
  if (type === 'success') icon = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" /></svg>';
  else if (type === 'error') icon = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" /></svg>';
  else icon = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>';

  notification.innerHTML = `
    <div class="notification-icon">${icon}</div>
    <div class="notification-content">
      <div class="notification-title">${title}</div>
      <div class="notification-message">${message}</div>
    </div>
    <button class="notification-close">&times;</button>
  `;

  container.appendChild(notification);

  notification.querySelector('.notification-close').addEventListener('click', () => {
    notification.classList.add('slide-out');
    setTimeout(() => notification.remove(), 300);
  });

  setTimeout(() => {
    if (notification.parentNode) {
      notification.classList.add('slide-out');
      setTimeout(() => notification.remove(), 300);
    }
  }, 5000);
}
