/**
 * db.js - WebSocket-based Database Operations
 * Provides promise-based access to SQLite database via WebSocket
 * Zero HTTP overhead, uses existing WebSocket connection
 */

import * as websocket from './websocket.js'

// Pending request handlers waiting for responses
const pendingRequests = new Map()

// Subscribe to all WebSocket messages to route responses
let listenerRegistered = false

function ensureListener() {
  if (listenerRegistered) return

  websocket.onMessage((message) => {
    const { type, data, error } = message

    // Check if this is a response we're waiting for
    if (pendingRequests.has(type)) {
      const { resolve, reject } = pendingRequests.get(type)
      pendingRequests.delete(type)
      resolve(data)
    }

    // Handle database errors
    if (type === 'db_error' && error) {
      // Find and reject any pending request (error responses don't have matching type)
      // In practice, errors come right after the request, so the most recent pending request is the one
      const pending = Array.from(pendingRequests.entries())
      if (pending.length > 0) {
        const [responseType, { reject }] = pending[pending.length - 1]
        pendingRequests.delete(responseType)
        reject(new Error(error))
      }
    }
  })

  listenerRegistered = true
}

/**
 * Send a database request and wait for response
 * @param {string} requestType - Message type to send
 * @param {string} responseType - Expected response message type
 * @param {object} payload - Data to send
 * @param {number} timeout - Timeout in ms (default 5000)
 * @returns {Promise<any>}
 */
function request(requestType, responseType, payload = {}, timeout = 5000) {
  ensureListener()

  return new Promise((resolve, reject) => {
    // Set up timeout
    const timeoutId = setTimeout(() => {
      if (pendingRequests.has(responseType)) {
        pendingRequests.delete(responseType)
        reject(new Error(`Request timeout: ${requestType}`))
      }
    }, timeout)

    // Register pending request
    pendingRequests.set(responseType, {
      resolve: (data) => {
        clearTimeout(timeoutId)
        resolve(data)
      },
      reject: (err) => {
        clearTimeout(timeoutId)
        reject(err)
      }
    })

    // Send request
    websocket.sendText({ type: requestType, data: payload })
  })
}

// ============================================
// Characters
// ============================================

/**
 * Get all characters
 * @returns {Promise<Array>}
 */
export async function getCharacters() {
  return request('get_characters', 'characters_data')
}

/**
 * Get a single character by ID
 * @param {string} id
 * @returns {Promise<object>}
 */
export async function getCharacter(id) {
  return request('get_character', 'character_data', { id })
}

/**
 * Create a new character
 * @param {object} characterData - {name, voice, system_prompt, image_url, images, is_active}
 * @returns {Promise<object>}
 */
export async function createCharacter(characterData) {
  return request('create_character', 'character_created', characterData)
}

/**
 * Update an existing character
 * @param {string} id
 * @param {object} updates
 * @returns {Promise<object>}
 */
export async function updateCharacter(id, updates) {
  return request('update_character', 'character_updated', { id, ...updates })
}

/**
 * Delete a character
 * @param {string} id
 * @returns {Promise<object>}
 */
export async function deleteCharacter(id) {
  return request('delete_character', 'character_deleted', { id })
}

// ============================================
// Voices
// ============================================

/**
 * Get all voices
 * @returns {Promise<Array>}
 */
export async function getVoices() {
  return request('get_voices', 'voices_data')
}

/**
 * Get a single voice by name
 * @param {string} voice
 * @returns {Promise<object>}
 */
export async function getVoice(voice) {
  return request('get_voice', 'voice_data', { voice })
}

/**
 * Create a new voice
 * @param {object} voiceData
 * @returns {Promise<object>}
 */
export async function createVoice(voiceData) {
  return request('create_voice', 'voice_created', voiceData)
}

/**
 * Update an existing voice
 * @param {string} voice - Voice name
 * @param {object} updates
 * @returns {Promise<object>}
 */
export async function updateVoice(voice, updates) {
  return request('update_voice', 'voice_updated', { voice, ...updates })
}

/**
 * Delete a voice
 * @param {string} voice
 * @returns {Promise<object>}
 */
export async function deleteVoice(voice) {
  return request('delete_voice', 'voice_deleted', { voice })
}

// ============================================
// Conversations
// ============================================

/**
 * Get all conversations
 * @param {object} options - {limit, offset}
 * @returns {Promise<Array>}
 */
export async function getConversations(options = {}) {
  return request('get_conversations', 'conversations_data', options)
}

/**
 * Get a single conversation by ID
 * @param {string} conversationId
 * @returns {Promise<object>}
 */
export async function getConversation(conversationId) {
  return request('get_conversation', 'conversation_data', { conversation_id: conversationId })
}

/**
 * Create a new conversation
 * @param {object} conversationData - {title, active_characters}
 * @returns {Promise<object>}
 */
export async function createConversation(conversationData = {}) {
  return request('create_conversation', 'conversation_created', conversationData)
}

/**
 * Update an existing conversation
 * @param {string} conversationId
 * @param {object} updates - {title, active_characters}
 * @returns {Promise<object>}
 */
export async function updateConversation(conversationId, updates) {
  return request('update_conversation', 'conversation_updated', { conversation_id: conversationId, ...updates })
}

/**
 * Delete a conversation
 * @param {string} conversationId
 * @returns {Promise<object>}
 */
export async function deleteConversation(conversationId) {
  return request('delete_conversation', 'conversation_deleted', { conversation_id: conversationId })
}

// ============================================
// Messages
// ============================================

/**
 * Get messages for a conversation
 * @param {string} conversationId
 * @param {object} options - {limit, offset}
 * @returns {Promise<Array>}
 */
export async function getMessages(conversationId, options = {}) {
  return request('get_messages', 'messages_data', { conversation_id: conversationId, ...options })
}

/**
 * Create a new message
 * @param {object} messageData - {conversation_id, role, content, name, character_id}
 * @returns {Promise<object>}
 */
export async function createMessage(messageData) {
  return request('create_message', 'message_created', messageData)
}

/**
 * Delete a message
 * @param {string} messageId
 * @returns {Promise<object>}
 */
export async function deleteMessage(messageId) {
  return request('delete_message', 'message_deleted', { message_id: messageId })
}

// ============================================
// Error Handling Utility
// ============================================

/**
 * Handle database errors with user-friendly messages
 * @param {Error} error
 * @returns {string}
 */
export function handleDbError(error) {
  const message = error?.message || 'Unknown database error'

  if (message.includes('not found')) {
    return 'The requested item was not found'
  }
  if (message.includes('timeout')) {
    return 'Request timed out. Please check your connection.'
  }
  if (message.includes('unique') || message.includes('duplicate')) {
    return 'An item with this name already exists'
  }

  return message
}
