-- SQLite Schema for Voice Chat Application
-- This file is for reference. Schema is auto-created by database_director.py on startup.

-- Characters table
-- Stores character configurations that can be activated for conversations
CREATE TABLE IF NOT EXISTS characters (
    id TEXT PRIMARY KEY,                           -- e.g. "luna-001", auto-generated from name
    name TEXT NOT NULL,                            -- Display name
    voice TEXT DEFAULT '',                         -- Voice ID reference (from voices table)
    system_prompt TEXT DEFAULT '',                 -- Character's system prompt for LLM
    image_url TEXT DEFAULT '',                     -- Primary image URL
    images TEXT DEFAULT '[]',                      -- JSON array of additional image URLs
    is_active INTEGER DEFAULT 0,                   -- 1 = active in current conversation
    last_message TEXT DEFAULT '',                  -- Last message from this character
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Voices table
-- Stores voice configurations for TTS, independent of characters
CREATE TABLE IF NOT EXISTS voices (
    voice TEXT PRIMARY KEY,                        -- Voice name/ID (e.g. "luna_warm")
    method TEXT DEFAULT '',                        -- TTS method
    audio_path TEXT DEFAULT '',                    -- Path to reference audio file
    text_path TEXT DEFAULT '',                     -- Path to reference text file
    speaker_desc TEXT DEFAULT '',                  -- Speaker description for TTS
    scene_prompt TEXT DEFAULT '',                  -- Scene/context prompt
    audio_tokens TEXT DEFAULT NULL,                -- JSON: Cached audio tokens for fast TTS
    id TEXT DEFAULT NULL,                          -- UUID for additional reference
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Conversations table
-- Stores conversation sessions
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,              -- UUID
    title TEXT,                                    -- Auto-generated or custom title
    active_characters TEXT DEFAULT '[]',           -- JSON array of character references
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Messages table
-- Stores individual messages within conversations
CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,                   -- UUID
    conversation_id TEXT NOT NULL,                 -- FK to conversations
    role TEXT NOT NULL,                            -- "user", "assistant", "system"
    name TEXT,                                     -- Speaker name (user name or character name)
    content TEXT NOT NULL,                         -- Message content
    character_id TEXT,                             -- FK to characters (for assistant messages)
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_characters_is_active ON characters(is_active);
