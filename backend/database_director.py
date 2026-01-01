"""
Database Director Module
SQLite-based database operations for Characters, Voices, Conversations, and Messages.
Optimized for low-latency single-user voice chat application.
"""

import os
import re
import json
import logging
import asyncio
import uuid
import aiosqlite
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel
from fastapi import HTTPException

logger = logging.getLogger(__name__)

########################################
##--         Configuration          --##
########################################

DB_PATH = os.path.join(os.path.dirname(__file__), "voicechat.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS characters (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    voice TEXT DEFAULT '',
    system_prompt TEXT DEFAULT '',
    image_url TEXT DEFAULT '',
    images TEXT DEFAULT '[]',
    is_active INTEGER DEFAULT 0,
    last_message TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS voices (
    voice TEXT PRIMARY KEY,
    method TEXT DEFAULT '',
    audio_path TEXT DEFAULT '',
    text_path TEXT DEFAULT '',
    speaker_desc TEXT DEFAULT '',
    scene_prompt TEXT DEFAULT '',
    audio_tokens TEXT DEFAULT NULL,
    id TEXT DEFAULT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    title TEXT,
    active_characters TEXT DEFAULT '[]',
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    name TEXT,
    content TEXT NOT NULL,
    character_id TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_characters_is_active ON characters(is_active);
"""

########################################
##--          Data Models           --##
########################################

class Character(BaseModel):
    id: str
    name: str
    voice: str = ""
    system_prompt: str = ""
    image_url: str = ""
    images: List[str] = []
    is_active: bool = False
    last_message: str = ""
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class CharacterCreate(BaseModel):
    name: str
    voice: str = ""
    system_prompt: str = ""
    image_url: str = ""
    images: List[str] = []
    is_active: bool = False


class CharacterUpdate(BaseModel):
    name: Optional[str] = None
    voice: Optional[str] = None
    system_prompt: Optional[str] = None
    image_url: Optional[str] = None
    images: Optional[List[str]] = None
    is_active: Optional[bool] = None
    last_message: Optional[str] = None


class Voice(BaseModel):
    voice: str
    method: str = ""
    audio_path: str = ""
    text_path: str = ""
    speaker_desc: str = ""
    scene_prompt: str = ""
    audio_tokens: Optional[Any] = None
    id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class VoiceCreate(BaseModel):
    voice: str
    method: str = ""
    audio_path: str = ""
    text_path: str = ""
    speaker_desc: str = ""
    scene_prompt: str = ""


class VoiceUpdate(BaseModel):
    method: Optional[str] = None
    audio_path: Optional[str] = None
    text_path: Optional[str] = None
    speaker_desc: Optional[str] = None
    scene_prompt: Optional[str] = None
    audio_tokens: Optional[Any] = None


class Conversation(BaseModel):
    conversation_id: str
    title: Optional[str] = None
    active_characters: List[Dict[str, Any]] = []
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ConversationCreate(BaseModel):
    title: Optional[str] = None
    active_characters: List[Dict[str, Any]] = []


class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    active_characters: Optional[List[Dict[str, Any]]] = None


class Message(BaseModel):
    message_id: str
    conversation_id: str
    role: str
    name: Optional[str] = None
    content: str
    character_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class MessageCreate(BaseModel):
    conversation_id: str
    role: str
    content: str
    name: Optional[str] = None
    character_id: Optional[str] = None


########################################
##--       Database Director        --##
########################################

class DatabaseDirector:
    """
    SQLite-based database management for voice chat application.
    Uses in-memory caching for characters and voices at startup.
    Background writes for messages to avoid latency.
    """

    def __init__(self):
        self._character_cache: Dict[str, Character] = {}
        self._voice_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_loaded = False

    async def init_database(self):
        """Initialize database schema and load caches."""
        async with aiosqlite.connect(DB_PATH) as conn:
            await conn.executescript(SCHEMA_SQL)
            await conn.commit()
        logger.info(f"SQLite database initialized at {DB_PATH}")

        await self._load_caches()

    async def _load_caches(self):
        """Load characters and voices into memory at startup."""
        async with aiosqlite.connect(DB_PATH) as conn:
            conn.row_factory = aiosqlite.Row

            # Load all characters
            cursor = await conn.execute("SELECT * FROM characters")
            rows = await cursor.fetchall()
            for row in rows:
                character = self._row_to_character(row)
                self._character_cache[character.id] = character

            # Load all voices
            cursor = await conn.execute("SELECT * FROM voices")
            rows = await cursor.fetchall()
            for row in rows:
                voice = self._row_to_voice(row)
                self._voice_cache[voice.voice] = {
                    "config": voice,
                    "audio_tokens": voice.audio_tokens
                }

        self._cache_loaded = True
        logger.info(f"Loaded {len(self._character_cache)} characters and {len(self._voice_cache)} voices into cache")

    def _row_to_character(self, row: aiosqlite.Row) -> Character:
        """Convert database row to Character model."""
        return Character(
            id=row["id"],
            name=row["name"],
            voice=row["voice"] or "",
            system_prompt=row["system_prompt"] or "",
            image_url=row["image_url"] or "",
            images=json.loads(row["images"]) if row["images"] else [],
            is_active=bool(row["is_active"]),
            last_message=row["last_message"] or "",
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        )

    def _row_to_voice(self, row: aiosqlite.Row) -> Voice:
        """Convert database row to Voice model."""
        audio_tokens = None
        if row["audio_tokens"]:
            try:
                audio_tokens = json.loads(row["audio_tokens"])
            except (json.JSONDecodeError, TypeError):
                audio_tokens = row["audio_tokens"]

        return Voice(
            voice=row["voice"],
            method=row["method"] or "",
            audio_path=row["audio_path"] or "",
            text_path=row["text_path"] or "",
            speaker_desc=row["speaker_desc"] or "",
            scene_prompt=row["scene_prompt"] or "",
            audio_tokens=audio_tokens,
            id=row["id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        )

    def _row_to_conversation(self, row: aiosqlite.Row) -> Conversation:
        """Convert database row to Conversation model."""
        return Conversation(
            conversation_id=row["conversation_id"],
            title=row["title"],
            active_characters=json.loads(row["active_characters"]) if row["active_characters"] else [],
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        )

    def _row_to_message(self, row: aiosqlite.Row) -> Message:
        """Convert database row to Message model."""
        return Message(
            message_id=row["message_id"],
            conversation_id=row["conversation_id"],
            role=row["role"],
            name=row["name"],
            content=row["content"],
            character_id=row["character_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        )

    ########################################
    ##--      Character Operations      --##
    ########################################

    async def _generate_character_id(self, name: str) -> str:
        """Generate a sequential ID from the character name."""
        base_id = name.lower().strip()
        base_id = re.sub(r'[^a-z0-9\s-]', '', base_id)
        base_id = re.sub(r'\s+', '-', base_id)
        base_id = re.sub(r'-+', '-', base_id)
        base_id = base_id.strip('-')

        if not base_id:
            base_id = "character"

        try:
            async with aiosqlite.connect(DB_PATH) as conn:
                cursor = await conn.execute(
                    "SELECT id FROM characters WHERE id LIKE ?",
                    (f"{base_id}-%",)
                )
                rows = await cursor.fetchall()

            highest_num = 0
            pattern = re.compile(rf"^{re.escape(base_id)}-(\d{{3}})$")

            for row in rows:
                match = pattern.match(row[0])
                if match:
                    num = int(match.group(1))
                    highest_num = max(highest_num, num)

            next_num = highest_num + 1
            character_id = f"{base_id}-{next_num:03d}"

            logger.info(f"Generated character id: {character_id}")
            return character_id

        except Exception as e:
            logger.error(f"Error generating character id: {e}")
            return f"{base_id}-001"

    async def get_all_characters(self) -> List[Character]:
        """Get all characters from cache."""
        if self._cache_loaded:
            characters = list(self._character_cache.values())
            logger.info(f"Retrieved {len(characters)} characters from cache")
            return characters

        # Fallback to database if cache not loaded
        try:
            async with aiosqlite.connect(DB_PATH) as conn:
                conn.row_factory = aiosqlite.Row
                cursor = await conn.execute("SELECT * FROM characters")
                rows = await cursor.fetchall()

            characters = [self._row_to_character(row) for row in rows]
            logger.info(f"Retrieved {len(characters)} characters from database")
            return characters

        except Exception as e:
            logger.error(f"Error getting characters: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def get_active_characters(self) -> List[Character]:
        """Get all active characters from cache."""
        if self._cache_loaded:
            characters = [c for c in self._character_cache.values() if c.is_active]
            logger.info(f"Retrieved {len(characters)} active characters from cache")
            return characters

        try:
            async with aiosqlite.connect(DB_PATH) as conn:
                conn.row_factory = aiosqlite.Row
                cursor = await conn.execute("SELECT * FROM characters WHERE is_active = 1")
                rows = await cursor.fetchall()

            characters = [self._row_to_character(row) for row in rows]
            logger.info(f"Retrieved {len(characters)} active characters from database")
            return characters

        except Exception as e:
            logger.error(f"Error getting active characters: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def get_character(self, character_id: str) -> Character:
        """Get a specific character by ID."""
        if self._cache_loaded and character_id in self._character_cache:
            return self._character_cache[character_id]

        try:
            async with aiosqlite.connect(DB_PATH) as conn:
                conn.row_factory = aiosqlite.Row
                cursor = await conn.execute(
                    "SELECT * FROM characters WHERE id = ?",
                    (character_id,)
                )
                row = await cursor.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail="Character not found")

            return self._row_to_character(row)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting character {character_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def create_character(self, character_data: CharacterCreate) -> Character:
        """Create a new character."""
        try:
            character_id = await self._generate_character_id(character_data.name)
            now = datetime.now().isoformat()

            async with aiosqlite.connect(DB_PATH) as conn:
                await conn.execute(
                    """INSERT INTO characters (id, name, voice, system_prompt, image_url, images, is_active, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (character_id, character_data.name, character_data.voice,
                     character_data.system_prompt, character_data.image_url,
                     json.dumps(character_data.images), 1 if character_data.is_active else 0,
                     now, now)
                )
                await conn.commit()

            character = Character(
                id=character_id,
                name=character_data.name,
                voice=character_data.voice,
                system_prompt=character_data.system_prompt,
                image_url=character_data.image_url,
                images=character_data.images,
                is_active=character_data.is_active,
                last_message="",
                created_at=now,
                updated_at=now
            )

            self._character_cache[character_id] = character
            logger.info(f"Created character: {character_id}")
            return character

        except Exception as e:
            logger.error(f"Error creating character: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def update_character(self, character_id: str, character_data: CharacterUpdate) -> Character:
        """Update an existing character."""
        try:
            await self.get_character(character_id)

            updates = []
            params = []

            if character_data.name is not None:
                updates.append("name = ?")
                params.append(character_data.name)
            if character_data.voice is not None:
                updates.append("voice = ?")
                params.append(character_data.voice)
            if character_data.system_prompt is not None:
                updates.append("system_prompt = ?")
                params.append(character_data.system_prompt)
            if character_data.image_url is not None:
                updates.append("image_url = ?")
                params.append(character_data.image_url)
            if character_data.images is not None:
                updates.append("images = ?")
                params.append(json.dumps(character_data.images))
            if character_data.is_active is not None:
                updates.append("is_active = ?")
                params.append(1 if character_data.is_active else 0)
            if character_data.last_message is not None:
                updates.append("last_message = ?")
                params.append(character_data.last_message)

            if not updates:
                raise HTTPException(status_code=400, detail="No fields to update")

            updates.append("updated_at = ?")
            params.append(datetime.now().isoformat())
            params.append(character_id)

            async with aiosqlite.connect(DB_PATH) as conn:
                await conn.execute(
                    f"UPDATE characters SET {', '.join(updates)} WHERE id = ?",
                    params
                )
                await conn.commit()

            character = await self.get_character(character_id)
            self._character_cache[character_id] = character
            logger.info(f"Updated character: {character_id}")
            return character

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating character {character_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def set_character_active(self, character_id: str, is_active: bool) -> Character:
        """Set character active status."""
        return await self.update_character(character_id, CharacterUpdate(is_active=is_active))

    async def delete_character(self, character_id: str) -> bool:
        """Delete a character."""
        try:
            await self.get_character(character_id)

            async with aiosqlite.connect(DB_PATH) as conn:
                await conn.execute("DELETE FROM characters WHERE id = ?", (character_id,))
                await conn.commit()

            if character_id in self._character_cache:
                del self._character_cache[character_id]

            logger.info(f"Deleted character: {character_id}")
            return True

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting character {character_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def search_characters(self, query: str) -> List[Character]:
        """Search characters by name."""
        if self._cache_loaded:
            query_lower = query.lower()
            characters = [c for c in self._character_cache.values() if query_lower in c.name.lower()]
            logger.info(f"Found {len(characters)} characters matching '{query}'")
            return characters

        try:
            async with aiosqlite.connect(DB_PATH) as conn:
                conn.row_factory = aiosqlite.Row
                cursor = await conn.execute(
                    "SELECT * FROM characters WHERE name LIKE ?",
                    (f"%{query}%",)
                )
                rows = await cursor.fetchall()

            characters = [self._row_to_character(row) for row in rows]
            logger.info(f"Found {len(characters)} characters matching '{query}'")
            return characters

        except Exception as e:
            logger.error(f"Error searching characters: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def refresh_character_cache(self):
        """Reload character cache from database."""
        self._character_cache.clear()
        async with aiosqlite.connect(DB_PATH) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute("SELECT * FROM characters")
            rows = await cursor.fetchall()
            for row in rows:
                character = self._row_to_character(row)
                self._character_cache[character.id] = character
        logger.info(f"Refreshed character cache: {len(self._character_cache)} characters")

    ########################################
    ##--        Voice Operations        --##
    ########################################

    async def get_all_voices(self) -> List[Voice]:
        """Get all voices from cache."""
        if self._cache_loaded:
            voices = [v["config"] for v in self._voice_cache.values()]
            logger.info(f"Retrieved {len(voices)} voices from cache")
            return voices

        try:
            async with aiosqlite.connect(DB_PATH) as conn:
                conn.row_factory = aiosqlite.Row
                cursor = await conn.execute("SELECT * FROM voices")
                rows = await cursor.fetchall()

            voices = [self._row_to_voice(row) for row in rows]
            logger.info(f"Retrieved {len(voices)} voices from database")
            return voices

        except Exception as e:
            logger.error(f"Error getting all voices: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def get_voice(self, voice_name: str) -> Voice:
        """Get a specific voice by name."""
        if self._cache_loaded and voice_name in self._voice_cache:
            logger.debug(f"Retrieved voice {voice_name} from cache")
            return self._voice_cache[voice_name]["config"]

        try:
            async with aiosqlite.connect(DB_PATH) as conn:
                conn.row_factory = aiosqlite.Row
                cursor = await conn.execute(
                    "SELECT * FROM voices WHERE voice = ?",
                    (voice_name,)
                )
                row = await cursor.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail="Voice not found")

            voice = self._row_to_voice(row)

            self._voice_cache[voice_name] = {
                "config": voice,
                "audio_tokens": voice.audio_tokens
            }

            logger.info(f"Retrieved voice {voice_name} from database")
            return voice

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting voice {voice_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def create_voice(self, voice_data: VoiceCreate) -> Voice:
        """Create a new voice."""
        try:
            now = datetime.now().isoformat()
            voice_id = str(uuid.uuid4())

            async with aiosqlite.connect(DB_PATH) as conn:
                await conn.execute(
                    """INSERT INTO voices (voice, method, audio_path, text_path, speaker_desc, scene_prompt, id, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (voice_data.voice, voice_data.method, voice_data.audio_path,
                     voice_data.text_path, voice_data.speaker_desc, voice_data.scene_prompt,
                     voice_id, now, now)
                )
                await conn.commit()

            voice = Voice(
                voice=voice_data.voice,
                method=voice_data.method,
                audio_path=voice_data.audio_path,
                text_path=voice_data.text_path,
                speaker_desc=voice_data.speaker_desc,
                scene_prompt=voice_data.scene_prompt,
                audio_tokens=None,
                id=voice_id,
                created_at=now,
                updated_at=now
            )

            self._voice_cache[voice_data.voice] = {
                "config": voice,
                "audio_tokens": None
            }

            logger.info(f"Created voice: {voice.voice}")
            return voice

        except Exception as e:
            logger.error(f"Error creating voice: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def update_voice(self, voice_name: str, voice_data: VoiceUpdate) -> Voice:
        """Update an existing voice."""
        try:
            await self.get_voice(voice_name)

            updates = []
            params = []

            if voice_data.method is not None:
                updates.append("method = ?")
                params.append(voice_data.method)
            if voice_data.audio_path is not None:
                updates.append("audio_path = ?")
                params.append(voice_data.audio_path)
            if voice_data.text_path is not None:
                updates.append("text_path = ?")
                params.append(voice_data.text_path)
            if voice_data.speaker_desc is not None:
                updates.append("speaker_desc = ?")
                params.append(voice_data.speaker_desc)
            if voice_data.scene_prompt is not None:
                updates.append("scene_prompt = ?")
                params.append(voice_data.scene_prompt)
            if voice_data.audio_tokens is not None:
                updates.append("audio_tokens = ?")
                params.append(json.dumps(voice_data.audio_tokens))

            if not updates:
                raise HTTPException(status_code=400, detail="No fields to update")

            updates.append("updated_at = ?")
            params.append(datetime.now().isoformat())
            params.append(voice_name)

            async with aiosqlite.connect(DB_PATH) as conn:
                await conn.execute(
                    f"UPDATE voices SET {', '.join(updates)} WHERE voice = ?",
                    params
                )
                await conn.commit()

            voice = await self.get_voice(voice_name)

            self._voice_cache[voice_name] = {
                "config": voice,
                "audio_tokens": voice.audio_tokens
            }

            logger.info(f"Updated voice: {voice_name}")
            return voice

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating voice {voice_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def delete_voice(self, voice_name: str) -> bool:
        """Delete a voice."""
        try:
            await self.get_voice(voice_name)

            async with aiosqlite.connect(DB_PATH) as conn:
                await conn.execute("DELETE FROM voices WHERE voice = ?", (voice_name,))
                await conn.commit()

            if voice_name in self._voice_cache:
                del self._voice_cache[voice_name]

            logger.info(f"Deleted voice: {voice_name}")
            return True

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting voice {voice_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    def get_cached_audio_tokens(self, voice_name: str) -> Optional[Any]:
        """Get audio tokens from cache if available."""
        if voice_name in self._voice_cache:
            return self._voice_cache[voice_name]["audio_tokens"]
        return None

    def update_cached_audio_tokens(self, voice_name: str, audio_tokens: Any):
        """Update audio tokens in cache and persist to database in background."""
        if voice_name in self._voice_cache:
            self._voice_cache[voice_name]["audio_tokens"] = audio_tokens
            asyncio.create_task(self._persist_audio_tokens(voice_name, audio_tokens))

    async def _persist_audio_tokens(self, voice_name: str, audio_tokens: Any):
        """Background task to persist audio tokens to database."""
        try:
            async with aiosqlite.connect(DB_PATH) as conn:
                await conn.execute(
                    "UPDATE voices SET audio_tokens = ?, updated_at = ? WHERE voice = ?",
                    (json.dumps(audio_tokens), datetime.now().isoformat(), voice_name)
                )
                await conn.commit()
            logger.debug(f"Persisted audio tokens for voice: {voice_name}")
        except Exception as e:
            logger.error(f"Failed to persist audio tokens for {voice_name}: {e}")

    def clear_voice_cache(self):
        """Clear the voice cache."""
        self._voice_cache.clear()
        logger.info("Voice cache cleared")

    async def refresh_voice_cache(self):
        """Reload voice cache from database."""
        self._voice_cache.clear()
        async with aiosqlite.connect(DB_PATH) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute("SELECT * FROM voices")
            rows = await cursor.fetchall()
            for row in rows:
                voice = self._row_to_voice(row)
                self._voice_cache[voice.voice] = {
                    "config": voice,
                    "audio_tokens": voice.audio_tokens
                }
        logger.info(f"Refreshed voice cache: {len(self._voice_cache)} voices")

    ########################################
    ##--    Conversation Operations     --##
    ########################################

    def _generate_conversation_title(self, first_message: Optional[str] = None) -> str:
        """Generate a conversation title."""
        if first_message and first_message.strip():
            title = first_message.strip()[:50]
            if len(first_message.strip()) > 50:
                title += "..."
            return title
        return f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    async def create_conversation(
        self,
        conversation_data: ConversationCreate,
        auto_generate_title: bool = True
    ) -> Conversation:
        """Create a new conversation."""
        try:
            conversation_id = str(uuid.uuid4())
            now = datetime.now().isoformat()

            title = conversation_data.title
            if auto_generate_title and not title:
                title = self._generate_conversation_title()

            async with aiosqlite.connect(DB_PATH) as conn:
                await conn.execute(
                    """INSERT INTO conversations (conversation_id, title, active_characters, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (conversation_id, title, json.dumps(conversation_data.active_characters or []),
                     now, now)
                )
                await conn.commit()

            conversation = Conversation(
                conversation_id=conversation_id,
                title=title,
                active_characters=conversation_data.active_characters or [],
                created_at=now,
                updated_at=now
            )

            logger.info(f"Created conversation {conversation.conversation_id}")
            return conversation

        except Exception as e:
            logger.error(f"Error creating conversation: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def create_conversation_background(self, conversation_data: ConversationCreate) -> str:
        """Create conversation in background, return ID immediately."""
        conversation_id = str(uuid.uuid4())
        asyncio.create_task(self._create_conversation_async(conversation_id, conversation_data))
        return conversation_id

    async def _create_conversation_async(self, conversation_id: str, conversation_data: ConversationCreate):
        """Background task to create conversation."""
        try:
            now = datetime.now().isoformat()
            title = conversation_data.title or self._generate_conversation_title()

            async with aiosqlite.connect(DB_PATH) as conn:
                await conn.execute(
                    """INSERT INTO conversations (conversation_id, title, active_characters, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (conversation_id, title, json.dumps(conversation_data.active_characters or []),
                     now, now)
                )
                await conn.commit()
            logger.debug(f"Background created conversation: {conversation_id}")
        except Exception as e:
            logger.error(f"Background conversation creation failed: {e}")

    async def get_conversation(self, conversation_id: str) -> Conversation:
        """Get a specific conversation by ID."""
        try:
            async with aiosqlite.connect(DB_PATH) as conn:
                conn.row_factory = aiosqlite.Row
                cursor = await conn.execute(
                    "SELECT * FROM conversations WHERE conversation_id = ?",
                    (conversation_id,)
                )
                row = await cursor.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail="Conversation not found")

            return self._row_to_conversation(row)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting conversation {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def get_all_conversations(
        self,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Conversation]:
        """Get all conversations ordered by most recent first."""
        try:
            query = "SELECT * FROM conversations ORDER BY updated_at DESC"
            params = []

            if limit is not None:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])

            async with aiosqlite.connect(DB_PATH) as conn:
                conn.row_factory = aiosqlite.Row
                cursor = await conn.execute(query, params)
                rows = await cursor.fetchall()

            conversations = [self._row_to_conversation(row) for row in rows]
            logger.info(f"Retrieved {len(conversations)} conversations")
            return conversations

        except Exception as e:
            logger.error(f"Error getting all conversations: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def update_conversation(
        self,
        conversation_id: str,
        conversation_data: ConversationUpdate
    ) -> Conversation:
        """Update an existing conversation."""
        try:
            updates = []
            params = []

            if conversation_data.title is not None:
                updates.append("title = ?")
                params.append(conversation_data.title)
            if conversation_data.active_characters is not None:
                updates.append("active_characters = ?")
                params.append(json.dumps(conversation_data.active_characters))

            if not updates:
                raise HTTPException(status_code=400, detail="No fields to update")

            updates.append("updated_at = ?")
            params.append(datetime.now().isoformat())
            params.append(conversation_id)

            async with aiosqlite.connect(DB_PATH) as conn:
                cursor = await conn.execute(
                    f"UPDATE conversations SET {', '.join(updates)} WHERE conversation_id = ?",
                    params
                )
                await conn.commit()

                if cursor.rowcount == 0:
                    raise HTTPException(status_code=404, detail="Conversation not found")

            logger.info(f"Updated conversation {conversation_id}")
            return await self.get_conversation(conversation_id)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating conversation {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def update_conversation_title(self, conversation_id: str, title: str) -> Conversation:
        """Update just the title of a conversation."""
        return await self.update_conversation(conversation_id, ConversationUpdate(title=title))

    async def update_conversation_active_characters(
        self,
        conversation_id: str,
        active_characters: List[Dict[str, Any]]
    ) -> Conversation:
        """Update the active characters in a conversation."""
        return await self.update_conversation(
            conversation_id,
            ConversationUpdate(active_characters=active_characters)
        )

    async def add_character_to_conversation(
        self,
        conversation_id: str,
        character_data: Dict[str, Any]
    ) -> Conversation:
        """Add a character to the conversation's active_characters list."""
        try:
            conversation = await self.get_conversation(conversation_id)

            character_ids = [c.get("id") for c in conversation.active_characters]
            if character_data.get("id") not in character_ids:
                active_characters = conversation.active_characters + [character_data]
                return await self.update_conversation_active_characters(conversation_id, active_characters)

            return conversation

        except Exception as e:
            logger.error(f"Error adding character to conversation {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def remove_character_from_conversation(
        self,
        conversation_id: str,
        character_id: str
    ) -> Conversation:
        """Remove a character from the conversation's active_characters list."""
        try:
            conversation = await self.get_conversation(conversation_id)
            active_characters = [c for c in conversation.active_characters if c.get("id") != character_id]
            return await self.update_conversation_active_characters(conversation_id, active_characters)

        except Exception as e:
            logger.error(f"Error removing character from conversation {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation (messages will be cascade deleted)."""
        try:
            async with aiosqlite.connect(DB_PATH) as conn:
                await conn.execute("PRAGMA foreign_keys = ON")
                cursor = await conn.execute(
                    "DELETE FROM conversations WHERE conversation_id = ?",
                    (conversation_id,)
                )
                await conn.commit()

                if cursor.rowcount == 0:
                    raise HTTPException(status_code=404, detail="Conversation not found")

            logger.info(f"Deleted conversation {conversation_id}")
            return True

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting conversation {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def auto_update_conversation_title(
        self,
        conversation_id: str,
        first_message: str
    ) -> Conversation:
        """Auto-generate and update title from the first message."""
        try:
            conversation = await self.get_conversation(conversation_id)

            if not conversation.title or "Conversation" in conversation.title:
                new_title = self._generate_conversation_title(first_message)
                return await self.update_conversation_title(conversation_id, new_title)

            return conversation

        except Exception as e:
            logger.error(f"Error auto-updating title for conversation {conversation_id}: {e}")
            raise

    ########################################
    ##--       Message Operations       --##
    ########################################

    async def create_message(self, message_data: MessageCreate) -> Message:
        """Create a single message."""
        try:
            message_id = str(uuid.uuid4())
            now = datetime.now().isoformat()

            async with aiosqlite.connect(DB_PATH) as conn:
                await conn.execute(
                    """INSERT INTO messages (message_id, conversation_id, role, name, content, character_id, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (message_id, message_data.conversation_id, message_data.role,
                     message_data.name, message_data.content, message_data.character_id,
                     now, now)
                )
                await conn.commit()

            message = Message(
                message_id=message_id,
                conversation_id=message_data.conversation_id,
                role=message_data.role,
                name=message_data.name,
                content=message_data.content,
                character_id=message_data.character_id,
                created_at=now,
                updated_at=now
            )

            logger.info(f"Created message {message.message_id} in conversation {message.conversation_id}")
            return message

        except Exception as e:
            logger.error(f"Error creating message: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    def create_message_background(self, message_data: MessageCreate) -> str:
        """Create message in background (fire-and-forget). Returns message_id immediately."""
        message_id = str(uuid.uuid4())
        asyncio.create_task(self._create_message_async(message_id, message_data))
        return message_id

    async def _create_message_async(self, message_id: str, message_data: MessageCreate):
        """Background task to create message."""
        try:
            now = datetime.now().isoformat()
            async with aiosqlite.connect(DB_PATH) as conn:
                await conn.execute(
                    """INSERT INTO messages (message_id, conversation_id, role, name, content, character_id, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (message_id, message_data.conversation_id, message_data.role,
                     message_data.name, message_data.content, message_data.character_id,
                     now, now)
                )
                await conn.commit()
            logger.debug(f"Background created message: {message_id}")
        except Exception as e:
            logger.error(f"Background message creation failed for conversation {message_data.conversation_id}: {e}")

    async def create_messages_batch(self, messages: List[MessageCreate]) -> List[Message]:
        """Create multiple messages in a single batch operation."""
        try:
            now = datetime.now().isoformat()
            created_messages = []

            async with aiosqlite.connect(DB_PATH) as conn:
                for msg in messages:
                    message_id = str(uuid.uuid4())
                    await conn.execute(
                        """INSERT INTO messages (message_id, conversation_id, role, name, content, character_id, created_at, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (message_id, msg.conversation_id, msg.role,
                         msg.name, msg.content, msg.character_id, now, now)
                    )
                    created_messages.append(Message(
                        message_id=message_id,
                        conversation_id=msg.conversation_id,
                        role=msg.role,
                        name=msg.name,
                        content=msg.content,
                        character_id=msg.character_id,
                        created_at=now,
                        updated_at=now
                    ))
                await conn.commit()

            logger.info(f"Created {len(created_messages)} messages in batch")
            return created_messages

        except Exception as e:
            logger.error(f"Error creating messages batch: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    def create_messages_batch_background(self, messages: List[MessageCreate]):
        """Create multiple messages in background (fire-and-forget)."""
        asyncio.create_task(self._create_messages_batch_async(messages))

    async def _create_messages_batch_async(self, messages: List[MessageCreate]):
        """Background task to create messages in batch."""
        try:
            now = datetime.now().isoformat()
            async with aiosqlite.connect(DB_PATH) as conn:
                for msg in messages:
                    message_id = str(uuid.uuid4())
                    await conn.execute(
                        """INSERT INTO messages (message_id, conversation_id, role, name, content, character_id, created_at, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (message_id, msg.conversation_id, msg.role,
                         msg.name, msg.content, msg.character_id, now, now)
                    )
                await conn.commit()
            logger.debug(f"Background created {len(messages)} messages in batch")
        except Exception as e:
            logger.error(f"Background batch message creation failed: {e}")

    async def get_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Message]:
        """Get messages for a conversation with optional pagination."""
        try:
            query = "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC"
            params = [conversation_id]

            if limit is not None:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])

            async with aiosqlite.connect(DB_PATH) as conn:
                conn.row_factory = aiosqlite.Row
                cursor = await conn.execute(query, params)
                rows = await cursor.fetchall()

            messages = [self._row_to_message(row) for row in rows]
            logger.info(f"Retrieved {len(messages)} messages for conversation {conversation_id}")
            return messages

        except Exception as e:
            logger.error(f"Error getting messages for conversation {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def get_recent_messages(self, conversation_id: str, n: int = 10) -> List[Message]:
        """Get the last N messages from a conversation."""
        try:
            async with aiosqlite.connect(DB_PATH) as conn:
                conn.row_factory = aiosqlite.Row
                cursor = await conn.execute(
                    """SELECT * FROM messages WHERE conversation_id = ?
                       ORDER BY created_at DESC LIMIT ?""",
                    (conversation_id, n)
                )
                rows = await cursor.fetchall()

            # Reverse to get chronological order
            messages = [self._row_to_message(row) for row in reversed(rows)]
            logger.info(f"Retrieved last {len(messages)} messages for conversation {conversation_id}")
            return messages

        except Exception as e:
            logger.error(f"Error getting recent messages for conversation {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def get_last_message(self, conversation_id: str) -> Optional[Message]:
        """Get the last message from a conversation."""
        try:
            async with aiosqlite.connect(DB_PATH) as conn:
                conn.row_factory = aiosqlite.Row
                cursor = await conn.execute(
                    """SELECT * FROM messages WHERE conversation_id = ?
                       ORDER BY created_at DESC LIMIT 1""",
                    (conversation_id,)
                )
                row = await cursor.fetchone()

            if not row:
                return None

            return self._row_to_message(row)

        except Exception as e:
            logger.error(f"Error getting last message for conversation {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def get_message_count(self, conversation_id: str) -> int:
        """Get the total number of messages in a conversation."""
        try:
            async with aiosqlite.connect(DB_PATH) as conn:
                cursor = await conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                    (conversation_id,)
                )
                row = await cursor.fetchone()

            count = row[0] if row else 0
            logger.info(f"Conversation {conversation_id} has {count} messages")
            return count

        except Exception as e:
            logger.error(f"Error getting message count for conversation {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def delete_message(self, message_id: str) -> bool:
        """Delete a single message."""
        try:
            async with aiosqlite.connect(DB_PATH) as conn:
                await conn.execute("DELETE FROM messages WHERE message_id = ?", (message_id,))
                await conn.commit()

            logger.info(f"Deleted message {message_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting message {message_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def delete_messages_for_conversation(self, conversation_id: str) -> bool:
        """Delete all messages for a conversation."""
        try:
            async with aiosqlite.connect(DB_PATH) as conn:
                await conn.execute(
                    "DELETE FROM messages WHERE conversation_id = ?",
                    (conversation_id,)
                )
                await conn.commit()

            logger.info(f"Deleted messages for conversation {conversation_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting messages for conversation {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


########################################
##--      Module-Level Instance     --##
########################################

db = DatabaseDirector()
