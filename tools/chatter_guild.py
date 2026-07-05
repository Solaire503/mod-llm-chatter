"""Guild chat replies.

Handles `player_guild_msg` events queued by the C++
guild-chat hook: a real player spoke in guild chat, and
an online bot guild-mate may reply in the guild channel.

Reply selection: random online RNDBOT-account guild
member, gated by a per-guild chance roll and an
in-memory per-guild cooldown so guild chat stays a
conversation, not a chorus. Knowledge grounding applies —
guildies ask "who drops X" constantly.

History: C++ stores the player's line on capture; this
handler stores the bot's reply after queueing it, so
`llm_guild_chat_history` holds both sides.
"""

import logging
import random
import threading
import time

from chatter_shared import (
    parse_extra_data,
    run_single_reaction,
    build_bot_identity_from_dict,
    get_class_name,
    get_race_name,
    get_gender_label,
    calculate_dynamic_delay,
)
from chatter_db import fail_event
from chatter_group_state import _mark_event
from chatter_knowledge import lookup_game_knowledge
from chatter_llm import quick_llm_analyze

logger = logging.getLogger(__name__)

_DEFAULT_REPLY_CHANCE = 60
_DEFAULT_COOLDOWN_SECONDS = 45
_DEFAULT_HISTORY_LIMIT = 15

_guild_last_reply = {}
_guild_lock = threading.Lock()

# Addon handshake spam that leaks through as guild
# "chat" (until the C++ LANG_ADDON filter is compiled
# in): tab-separated payloads and known prefixes.
_ADDON_PREFIXES = (
    'dbm', 'elvui_', 'bigwigs', 'questie',
    'details!', 'atlasloot', 'healbot',
)


def _is_addon_payload(message):
    """True for addon protocol traffic, not speech.

    Tabs are the strong signal. Bare prefixes only
    count for space-free payloads so real speech like
    "dbm saved us last night" isn't eaten.
    """
    if not message:
        return True
    if '\t' in message:
        return True
    if ' ' in message:
        return False
    lowered = message.lower()
    return any(
        lowered.startswith(p) for p in _ADDON_PREFIXES
    )


def _cfg_int(config, key, default):
    """Read an int config value, falling back on error."""
    if config is None:
        return default
    try:
        return int(config.get(key, default))
    except (TypeError, ValueError):
        return default


def _cooldown_ok(guild_id, config):
    """Check and claim the per-guild reply cooldown."""
    seconds = _cfg_int(
        config, 'LLMChatter.GuildChatter.ReplyCooldown',
        _DEFAULT_COOLDOWN_SECONDS,
    )
    now = time.time()
    with _guild_lock:
        if now - _guild_last_reply.get(
            int(guild_id), 0
        ) < seconds:
            return False
        _guild_last_reply[int(guild_id)] = now
        return True


def _pick_guild_bot(db, guild_id, exclude_guid=0):
    """Random online bot member of the guild, or None.

    RNDBOT-account members only (same detection the
    rest of the bridge uses).
    """
    cursor = db.cursor(dictionary=True)
    cursor.execute(
        "SELECT c.guid, c.name, c.class, c.race, "
        "c.level, c.gender "
        "FROM guild_member gm "
        "JOIN characters c ON c.guid = gm.guid "
        "JOIN acore_auth.account a "
        "  ON c.account = a.id "
        "WHERE gm.guildid = %s AND c.online = 1 "
        "  AND c.guid != %s "
        "  AND a.username LIKE 'RNDBOT%%' "
        "ORDER BY RAND() LIMIT 1",
        (int(guild_id), int(exclude_guid)),
    )
    return cursor.fetchone()


def get_guild_chat_history(
    db, guild_id, limit=_DEFAULT_HISTORY_LIMIT
):
    """Recent guild chat rows, oldest first."""
    cursor = db.cursor(dictionary=True)
    cursor.execute(
        "SELECT speaker_name, is_bot, message "
        "FROM llm_guild_chat_history "
        "WHERE guild_id = %s "
        "ORDER BY id DESC LIMIT %s",
        (int(guild_id), limit),
    )
    return list(reversed(cursor.fetchall()))


def store_guild_chat(db, guild_id, speaker_name,
                     is_bot, message):
    """Store a guild line + prune per guild."""
    cursor = db.cursor()
    cursor.execute(
        "INSERT INTO llm_guild_chat_history "
        "(guild_id, speaker_name, is_bot, message) "
        "VALUES (%s, %s, %s, %s)",
        (
            int(guild_id), speaker_name,
            1 if is_bot else 0, message[:500],
        ),
    )
    db.commit()
    cursor.execute(
        "DELETE FROM llm_guild_chat_history "
        "WHERE guild_id = %s AND id NOT IN ("
        "  SELECT id FROM ("
        "    SELECT id FROM llm_guild_chat_history "
        "    WHERE guild_id = %s "
        "    ORDER BY id DESC LIMIT 50"
        "  ) AS keep"
        ")",
        (int(guild_id), int(guild_id)),
    )
    db.commit()


# ============================================================
# GUILD CULTURE (per-guild personality layer)
# ============================================================
# Every guild develops a chat culture. The first time a
# guild speaks, we generate one from its NAME (a guild
# called <GG NO RE> talks differently than <Disciples of
# the Night>) and store it. Server owners can edit the row
# freely — an existing row is never overwritten.

_culture_cache = {}
_culture_lock = threading.Lock()
_CULTURE_TTL = 600


def get_guild_culture(db, client, config,
                      guild_id, guild_name):
    """Culture text for a guild, or ''.

    Cached 10 min; auto-generated from the guild name on
    first encounter when CultureAutoGenerate is on.
    """
    if not _cfg_int(
        config, 'LLMChatter.GuildChatter.CultureEnable', 1
    ):
        return ''
    now = time.time()
    with _culture_lock:
        hit = _culture_cache.get(int(guild_id))
        if hit and now - hit[1] < _CULTURE_TTL:
            return hit[0]
    try:
        cursor = db.cursor(dictionary=True)
        cursor.execute(
            "SELECT culture FROM llm_guild_culture "
            "WHERE guild_id = %s",
            (int(guild_id),),
        )
        row = cursor.fetchone()
        culture = row['culture'] if row else None

        if culture is None and _cfg_int(
            config,
            'LLMChatter.GuildChatter.CultureAutoGenerate',
            1,
        ) and guild_name:
            culture = _generate_culture(
                client, config, guild_name,
            )
            if culture:
                cursor2 = db.cursor()
                cursor2.execute(
                    "INSERT IGNORE INTO "
                    "llm_guild_culture "
                    "(guild_id, guild_name, culture, "
                    "auto_generated) "
                    "VALUES (%s, %s, %s, 1)",
                    (
                        int(guild_id), guild_name,
                        culture,
                    ),
                )
                db.commit()
                logger.info(
                    "[GUILD] culture generated for "
                    "<%s>: %s",
                    guild_name, culture[:80],
                )
        culture = culture or ''
        with _culture_lock:
            _culture_cache[int(guild_id)] = (
                culture, now,
            )
        return culture
    except Exception:
        logger.error(
            "Guild culture load failed for guild=%s",
            guild_id, exc_info=True,
        )
        return ''


def _generate_culture(client, config, guild_name):
    """One-time culture generation from the guild name."""
    prompt = (
        f"A World of Warcraft guild is named "
        f"\"{guild_name}\".\n"
        "Invent this guild's chat culture in 1-2 "
        "punchy sentences, written as instructions "
        "for how its members talk in guild chat. "
        "Infer the vibe from the name: a tryhard "
        "raiding name means meta-obsessed banter and "
        "friendly toxicity; a roleplay-flavored name "
        "means in-character speech and dramatics; a "
        "silly name means memes and chaos. Be "
        "specific and characterful, not generic. "
        "Output ONLY the instruction sentences, no "
        "preamble, no quotes."
    )
    raw = quick_llm_analyze(
        client, config, prompt,
        max_tokens=120,
        label='guild_culture_gen',
    )
    if not raw:
        return None
    culture = ' '.join(raw.strip().split())
    culture = culture.strip('"\'')
    if not (10 <= len(culture) <= 400):
        return None
    return culture


# -- Culture bleed: any speech surface can ask for a
# -- bot's guild-culture line. Bot->guild membership is
# -- cached alongside the culture itself.

_bot_guild_cache = {}


def get_bot_culture_line(db, client, config, bot_guid):
    """Culture prompt line for a bot's guild, or ''.

    Used OUTSIDE guild chat (General, party, proximity,
    whispers, broadcasts) so guild personality bleeds
    into the world. Gated by GuildCulture.Bleed.
    """
    try:
        if not _cfg_int(
            config, 'LLMChatter.GuildCulture.Bleed', 1
        ):
            return ''
        now = time.time()
        with _culture_lock:
            hit = _bot_guild_cache.get(int(bot_guid))
        if hit and now - hit[2] < _CULTURE_TTL:
            guild_id, guild_name = hit[0], hit[1]
        else:
            cursor = db.cursor(dictionary=True)
            cursor.execute(
                "SELECT gm.guildid, g.name "
                "FROM guild_member gm "
                "JOIN guild g ON g.guildid = gm.guildid "
                "WHERE gm.guid = %s",
                (int(bot_guid),),
            )
            row = cursor.fetchone()
            guild_id = int(row['guildid']) if row else 0
            guild_name = row['name'] if row else ''
            with _culture_lock:
                _bot_guild_cache[int(bot_guid)] = (
                    guild_id, guild_name, now,
                )
        if not guild_id:
            return ''
        culture = get_guild_culture(
            db, client, config, guild_id, guild_name,
        )
        if not culture:
            return ''
        return (
            f"\nYou belong to the guild <{guild_name}>,"
            f" and its culture colors how you talk "
            f"everywhere, not just guild chat: {culture}"
        )
    except Exception:
        logger.error(
            "Culture bleed lookup failed for bot=%s",
            bot_guid, exc_info=True,
        )
        return ''


def get_named_culture_line(db, client, config,
                           bot_guid, bot_name):
    """Third-person culture line for multi-speaker
    prompts ('X belongs to <G>: ...'), or ''."""
    line = get_bot_culture_line(
        db, client, config, bot_guid,
    )
    if not line:
        return ''
    # Reuse the cached lookup; just re-voice it.
    with _culture_lock:
        hit = _bot_guild_cache.get(int(bot_guid))
    guild_name = hit[1] if hit else 'their guild'
    culture = line.split(': ', 1)[-1]
    return (
        f"\n{bot_name} belongs to <{guild_name}> -- "
        f"let it color their lines: {culture}"
    )


def _load_identity(db, bot_guid):
    """Persistent traits/tone, or {}."""
    try:
        cursor = db.cursor(dictionary=True)
        cursor.execute(
            "SELECT trait1, trait2, trait3, tone "
            "FROM llm_bot_identities "
            "WHERE bot_guid = %s",
            (int(bot_guid),),
        )
        return cursor.fetchone() or {}
    except Exception:
        return {}


def _build_guild_prompt(
    bot, identity, guild_name, player_name,
    player_message, history_text, knowledge_block,
    culture='',
):
    """Build the guild-chat reply prompt."""
    bot_name = bot['name']
    traits = [
        t for t in (
            identity.get('trait1'),
            identity.get('trait2'),
            identity.get('trait3'),
        ) if t
    ]

    parts = [build_bot_identity_from_dict(bot, suffix='.')]
    if traits:
        parts.append(
            f"Your personality: {', '.join(traits)}"
        )
    if identity.get('tone'):
        parts.append(f"Your tone: {identity['tone']}")
    if guild_name:
        parts.append(
            f"You are a member of the guild "
            f"<{guild_name}>; {player_name} is a "
            f"guildmate."
        )
    if culture:
        parts.append(
            f"Your guild's chat culture (let it "
            f"color your voice on top of your own "
            f"personality): {culture}"
        )

    if history_text:
        parts.append("")
        parts.append("<guild_chat>")
        parts.append("Recent guild chat:")
        parts.append(history_text)
        parts.append("</guild_chat>")

    parts.append("")
    parts.append(
        f"{player_name} just said in guild chat: "
        f"\"{player_message}\""
    )

    if knowledge_block:
        parts.append(knowledge_block)

    parts.append("")
    parts.append("Rules:")
    parts.append(
        f"- Speak only as {bot_name}, in first person."
    )
    parts.append(
        "- Guild chat is relaxed and familiar -- "
        "these people know each other. React to "
        "what was said; banter is welcome."
    )
    parts.append(
        "- Keep it brief: one casual line, under "
        "180 characters."
    )
    parts.append(
        "- Never claim someone said something "
        "unless it appears in the chat above."
    )
    parts.append(
        "- Don't claim kills, loot, quests, or "
        "trades you didn't actually do."
    )
    parts.append("")
    parts.append(
        "Respond with ONLY a JSON object: "
        "{\"message\": \"...\"}. Double quotes, no "
        "code fences, nothing else."
    )
    return '\n'.join(parts)


def process_guild_player_msg_event(db, client, config, event):
    """Reply to a real player's guild-chat message."""
    event_id = event['id']
    extra_data = parse_extra_data(
        event.get('extra_data'),
        event_id, 'player_guild_msg',
    )
    if not extra_data:
        _mark_event(db, event_id, 'skipped')
        return False

    try:
        guild_id = int(extra_data.get('guild_id', 0))
        guild_name = extra_data.get('guild_name', '')
        player_name = extra_data.get(
            'player_name', 'someone'
        )
        player_message = extra_data.get(
            'player_message', ''
        )
        if not guild_id or not player_message:
            _mark_event(db, event_id, 'skipped')
            return False

        if _is_addon_payload(player_message):
            _mark_event(db, event_id, 'skipped')
            return False

        chance = _cfg_int(
            config,
            'LLMChatter.GuildChatter.ReplyChance',
            _DEFAULT_REPLY_CHANCE,
        )
        # Questions get grounded answers; don't
        # leave a direct question hanging on a bad
        # roll.
        is_question = '?' in player_message
        if not is_question \
                and random.randint(1, 100) > chance:
            _mark_event(db, event_id, 'skipped')
            return False

        # Pick the bot BEFORE claiming the cooldown so
        # a no-bots-online pass doesn't burn the
        # window and mute the next real reply.
        row = _pick_guild_bot(db, guild_id)
        if not row:
            _mark_event(db, event_id, 'skipped')
            return False

        if not _cooldown_ok(guild_id, config):
            _mark_event(db, event_id, 'skipped')
            return False

        bot = {
            'guid': int(row['guid']),
            'name': row['name'],
            'class': get_class_name(row['class']),
            'race': get_race_name(row['race']),
            'level': int(row['level']),
            'gender': get_gender_label(
                row.get('gender', 0)
            ),
        }
        identity = _load_identity(db, bot['guid'])

        knowledge_block = lookup_game_knowledge(
            client, config, player_message,
        )

        culture = get_guild_culture(
            db, client, config, guild_id, guild_name,
        )

        history = get_guild_chat_history(
            db, guild_id,
            limit=_cfg_int(
                config,
                'LLMChatter.GuildChatter.HistoryLimit',
                _DEFAULT_HISTORY_LIMIT,
            ),
        )
        history_text = '\n'.join(
            f"  {h['speaker_name']}"
            f"{'' if h['is_bot'] else ' (player)'}: "
            f"{h['message']}"
            for h in history
            if not _is_addon_payload(h['message'])
        )

        prompt = _build_guild_prompt(
            bot, identity, guild_name, player_name,
            player_message, history_text,
            knowledge_block, culture=culture,
        )

        result = run_single_reaction(
            db, client, config,
            prompt=prompt,
            speaker_name=bot['name'],
            bot_guid=bot['guid'],
            channel='guild',
            delay_seconds=min(
                calculate_dynamic_delay(
                    40, config,
                    prev_message_length=len(
                        player_message
                    ),
                    responsive=True,
                ),
                6.0,
            ),
            event_id=event_id,
            allow_emote_fallback=False,
            context=(
                f"guild:#{event_id}:{bot['name']}"
            ),
            label='guild_player_msg',
            delivery_reason='player_guild_msg',
        )
        if not result['ok']:
            _mark_event(db, event_id, 'skipped')
            return False

        store_guild_chat(
            db, guild_id, bot['name'], True,
            result['message'],
        )
        logger.info(
            "[GUILD] reply | guild=%s bot=%s",
            guild_id, bot['name'],
        )
        _mark_event(db, event_id, 'completed')
        return True
    except Exception:
        fail_event(
            db, event_id, 'player_guild_msg',
            'handler error',
        )
        return False
