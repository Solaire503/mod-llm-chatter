"""Companion conversation mode.

Gives hand-authored "companion" bots deeper, multi-line,
memory-rich replies in PARTY chat when a real player
speaks to them — distinct from the module's ambient,
one-shot reactions.

A companion is currently detected by a long backstory
(see LLMChatter.Companion.BackstoryMinChars); this is a
pragmatic proxy until an explicit flag column exists.
All "is this a companion?" logic lives behind
is_companion() so that future signal can be swapped in
one place.

This module imports only downward (chatter_shared,
chatter_db, chatter_group_state, chatter_memory) and never
chatter_group or chatter_handler_pipeline, so those two
consumers can import it without a cycle.
"""

import logging
import threading
import time

from chatter_shared import (
    call_llm,
    cleanup_message,
    strip_speaker_prefix,
    insert_chat_message,
    parse_conversation_response,
    parse_single_response,
    calculate_dynamic_delay,
    get_chatter_mode,
    get_language_rule,
    build_bot_identity_from_dict,
)
from chatter_db import (
    get_character_info_by_name,
    get_bot_strategies,
)
from chatter_group_state import (
    _get_recent_chat,
    format_chat_history,
    _store_chat,
)
from chatter_memory import (
    get_all_bot_memories,
    sanitize_memory_for_prompt,
)
from chatter_knowledge import lookup_game_knowledge

logger = logging.getLogger(__name__)

# Defaults mirror the conf template so the feature works
# even if the keys are absent from a stock config.
_DEFAULT_MIN_CHARS = 500
_DEFAULT_MAX_MEMORIES = 20
_DEFAULT_HISTORY_LIMIT = 35
_DEFAULT_MAX_TOKENS = 450
_DEFAULT_MAX_LINES = 4
_DEFAULT_SUPPRESS_SECONDS = 60
_DEFAULT_BEHAVIOR_ENABLE = 1


def _cfg_int(config, key, default):
    """Read an int config value, falling back on error."""
    if config is None:
        return default
    try:
        return int(config.get(key, default))
    except (TypeError, ValueError):
        return default


# ============================================================
# DETECTION
# ============================================================

def _load_backstory(db, bot_guid):
    """Load a bot's persistent backstory, or None."""
    try:
        cursor = db.cursor(dictionary=True)
        cursor.execute(
            "SELECT backstory"
            " FROM llm_bot_identities"
            " WHERE bot_guid = %s",
            (int(bot_guid),),
        )
        row = cursor.fetchone()
        return (row.get('backstory') if row else None) or None
    except Exception:
        logger.error(
            "Companion backstory load failed for bot=%s",
            bot_guid, exc_info=True,
        )
        return None


def is_companion(db, bot_guid, config=None):
    """Return True if this bot is a hand-authored companion.

    Companionship is an explicit flag:
    llm_bot_identities.is_manual = 1, set by the server
    owner. (The old backstory-length proxy is retired —
    auto-generated backstories routinely crossed the
    threshold, giving 47 accidental "companions" on one
    test server.)
    """
    try:
        cursor = db.cursor(dictionary=True)
        cursor.execute(
            "SELECT is_manual FROM llm_bot_identities"
            " WHERE bot_guid = %s",
            (int(bot_guid),),
        )
        row = cursor.fetchone()
        return bool(row and row.get('is_manual'))
    except Exception:
        logger.error(
            "Companion flag check failed for bot=%s",
            bot_guid, exc_info=True,
        )
        return False


def is_significant_for_companion(player_message, bot_name):
    """Heuristic: does this message warrant a deep
    companion reply rather than a quick one-off?

    Trigger companion mode only when the player seems to
    be engaging the companion directly or substantively:
      - mentions the companion by name (case-insensitive)
      - asks a question ("?")
      - is reasonably long (> 60 chars)
    Otherwise the caller falls through to the normal path.
    Deliberately small so it can be refined later (e.g.
    sentiment, addressed-bot signal, conversation streak).
    """
    if not player_message:
        return False
    msg = player_message.strip()
    if len(msg) > 60:
        return True
    if '?' in msg:
        return True
    if bot_name and bot_name.lower() in msg.lower():
        return True
    return False


# ============================================================
# BEHAVIOR CONTEXT (playerbot standing orders)
# ============================================================
# mod-playerbots saves each bot's strategy sets to
# playerbots_db_store when the master issues strategy
# commands. Translating the player-meaningful ones into
# plain language lets a companion talk about what it has
# actually been told to do ("You asked me to wait here").
# Strategies not in this map are AI plumbing (chat, racials,
# default, threat, ...) and are deliberately ignored.

_STRATEGY_DESCRIPTIONS = {
    # movement / stance (chat-shortcut toggles)
    'stay': 'holding position where you were told to wait',
    'follow': 'staying close and following',
    'passive': 'keeping out of fights unless told otherwise',
    'grind': 'roaming free and fighting anything nearby',
    'free': 'moving about on your own judgement',
    'move from group': 'keeping some distance from the group',
    'runaway': 'keeping away from enemies',
    # combat style
    'tank aoe': 'holding the attention of multiple enemies',
    'behind': 'striking from behind',
    'ranged': 'fighting from range',
    'close': 'fighting toe-to-toe',
    'aoe': 'using sweeping attacks on groups',
    'stealth': 'staying hidden when possible',
}


def _build_behavior_lines(strategies):
    """Translate saved strategy sets into prompt lines.

    Returns e.g. ["In a fight: holding the attention of
    multiple enemies", "Out of combat: staying close and
    following"], or [] if nothing player-meaningful is set.
    """
    if not strategies:
        return []
    labels = (
        ('combat', 'In a fight'),
        ('noncombat', 'Out of combat'),
    )
    lines = []
    for key, label in labels:
        described = [
            _STRATEGY_DESCRIPTIONS[name]
            for name in strategies.get(key, [])
            if name in _STRATEGY_DESCRIPTIONS
        ]
        if described:
            lines.append(f"{label}: {'; '.join(described)}")
    return lines


# ============================================================
# COOLDOWN REGISTRY (in-memory)
# ============================================================
# After a companion reply, suppress that bot's idle barks
# and event reactions for a short window so random chatter
# does not stomp on a meaningful conversation. In-memory is
# sufficient: the window is short and need not survive a
# bridge restart; the bridge is single-process with worker
# threads, hence the lock.

_companion_cooldowns = {}
_cooldown_lock = threading.Lock()


def mark_companion_response(group_id, bot_guid, config=None):
    """Start the suppression window for a companion bot."""
    seconds = _cfg_int(
        config, 'LLMChatter.Companion.IdleSuppressSeconds',
        _DEFAULT_SUPPRESS_SECONDS,
    )
    with _cooldown_lock:
        _companion_cooldowns[
            (int(group_id), int(bot_guid))
        ] = time.time() + seconds


def is_companion_cooldown_active(group_id, bot_guid):
    """True if the bot is inside its suppression window."""
    key = (int(group_id), int(bot_guid))
    now = time.time()
    with _cooldown_lock:
        expiry = _companion_cooldowns.get(key)
        if expiry is None:
            return False
        if expiry <= now:
            del _companion_cooldowns[key]
            return False
        return True


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_companion_prompt(
    bot, traits, tone, backstory,
    memories, chat_history,
    player_name, player_message, mode,
    *, max_lines, config, behavior_lines=None,
    knowledge_block=None, extra_context=None,
):
    """Build a rich companion conversation prompt.

    Full backstory (no RNG gate), all shared memories,
    extended chat history, current standing orders (if any),
    and an instruction to speak with depth across up to
    max_lines. No brevity guidelines.
    """
    bot_name = bot['name']
    trait_str = ', '.join([t for t in traits if t])

    parts = [build_bot_identity_from_dict(bot, suffix='.')]
    if trait_str:
        parts.append(f"Your personality: {trait_str}")
    if tone:
        parts.append(f"Your tone: {tone}")

    parts.append("")
    parts.append("<backstory>")
    parts.append(backstory.strip())
    parts.append("</backstory>")

    if behavior_lines:
        parts.append("")
        parts.append("<current_behavior>")
        parts.append(
            "What you're currently doing, per your "
            "standing orders:"
        )
        for line in behavior_lines:
            parts.append(f"  - {line}")
        parts.append("</current_behavior>")
        parts.append(
            "You know what you've been asked to do. Let "
            "it inform your answer when it's relevant -- "
            "for instance if asked why you're waiting "
            "somewhere -- but don't recite your orders "
            "unprompted."
        )

    if memories:
        parts.append("")
        parts.append("<shared_history>")
        parts.append(
            f"Things you and {player_name} have been "
            f"through together (most recent first):"
        )
        for mem in memories:
            parts.append(f"  - {mem}")
        parts.append("</shared_history>")
        parts.append(
            "If none of these memories feel relevant to "
            "this moment, don't force a reference. Let "
            "them inform your familiarity and warmth, "
            "not every line."
        )

    if chat_history:
        parts.append("")
        parts.append("<recent_conversation>")
        parts.append(chat_history.strip())
        parts.append("</recent_conversation>")

    parts.append("")
    parts.append(
        f"{player_name} just said to you: "
        f"\"{player_message}\""
    )

    if knowledge_block:
        parts.append(knowledge_block)

    if extra_context:
        parts.append(extra_context)

    parts.append("")
    parts.append(
        "You are in a real conversation with someone you "
        "know well. Respond naturally, with depth. You "
        "can ask questions, reference your shared history, "
        "express emotion, and build on what they said. "
        "This is not a one-liner reaction -- speak as a "
        "real person would to someone they care about. "
        f"Stay in character as {bot_name}."
    )

    parts.append("")
    parts.append("Rules:")
    parts.append(
        f"- Speak only as {bot_name}, in first person."
    )
    parts.append(
        "- You can include brief physical actions to "
        "express what words don't -- a gesture, a "
        "glance, a touch. Write them naturally in your "
        "message wrapped in asterisks, like *leans "
        "against his shoulder* or *looks away, "
        "smiling*. Use them when the moment calls for "
        "it, not on every line."
    )
    parts.append(
        "- No quotes around your words, no emojis, no "
        "bracketed stage directions."
    )
    parts.append(
        "- Don't claim kills, loot, quests, or trades "
        "you didn't actually do."
    )
    parts.append(
        f"- You may answer across up to {max_lines} "
        f"short lines if you have more to say; use "
        f"fewer if a shorter reply feels right."
    )

    prompt = '\n'.join(parts)

    # Companion-specific JSON footer. Unlike the shared
    # conversation footer, this allows inline *asterisk*
    # gestures in the message, omits the separate action
    # field, and asks for 1..max_lines (not exactly N) so
    # short replies stay short.
    lang_rule = get_language_rule()
    footer = (
        f"\n\nRespond as a JSON array of 1 to {max_lines} "
        f"messages, each shaped like "
        f"{{\"speaker\": \"{bot_name}\", "
        f"\"message\": \"...\"}}. Use a single message for "
        f"a short reply; add more only if you genuinely "
        f"have more to say. Physical actions belong INSIDE "
        f"the \"message\" text, wrapped in *asterisks* "
        f"(e.g. *smiles softly*) -- do not use a separate "
        f"action field. "
        f"JSON rules: double quotes, escape "
        f"quotes/newlines, no trailing commas, no code "
        f"fences. Output ONLY the JSON array, nothing else."
        f"{lang_rule}"
    )
    return prompt + footer


# ============================================================
# ORCHESTRATOR
# ============================================================

def handle_companion_player_msg(
    db, client, config, event_id, group_id,
    bot, traits, stored_tone,
    player_name, player_message,
    extra_context=None,
):
    """Generate and deliver a companion reply.

    Returns True on success. On any failure returns False so
    the caller can fall through to the normal player-message
    path (no dead air).
    """
    try:
        bot_guid = int(bot['guid'])
        bot_name = bot['name']

        backstory = _load_backstory(db, bot_guid)
        if not backstory:
            return False

        # Resolve the player's guid for memory lookup.
        player_guid = None
        player_info = get_character_info_by_name(
            db, player_name,
        )
        if player_info:
            player_guid = int(player_info['guid'])

        # All shared memories (recency-first), sanitized.
        memories = []
        if player_guid:
            max_mem = _cfg_int(
                config, 'LLMChatter.Companion.MaxMemories',
                _DEFAULT_MAX_MEMORIES,
            )
            raw = get_all_bot_memories(
                db, bot_guid, player_guid, limit=max_mem,
            )
            memories = [
                s for s in (
                    sanitize_memory_for_prompt(m)
                    for m in raw
                ) if s
            ]

        # Extended conversation context.
        hist_limit = _cfg_int(
            config, 'LLMChatter.Companion.HistoryLimit',
            _DEFAULT_HISTORY_LIMIT,
        )
        history = _get_recent_chat(
            db, group_id, limit=hist_limit,
        )
        chat_history = format_chat_history(history)

        mode = get_chatter_mode(config)
        max_lines = _cfg_int(
            config, 'LLMChatter.Companion.MaxLines',
            _DEFAULT_MAX_LINES,
        )

        # Standing orders from mod-playerbots, if enabled
        # and available. Empty list = block omitted.
        behavior_lines = []
        if _cfg_int(
            config,
            'LLMChatter.Companion.BehaviorContext.Enable',
            _DEFAULT_BEHAVIOR_ENABLE,
        ):
            behavior_lines = _build_behavior_lines(
                get_bot_strategies(db, bot_guid, config)
            )

        # Ground factual gameplay questions in real
        # world-DB data ('' when not a data question).
        knowledge_block = lookup_game_knowledge(
            client, config, player_message,
        )

        prompt = build_companion_prompt(
            bot, traits, stored_tone, backstory,
            memories, chat_history,
            player_name, player_message, mode,
            max_lines=max_lines, config=config,
            behavior_lines=behavior_lines,
            knowledge_block=knowledge_block,
            extra_context=extra_context,
        )

        max_tokens = _cfg_int(
            config, 'LLMChatter.Companion.MaxTokens',
            _DEFAULT_MAX_TOKENS,
        )
        response = call_llm(
            client, prompt, config,
            max_tokens_override=max_tokens,
            context=(
                f"grp-companion:#{event_id}:{bot_name}"
            ),
            label='group_companion_msg',
        )
        if not response:
            return False

        # Multi-line: companion is the only speaker, so
        # parse the conversation array with a single name.
        entries = parse_conversation_response(
            response, [bot_name],
        )
        if not entries:
            # Fallback: a single (non-array) message.
            parsed = parse_single_response(response)
            single = cleanup_message(
                strip_speaker_prefix(
                    parsed.get('message', ''), bot_name,
                ),
                preserve_leading_action=True,
            )
            if single:
                entries = [{
                    'name': bot_name,
                    'message': single,
                    'emote': parsed.get('emote'),
                }]
        if not entries:
            return False

        entries = entries[:max_lines]
        cumulative = 0.0
        prev_len = len(player_message)
        delivered = 0
        for i, entry in enumerate(entries):
            msg = cleanup_message(
                strip_speaker_prefix(
                    entry.get('message', ''), bot_name,
                ),
                preserve_leading_action=True,
            )
            if not msg:
                continue
            if len(msg) > 255:
                msg = msg[:252] + "..."
            cumulative += calculate_dynamic_delay(
                len(msg), config,
                prev_message_length=prev_len,
                responsive=True,
            )
            insert_chat_message(
                db, bot_guid, bot_name, msg,
                channel='party',
                delay_seconds=cumulative,
                event_id=event_id,
                sequence=i,
                emote=entry.get('emote'),
                config=config,
                group_id=group_id,
                delivery_policy='responsive',
                delivery_reason='bot_group_companion_msg',
            )
            _store_chat(
                db, group_id, bot_guid,
                bot_name, True, msg,
            )
            prev_len = len(msg)
            delivered += 1

        if not delivered:
            return False

        # Start the suppression window so ambient barks do
        # not stomp on the conversation we just started.
        mark_companion_response(group_id, bot_guid, config)
        return True
    except Exception:
        logger.error(
            "Companion handler failed for event=%s bot=%s",
            event_id, bot.get('name'),
            exc_info=True,
        )
        return False
