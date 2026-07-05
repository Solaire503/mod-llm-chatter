"""Tier 1: natural-language playerbot commands.

When a player says something actionable to a bot in party
chat ("hang back and keep me healed", "Doran stop
pulling"), classify the intent into a playerbot command
string and queue it in `llm_chatter_commands`. The C++
poller validates it against a hard allowlist and injects
it via PlayerbotAI::HandleCommand with the requesting
player as sender — so the bot answers in character AND
does the thing, from one player utterance.

Defense in depth: the classifier output is validated
against ALLOWED_COMMANDS here before insert, and the C++
poller re-validates against its own configured allowlist.
False negatives are free (the bot just chats); false
positives annoy — the classifier is instructed to prefer
no command.

Consumer: chatter_group.process_group_player_msg_event
(lazy import there). This module imports only downward
(chatter_llm, chatter_db).
"""

import json
import logging
import re

from chatter_llm import quick_llm_analyze

logger = logging.getLogger(__name__)

_DEFAULT_EXPIRES_SECONDS = 30

# Mirror of the C++ allowlist (LLMChatter.Commands.
# Allowlist). Entries ending in ' ' are prefix rules.
ALLOWED_COMMANDS = (
    'follow', 'stay', 'flee', 'grind', 'passive',
    'attack', 'reset', 'los', 'formation',
    'co ', 'nc ',
)

# Strategy names the classifier may use inside co/nc
# edits; anything else is stripped at validation.
# NB: dash placed last in the class — [+-~] would be a
# character RANGE covering the whole alphabet.
_STRATEGY_TOKEN = re.compile(
    r'^[+~-][a-z][a-z ]{0,24}$'
)


def _cfg_int(config, key, default):
    """Read an int config value, falling back on error."""
    if config is None:
        return default
    try:
        return int(config.get(key, default))
    except (TypeError, ValueError):
        return default


def _cfg_enabled(config):
    """Shared master switch with the C++ poller."""
    return _cfg_int(
        config, 'LLMChatter.Commands.Enable', 1
    ) == 1


# ============================================================
# VALIDATION
# ============================================================

def validate_command(command):
    """Return a normalized command string or None.

    Mirrors the C++ matcher: exact match on plain
    entries, prefix match (with a required argument)
    on entries ending in ' '. co/nc arguments are
    additionally token-checked so the LLM can't smuggle
    arbitrary text through the prefix rule.
    """
    if not command or not isinstance(command, str):
        return None
    cmd = ' '.join(command.strip().lower().split())
    if not cmd or len(cmd) > 120:
        return None
    for entry in ALLOWED_COMMANDS:
        if entry.endswith(' '):
            if cmd.startswith(entry) \
                    and len(cmd) > len(entry):
                args = cmd[len(entry):]
                tokens = [
                    t.strip()
                    for t in args.split(',')
                ]
                if all(
                    _STRATEGY_TOKEN.match(t)
                    for t in tokens if t
                ) and any(t for t in tokens):
                    return cmd
                return None
        elif cmd == entry:
            return cmd
    return None


# ============================================================
# INTENT CLASSIFICATION
# ============================================================

def classify_command_intent(client, config,
                            player_message, bot_name):
    """Map a player's message to a playerbot command.

    Returns a validated command string, or None when the
    message isn't an actionable request (the common
    case).
    """
    prompt = (
        f"A player in a World of Warcraft party said "
        f"to their companion {bot_name}:\n"
        f"\"{player_message}\"\n\n"
        "Is this a DIRECT REQUEST for the companion to "
        "change its behavior right now? If so, map it "
        "to exactly one command from this list:\n"
        "- follow    (come with me / stick close / "
        "come back)\n"
        "- stay      (wait here / hold position / "
        "don't move)\n"
        "- flee      (run / retreat / get out)\n"
        "- grind     (go fight stuff on your own "
        "nearby)\n"
        "- passive   (stop attacking / don't engage / "
        "stand down)\n"
        "- attack    (attack my target / focus my "
        "target)\n"
        "- reset     (snap out of it / reset yourself)\n"
        "- co +X or co -X  (combat behavior edit; X "
        "in: tank aoe, behind, ranged, close, aoe)\n"
        "- nc +X or nc -X  (non-combat behavior edit; "
        "X in: stay, follow)\n\n"
        "Respond with ONLY a JSON object:\n"
        "{\"command\": \"<command>\"} or "
        "{\"command\": null}\n"
        "Rules: questions, banter, storytelling, "
        "combat chatter, compliments, and vague talk "
        "are NOT requests — use null. Past or "
        "hypothetical actions are NOT requests — use "
        "null. When in doubt, use null."
    )
    raw = quick_llm_analyze(
        client, config, prompt,
        max_tokens=40,
        label='command_intent',
    )
    if not raw:
        return None
    try:
        start = raw.find('{')
        end = raw.rfind('}')
        if start < 0 or end <= start:
            return None
        data = json.loads(raw[start:end + 1])
    except Exception:
        return None
    return validate_command(data.get('command'))


# ============================================================
# QUEUE + ACKNOWLEDGMENT
# ============================================================

def insert_bot_command(db, bot_guid, bot_name,
                       requester_guid, requester_name,
                       command, source_event_id=None,
                       config=None):
    """Queue a command row for the C++ poller.

    Returns True on success. expires_at bounds staleness
    if the poller is off or the world is lagging.
    """
    try:
        expires = _cfg_int(
            config,
            'LLMChatter.Commands.ExpiresSeconds',
            _DEFAULT_EXPIRES_SECONDS,
        )
        cursor = db.cursor()
        cursor.execute(
            "INSERT INTO llm_chatter_commands "
            "(bot_guid, bot_name, requester_guid, "
            "requester_name, command, source_event_id, "
            "expires_at) VALUES "
            "(%s, %s, %s, %s, %s, %s, "
            "NOW() + INTERVAL %s SECOND)",
            (
                int(bot_guid), bot_name,
                int(requester_guid), requester_name,
                command, source_event_id, expires,
            ),
        )
        db.commit()
        logger.info(
            "[COMMAND] queued '%s' for bot=%s "
            "requester=%s event=%s",
            command, bot_name, requester_name,
            source_event_id,
        )
        return True
    except Exception:
        logger.error(
            "Command insert failed for bot=%s "
            "cmd='%s'", bot_name, command,
            exc_info=True,
        )
        return False


def build_command_ack_block(command, player_name):
    """Prompt block: the bot is EXECUTING the request.

    Appended to whichever reply path runs, so the
    spoken acknowledgment matches the real action.
    """
    return (
        f"\n\nYou are ALREADY DOING what {player_name} "
        f"just asked (internally: \"{command}\") -- it "
        f"is happening right now. Acknowledge it "
        f"briefly, in your own voice and character. "
        f"Don't describe game mechanics or repeat the "
        f"command word; just respond like someone "
        f"complying (or complying while grumbling, if "
        f"that's your nature)."
    )


# ============================================================
# ORCHESTRATOR (called from the party-message handler)
# ============================================================

def maybe_queue_player_command(db, client, config,
                               event_id, bot_guid,
                               bot_name, requester_guid,
                               requester_name,
                               player_message):
    """Full pipeline: classify -> validate -> queue.

    Returns the ack prompt block ('' when no command).
    Never raises.
    """
    try:
        if not _cfg_enabled(config):
            return ''
        if not player_message or not requester_guid:
            return ''

        command = classify_command_intent(
            client, config, player_message, bot_name,
        )
        if not command:
            return ''

        if not insert_bot_command(
            db, bot_guid, bot_name,
            requester_guid, requester_name,
            command, source_event_id=event_id,
            config=config,
        ):
            return ''

        return build_command_ack_block(
            command, requester_name,
        )
    except Exception:
        logger.error(
            "Command pipeline failed for event=%s",
            event_id, exc_info=True,
        )
        return ''
