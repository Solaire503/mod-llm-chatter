"""Contextual broadcasting to the General channel.

When something notable happens to a group bot (boss/rare
kill, a death, a level-up, a quest completion, an
achievement, a wipe), the bot may also mention it in its
zone's General channel — so zone chat reflects what bots
are actually experiencing, and General replies have
grounded history to draw on.

Called from run_group_handler after the party reaction
succeeds. Broadcasting is garnish, never load-bearing:
every failure path returns quietly and the group event
still completes.

Broadcast lines are stored in llm_general_chat_history
(the same store the General reply path reads), which is
the point of the feature: bots replying in General can
reference what other bots just said happened to them.

This module imports only downward (chatter_shared,
chatter_db); the general-history helpers are lazily
imported from chatter_general to keep the import graph
acyclic. Consumers (chatter_handler_pipeline) lazy-import
this module, mirroring the chatter_companion pattern.
"""

import logging
import random
import threading
import time

from chatter_shared import (
    append_json_instruction,
    build_bot_identity_from_dict,
    get_zone_flavor,
    run_single_reaction,
)

logger = logging.getLogger(__name__)

_DEFAULT_ENABLE = 1
_DEFAULT_BOT_COOLDOWN = 600
_DEFAULT_ZONE_GAP = 120
_DEFAULT_HISTORY_LINES = 6

# Per event type: (config chance suffix, default chance %).
_ELIGIBLE = {
    'bot_group_kill': ('Kill', 25),
    'bot_group_death': ('Death', 15),
    'bot_group_levelup': ('Levelup', 30),
    'bot_group_quest_complete': ('QuestComplete', 10),
    'bot_group_achievement': ('Achievement', 25),
    'bot_group_wipe': ('Wipe', 25),
}


def _cfg_int(config, key, default):
    """Read an int config value, falling back on error."""
    if config is None:
        return default
    try:
        return int(config.get(key, default))
    except (TypeError, ValueError):
        return default


# ============================================================
# RATE LIMITING (in-memory)
# ============================================================
# Two gates so General never floods: a per-bot cooldown
# (one broadcast per bot per window) and a per-zone gap
# (minimum spacing between any two broadcasts in a zone).
# In-memory is sufficient — losing the window on a bridge
# restart is harmless.

_bot_last = {}
_zone_last = {}
_rate_lock = threading.Lock()


def _rate_limit_ok(bot_guid, zone_id, config):
    """Check and, if allowed, claim both rate windows."""
    bot_cd = _cfg_int(
        config,
        'LLMChatter.Broadcast.BotCooldownSeconds',
        _DEFAULT_BOT_COOLDOWN,
    )
    zone_gap = _cfg_int(
        config,
        'LLMChatter.Broadcast.ZoneGapSeconds',
        _DEFAULT_ZONE_GAP,
    )
    now = time.time()
    with _rate_lock:
        if now - _bot_last.get(int(bot_guid), 0) < bot_cd:
            return False
        if now - _zone_last.get(int(zone_id), 0) < zone_gap:
            return False
        _bot_last[int(bot_guid)] = now
        _zone_last[int(zone_id)] = now
        return True


# ============================================================
# FACT BUILDERS
# ============================================================
# Each turns the handler ctx into a second-person factual
# sentence for the prompt, or None to skip. Defensive:
# every field read has a fallback because some handlers
# reach the pipeline via pre-parsed conversation branches.


def _field(ctx, name, default=None):
    """Read a field from ctx, falling back to extra_data."""
    val = ctx.get(name)
    if val is None or val == '':
        val = (ctx.get('extra_data') or {}).get(
            name, default
        )
    return val if val not in (None, '') else default


def _fact_kill(ctx):
    if not (
        _field(ctx, 'is_boss') or _field(ctx, 'is_rare')
    ):
        return None
    name = _field(ctx, 'creature_name')
    if not name:
        return None
    kind = 'boss' if _field(ctx, 'is_boss') else (
        'rare creature'
    )
    return (
        f"Your group just brought down {name}, "
        f"a {kind}."
    )


def _fact_death(ctx):
    dead = _field(ctx, 'dead_name')
    if not dead:
        return None
    killer = _field(ctx, 'killer_name')
    if killer:
        return (
            f"Your groupmate {dead} just got killed "
            f"by {killer}."
        )
    return f"Your groupmate {dead} just died."


def _fact_levelup(ctx):
    leveler = _field(ctx, 'leveler_name')
    level = _field(ctx, 'new_level')
    if not leveler or not level:
        return None
    if leveler == ctx.get('bot_name'):
        return f"You just reached level {level}."
    return (
        f"{leveler} in your group just reached "
        f"level {level}."
    )


def _fact_quest_complete(ctx):
    quest = _field(ctx, 'quest_name')
    if not quest:
        return None
    completer = _field(ctx, 'completer_name')
    if completer and completer != ctx.get('bot_name'):
        return (
            f"Your group just finished the quest "
            f"\"{quest}\" with {completer}."
        )
    return f"You just finished the quest \"{quest}\"."


def _fact_achievement(ctx):
    achievement = _field(ctx, 'achievement_name')
    if not achievement:
        return None
    achiever = _field(ctx, 'achiever_name')
    if achiever and achiever != ctx.get('bot_name'):
        return (
            f"{achiever} in your group just earned the "
            f"achievement \"{achievement}\"."
        )
    return (
        f"You just earned the achievement "
        f"\"{achievement}\"."
    )


def _fact_wipe(ctx):
    killer = _field(ctx, 'killer_name')
    if killer:
        return (
            f"Your entire group just got wiped out "
            f"by {killer}."
        )
    return "Your entire group just got wiped out."


_FACT_BUILDERS = {
    'bot_group_kill': _fact_kill,
    'bot_group_death': _fact_death,
    'bot_group_levelup': _fact_levelup,
    'bot_group_quest_complete': _fact_quest_complete,
    'bot_group_achievement': _fact_achievement,
    'bot_group_wipe': _fact_wipe,
}


# ============================================================
# PROMPT
# ============================================================

def _build_broadcast_prompt(
    ctx, fact, zone_id, history_text,
):
    """Build the General-channel broadcast prompt."""
    bot = ctx['bot']
    traits = ctx.get('traits') or []
    trait_str = ', '.join([t for t in traits if t])

    parts = [build_bot_identity_from_dict(bot, suffix='.')]
    if trait_str:
        parts.append(f"Your personality: {trait_str}")
    tone = ctx.get('stored_tone')
    if tone:
        parts.append(f"Your tone: {tone}")

    flavor = get_zone_flavor(zone_id)
    if flavor:
        parts.append(f"Where you are: {flavor}")

    parts.append("")
    parts.append(fact)

    if history_text:
        parts.append(history_text)

    parts.append("")
    parts.append(
        "Mention this in the zone's General chat "
        "channel the way a real player would -- a "
        "brag, a groan, a warning, or a laugh, in "
        "your own voice. You are talking to the "
        "whole zone, not your party."
    )
    parts.append("")
    parts.append("Rules:")
    parts.append("- One casual line, under 150 characters.")
    parts.append(
        "- React to what just happened; don't "
        "announce it like a system message."
    )
    parts.append(
        "- Don't address your party members or "
        "reply to the chat history above."
    )
    parts.append("- No coordinates, no meta-game talk.")

    return append_json_instruction(
        '\n'.join(parts),
        allow_action=False,
        skip_emote=True,
    )


# ============================================================
# ENTRY POINT
# ============================================================

def maybe_broadcast_event(
    db, client, config, ctx, event_type_label,
):
    """Chance-gated General-channel broadcast of a group
    event the bot just reacted to in party chat.

    Never raises; returns True only if a broadcast line
    was generated and queued.
    """
    try:
        if not _cfg_int(
            config, 'LLMChatter.Broadcast.Enable',
            _DEFAULT_ENABLE,
        ):
            return False

        eligible = _ELIGIBLE.get(event_type_label)
        if not eligible:
            return False
        suffix, default_chance = eligible
        chance = _cfg_int(
            config,
            f'LLMChatter.Broadcast.Chance.{suffix}',
            default_chance,
        )
        if random.randint(1, 100) > chance:
            return False

        zone_id = int(
            (ctx.get('extra_data') or {}).get('zone_id', 0)
            or ctx.get('zone_id', 0)
            or 0
        )
        if not zone_id:
            return False

        builder = _FACT_BUILDERS.get(event_type_label)
        fact = builder(ctx) if builder else None
        if not fact:
            return False

        # Claim the rate windows last, so skipped facts
        # don't burn them.
        bot_guid = ctx['bot_guid']
        if not _rate_limit_ok(bot_guid, zone_id, config):
            return False

        # Recent General history for this zone: context
        # and anti-repetition. Lazy import to keep the
        # module graph acyclic.
        from chatter_general import (
            _get_general_chat_history,
            _format_general_history,
            _store_general_chat,
        )
        history_lines = _cfg_int(
            config, 'LLMChatter.Broadcast.HistoryLines',
            _DEFAULT_HISTORY_LINES,
        )
        history_text = ''
        if history_lines > 0:
            history_text = _format_general_history(
                _get_general_chat_history(
                    db, zone_id, limit=history_lines,
                )
            )

        prompt = _build_broadcast_prompt(
            ctx, fact, zone_id, history_text,
        )

        # Guild culture bleed (lazy import).
        from chatter_guild import (
            get_bot_culture_line,
        )
        prompt += get_bot_culture_line(
            db, client, config, ctx['bot_guid'],
        )

        bot_name = ctx['bot_name']
        result = run_single_reaction(
            db, client, config,
            prompt=prompt,
            speaker_name=bot_name,
            bot_guid=bot_guid,
            channel='general',
            delay_seconds=random.uniform(4.0, 12.0),
            allow_emote_fallback=False,
            context=(
                f"broadcast:{event_type_label}"
                f":{bot_name}"
            ),
            label='general_broadcast',
            delivery_reason='bot_group_broadcast',
        )
        if not result['ok']:
            return False

        # The reason this feature exists: the line joins
        # the zone's General history so replies to players
        # (and later broadcasts) know what bots have been
        # through.
        _store_general_chat(
            db, zone_id, bot_name, True,
            result['message'],
        )
        logger.info(
            "[BROADCAST] %s | zone=%s bot=%s",
            event_type_label, zone_id, bot_name,
        )
        return True
    except Exception:
        logger.error(
            "Broadcast failed for %s",
            event_type_label, exc_info=True,
        )
        return False
