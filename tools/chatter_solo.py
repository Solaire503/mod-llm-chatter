"""Solo-bot experience broadcasts.

Handles `bot_solo_kill` / `bot_solo_levelup` /
`bot_solo_death` events: an UNGROUPED bot (or bot-only
group) had a notable experience in a zone where a real
player is present, and posts about it in that zone's
General channel.

This is Phase 3 (contextual broadcasting) extended beyond
player groups. It reuses chatter_broadcast's fact
builders, prompt, rate limiters, and history storage —
solo events share the same per-bot/per-zone General flood
budget as group broadcasts ON PURPOSE.

The C++ side already applied the SoloChatter.*Chance roll
and per-bot cooldown before queueing, so this handler
does NOT roll again — it only enforces the shared General
rate limits.

Identity traits come from llm_bot_identities when
available (solo bots have no group traits row).
"""

import logging
import random

from chatter_shared import (
    parse_extra_data,
    run_single_reaction,
)
from chatter_db import fail_event
from chatter_group_state import _mark_event
from chatter_handler_pipeline import (
    _build_bot_from_extra,
)
from chatter_broadcast import (
    _build_broadcast_prompt,
    _rate_limit_ok,
    _cfg_int,
)

logger = logging.getLogger(__name__)


# Solo facts: the bot was ALONE — the group builders'
# "your group / your groupmate" phrasing would be a lie.

def _fact_solo_kill(extra):
    name = extra.get('creature_name')
    if not name:
        return None
    is_boss = extra.get('is_boss')
    is_rare = extra.get('is_rare')
    if not (is_boss or is_rare):
        return None
    kind = 'boss' if is_boss else 'rare creature'
    return (
        f"You just brought down {name}, a {kind} -- "
        f"on your own, with no group backing you up."
    )


def _fact_solo_levelup(extra):
    level = extra.get('bot_level')
    if not level:
        return None
    return f"You just reached level {level}."


def _fact_solo_death(extra):
    killer = extra.get('killer_name')
    if killer:
        return (
            f"You just got killed by {killer}, out "
            f"adventuring alone. You're at the "
            f"graveyard now."
        )
    return (
        "You just died out adventuring alone. "
        "You're at the graveyard now."
    )


_FACT_BUILDERS = {
    'bot_solo_kill': _fact_solo_kill,
    'bot_solo_levelup': _fact_solo_levelup,
    'bot_solo_death': _fact_solo_death,
}


def _identity_traits(db, bot_guid):
    """Persistent traits + tone, or ([], None)."""
    try:
        cursor = db.cursor(dictionary=True)
        cursor.execute(
            "SELECT trait1, trait2, trait3, tone "
            "FROM llm_bot_identities "
            "WHERE bot_guid = %s",
            (int(bot_guid),),
        )
        row = cursor.fetchone()
        if not row:
            return [], None
        traits = [
            t for t in (
                row.get('trait1'), row.get('trait2'),
                row.get('trait3'),
            ) if t
        ]
        return traits, row.get('tone')
    except Exception:
        return [], None


def _handle_solo(db, client, config, event, event_type):
    """Shared solo broadcast pipeline."""
    event_id = event['id']
    extra_data = parse_extra_data(
        event.get('extra_data'), event_id, event_type,
    )
    if not extra_data:
        _mark_event(db, event_id, 'skipped')
        return False

    try:
        if not _cfg_int(
            config, 'LLMChatter.Broadcast.Enable', 1
        ):
            _mark_event(db, event_id, 'skipped')
            return False

        bot_guid = int(extra_data.get('bot_guid', 0))
        bot_name = extra_data.get('bot_name', '')
        zone_id = int(
            event.get('zone_id')
            or extra_data.get('zone_id', 0)
            or 0
        )
        if not bot_guid or not zone_id:
            _mark_event(db, event_id, 'skipped')
            return False

        bot = _build_bot_from_extra(extra_data)
        traits, tone = _identity_traits(db, bot_guid)

        ctx = {
            'bot': bot,
            'bot_guid': bot_guid,
            'bot_name': bot_name,
            'traits': traits,
            'stored_tone': tone,
            'extra_data': extra_data,
            'zone_id': zone_id,
        }

        fact = _FACT_BUILDERS[event_type](extra_data)
        if not fact:
            _mark_event(db, event_id, 'skipped')
            return False

        # Shared General flood budget with group
        # broadcasts (C++ already rolled the solo
        # chance + per-bot solo cooldown).
        if not _rate_limit_ok(
            bot_guid, zone_id, config
        ):
            _mark_event(db, event_id, 'skipped')
            return False

        from chatter_general import (
            _get_general_chat_history,
            _format_general_history,
            _store_general_chat,
        )
        history_lines = _cfg_int(
            config, 'LLMChatter.Broadcast.HistoryLines',
            6,
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
            db, client, config, bot_guid,
        )

        result = run_single_reaction(
            db, client, config,
            prompt=prompt,
            speaker_name=bot_name,
            bot_guid=bot_guid,
            channel='general',
            delay_seconds=random.uniform(3.0, 10.0),
            event_id=event_id,
            allow_emote_fallback=False,
            context=(
                f"solo:{event_type}:{bot_name}"
            ),
            label='solo_broadcast',
            delivery_reason=event_type,
        )
        if not result['ok']:
            _mark_event(db, event_id, 'skipped')
            return False

        _store_general_chat(
            db, zone_id, bot_name, True,
            result['message'],
        )
        logger.info(
            "[SOLO] %s | zone=%s bot=%s",
            event_type, zone_id, bot_name,
        )
        _mark_event(db, event_id, 'completed')
        return True
    except Exception:
        fail_event(
            db, event_id, event_type, 'handler error',
        )
        return False


def process_solo_kill_event(db, client, config, event):
    """Solo bot brags about a boss/rare kill."""
    return _handle_solo(
        db, client, config, event, 'bot_solo_kill',
    )


def process_solo_levelup_event(db, client, config, event):
    """Solo bot announces a milestone level."""
    return _handle_solo(
        db, client, config, event, 'bot_solo_levelup',
    )


def process_solo_death_event(db, client, config, event):
    """Solo bot grumbles about dying."""
    return _handle_solo(
        db, client, config, event, 'bot_solo_death',
    )
