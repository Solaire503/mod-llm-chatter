"""Whisper conversations.

Handles `player_whisper_msg` events queued by the C++
whisper hook: a real player whispered a bot with
conversational text (playerbot control commands are
filtered C++-side and never arrive here).

Whispers are the most intimate channel, so companions get
the full deep treatment (backstory, all shared memories,
multi-line replies); regular bots answer with a brief
persona reply. Both paths get knowledge grounding and the
Tier 1 command pipeline ("come help me" whispered to a
grouped bot both answers AND acts).

History: the C++ hook stores the inbound line and C++
delivery stores the outbound line in llm_whisper_history,
so this handler only READS history — no double writes.

Identity comes from llm_bot_identities (persistent,
group-independent) — whispered bots may not be in any
group.
"""

import datetime
import logging
import random
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
    run_single_reaction,
    build_bot_identity_from_dict,
    get_class_name,
    get_race_name,
    get_gender_label,
    parse_extra_data,
)
from chatter_db import fail_event
from chatter_group_state import _mark_event
from chatter_memory import (
    get_all_bot_memories,
    get_bot_memories,
    sanitize_memory_for_prompt,
)
from chatter_knowledge import lookup_game_knowledge
from chatter_commands import maybe_queue_player_command
from chatter_companion import is_companion

logger = logging.getLogger(__name__)

_DEFAULT_HISTORY_LIMIT = 20
_DEFAULT_MAX_TOKENS_COMPANION = 450
_DEFAULT_MAX_LINES = 3


def _cfg_int(config, key, default):
    """Read an int config value, falling back on error."""
    if config is None:
        return default
    try:
        return int(config.get(key, default))
    except (TypeError, ValueError):
        return default


def _load_identity(db, bot_guid):
    """Persistent identity row, or {}."""
    try:
        cursor = db.cursor(dictionary=True)
        cursor.execute(
            "SELECT trait1, trait2, trait3, tone, "
            "backstory FROM llm_bot_identities "
            "WHERE bot_guid = %s",
            (int(bot_guid),),
        )
        return cursor.fetchone() or {}
    except Exception:
        return {}


def get_whisper_history(db, bot_guid, player_guid,
                        limit=_DEFAULT_HISTORY_LIMIT):
    """Recent two-way whisper rows, oldest first."""
    cursor = db.cursor(dictionary=True)
    cursor.execute(
        "SELECT from_bot, message "
        "FROM llm_whisper_history "
        "WHERE bot_guid = %s AND player_guid = %s "
        "ORDER BY id DESC LIMIT %s",
        (int(bot_guid), int(player_guid), limit),
    )
    return list(reversed(cursor.fetchall()))


def _format_history(rows, bot_name, player_name):
    """Format whisper rows for the prompt."""
    lines = []
    for r in rows:
        who = bot_name if r['from_bot'] else player_name
        lines.append(f"  {who}: {r['message']}")
    return '\n'.join(lines)


def _build_whisper_prompt(
    bot, identity, player_name, player_message,
    history_text, memories, knowledge_block,
    command_ack, deep, max_lines,
):
    """Build the whisper reply prompt.

    deep=True (companions): full backstory + shared
    memories + multi-line JSON array.
    deep=False: brief single JSON reply.
    """
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

    if deep and identity.get('backstory'):
        parts.append("")
        parts.append("<backstory>")
        parts.append(identity['backstory'].strip())
        parts.append("</backstory>")

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
            "If none of these memories feel relevant "
            "to this moment, don't force a reference."
        )

    if history_text:
        parts.append("")
        parts.append("<whisper_history>")
        parts.append(
            "Your private whisper conversation so far:"
        )
        parts.append(history_text)
        parts.append("</whisper_history>")

    parts.append("")
    parts.append(
        f"{player_name} just whispered you privately: "
        f"\"{player_message}\""
    )

    if knowledge_block:
        parts.append(knowledge_block)
    if command_ack:
        parts.append(command_ack)

    parts.append("")
    if deep:
        parts.append(
            "This is a private conversation -- just "
            "the two of you. Speak more openly and "
            "personally than you would in front of "
            "the group. Respond to what they actually "
            "said; a whisper deserves a real answer."
        )
    else:
        parts.append(
            "This is a private whisper. Reply "
            "briefly and in character -- one short "
            "message, like a real player typing back."
        )

    parts.append("")
    parts.append("Rules:")
    parts.append(
        f"- Speak only as {bot_name}, in first person."
    )
    parts.append(
        "- No quotes around your words, no emojis, no "
        "bracketed stage directions."
    )
    parts.append(
        "- Don't claim kills, loot, quests, or trades "
        "you didn't actually do."
    )

    prompt = '\n'.join(parts)

    if deep:
        footer = (
            f"\n\nRespond as a JSON array of 1 to "
            f"{max_lines} messages, each shaped like "
            f"{{\"speaker\": \"{bot_name}\", "
            f"\"message\": \"...\"}}. Use a single "
            f"message for a short reply. Physical "
            f"actions belong INSIDE the message text, "
            f"wrapped in *asterisks*. JSON rules: "
            f"double quotes, escape quotes/newlines, "
            f"no trailing commas, no code fences. "
            f"Output ONLY the JSON array."
        )
    else:
        footer = (
            "\n\nRespond with ONLY a JSON object: "
            "{\"message\": \"...\"}. Double quotes, "
            "no code fences, nothing else."
        )
    return prompt + footer


# ============================================================
# BOT-INITIATED WHISPERS
# ============================================================
# Rarely — and only into silence — a bot reaches out
# FIRST. Eligibility: bots on the player's friends list,
# flagged companions (is_manual), and optionally guild-
# mates. The "not overdone" contract is a stack of gates:
#   1. pair silence: no whispers either direction for
#      QuietHours
#   2. attempt spacing: a pair only ROLLS once per
#      AttemptHours (in-memory)
#   3. chance roll (higher when the player just returned
#      from a real absence)
#   4. daily cap per player across ALL bots
# Once the player replies, the normal whisper handler
# takes over — conversations are never rate-limited.

_pair_last_attempt = {}
_daily_counts = {}
_prev_online = set()
_online_baseline_set = False
_initiate_lock = threading.Lock()

# Reply debounce: min seconds between processed whisper
# replies per (bot, player) pair — rapid-fire whispers
# beyond this are skipped (each costs LLM calls).
_pair_last_reply = {}
_reply_lock = threading.Lock()


def _reply_debounce_ok(bot_guid, player_guid, config):
    min_s = _cfg_int(
        config, 'LLMChatter.Whisper.ReplyMinSeconds', 5
    )
    now = time.time()
    key = (int(bot_guid), int(player_guid))
    with _reply_lock:
        if now - _pair_last_reply.get(key, 0) < min_s:
            return False
        _pair_last_reply[key] = now
        return True


def _same_group(db, bot_guid, player_guid):
    """True if bot and player are in the same group."""
    try:
        cursor = db.cursor()
        cursor.execute(
            "SELECT 1 FROM group_member a "
            "JOIN group_member b ON b.guid = a.guid "
            "WHERE a.memberGuid = %s "
            "AND b.memberGuid = %s LIMIT 1",
            (int(bot_guid), int(player_guid)),
        )
        return cursor.fetchone() is not None
    except Exception:
        return False


def _initiate_cfg(config):
    def gi(key, default):
        try:
            return int(config.get(
                f'LLMChatter.Whisper.Initiate.{key}',
                default,
            ))
        except (TypeError, ValueError):
            return default
    return {
        'enable': gi('Enable', 1),
        'quiet_hours': gi('QuietHours', 12),
        'attempt_hours': gi('AttemptHours', 4),
        'chance': gi('Chance', 25),
        'login_chance': gi('LoginChance', 50),
        'min_away_hours': gi('MinAwayHours', 8),
        'daily_cap': gi('DailyCapPerPlayer', 2),
        'include_guild': gi('IncludeGuild', 0),
    }


def _daily_cap_ok(player_guid, cap):
    today = datetime.date.today()
    with _initiate_lock:
        day, count = _daily_counts.get(
            int(player_guid), (today, 0)
        )
        if day != today:
            count = 0
        return count < cap


def _bump_daily(player_guid):
    today = datetime.date.today()
    with _initiate_lock:
        day, count = _daily_counts.get(
            int(player_guid), (today, 0)
        )
        if day != today:
            count = 0
        _daily_counts[int(player_guid)] = (
            today, count + 1
        )


# WotLK faction race groups: cross-faction whispers are
# impossible in the real game, so initiations must be
# faction-matched or the illusion shatters.
_ALLIANCE_RACES = '1,3,4,7,11'
_HORDE_RACES = '2,5,6,8,10'


def _candidate_bots(db, player_guid, player_race,
                    include_guild):
    """Online bots allowed to initiate to this player.

    - friends-list bots (player chose them)
    - companions (is_manual) WITH shared history
      (memories or past whispers — no cold-opens)
    - optionally guild-mates
    All faction-matched; a bot is anything with an
    llm_bot_identities row or an RNDBOT account.
    """
    same_faction = (
        _ALLIANCE_RACES
        if int(player_race) in (1, 3, 4, 7, 11)
        else _HORDE_RACES
    )
    guild_join = ''
    if include_guild:
        guild_join = (
            " OR c.guid IN ("
            "   SELECT gm2.guid FROM guild_member gm2"
            "   WHERE gm2.guildid = ("
            "     SELECT gm1.guildid FROM guild_member"
            "     gm1 WHERE gm1.guid = %s LIMIT 1"
            "   )"
            " )"
        )
    sql = (
        "SELECT DISTINCT c.guid, c.name, c.class, "
        "c.race, c.level, c.gender "
        "FROM characters c "
        "LEFT JOIN character_social cs "
        "  ON cs.guid = %s AND cs.friend = c.guid "
        "  AND (cs.flags & 1) "
        "LEFT JOIN llm_bot_identities bi "
        "  ON bi.bot_guid = c.guid "
        "JOIN acore_auth.account a "
        "  ON c.account = a.id "
        "WHERE c.online = 1 AND c.guid != %s "
        "AND c.race IN (" + same_faction + ") "
        # Grouped-together = actively present; you
        # don't 'reach out into the silence' to
        # someone standing next to you.
        "AND NOT EXISTS ("
        "  SELECT 1 FROM group_member gmp "
        "  JOIN group_member gmb "
        "    ON gmb.guid = gmp.guid "
        "  WHERE gmp.memberGuid = %s "
        "    AND gmb.memberGuid = c.guid"
        ") "
        "AND (bi.bot_guid IS NOT NULL "
        "     OR a.username LIKE 'RNDBOT%%') "
        "AND (cs.friend IS NOT NULL "
        "     OR (bi.is_manual = 1 AND ("
        "       EXISTS (SELECT 1 FROM llm_bot_memories"
        "         m WHERE m.bot_guid = c.guid"
        "         AND m.player_guid = %s)"
        "       OR EXISTS (SELECT 1 FROM"
        "         llm_whisper_history w"
        "         WHERE w.bot_guid = c.guid"
        "         AND w.player_guid = %s)"
        "     ))" + guild_join + ")"
    )
    params = [
        int(player_guid), int(player_guid),
        int(player_guid), int(player_guid),
        int(player_guid),
    ]
    if include_guild:
        params.append(int(player_guid))
    cursor = db.cursor(dictionary=True)
    cursor.execute(sql, tuple(params))
    return cursor.fetchall()


def _hours_since_last_whisper(db, bot_guid,
                              player_guid):
    """Hours since the pair last whispered, or None."""
    cursor = db.cursor(dictionary=True)
    cursor.execute(
        "SELECT TIMESTAMPDIFF(MINUTE, "
        "MAX(created_at), NOW()) AS mins "
        "FROM llm_whisper_history "
        "WHERE bot_guid = %s AND player_guid = %s",
        (int(bot_guid), int(player_guid)),
    )
    row = cursor.fetchone()
    if not row or row['mins'] is None:
        return None
    return row['mins'] / 60.0


def _initiate_whisper(db, client, config, bot_row,
                      player_guid, player_name,
                      returned_after_hours):
    """Generate and queue one bot-initiated whisper."""
    bot_guid = int(bot_row['guid'])
    bot_name = bot_row['name']
    bot = {
        'guid': bot_guid,
        'name': bot_name,
        'class': get_class_name(bot_row['class']),
        'race': get_race_name(bot_row['race']),
        'level': int(bot_row['level']),
        'gender': get_gender_label(
            bot_row.get('gender', 0)
        ),
    }
    identity = _load_identity(db, bot_guid)
    deep = is_companion(db, bot_guid, config)

    raw = (
        get_all_bot_memories(
            db, bot_guid, player_guid, limit=10,
        ) if deep else (get_bot_memories(
            db, bot_guid, player_guid, count=3,
        ) or [])
    )
    memories = [
        s for s in (
            sanitize_memory_for_prompt(m)
            for m in raw
        ) if s
    ]
    history_text = _format_history(
        get_whisper_history(
            db, bot_guid, player_guid, limit=8,
        ),
        bot_name, player_name,
    )

    if returned_after_hours:
        trigger = (
            f"{player_name} just came back online "
            f"after being away for about "
            f"{int(returned_after_hours)} hours. "
            f"You noticed."
        )
    else:
        trigger = (
            f"It has been a long time since you and "
            f"{player_name} last spoke, and they "
            f"crossed your mind. You decided to "
            f"reach out."
        )

    parts = [build_bot_identity_from_dict(bot, suffix='.')]
    traits = [
        t for t in (
            identity.get('trait1'),
            identity.get('trait2'),
            identity.get('trait3'),
        ) if t
    ]
    if traits:
        parts.append(
            f"Your personality: {', '.join(traits)}"
        )
    if identity.get('tone'):
        parts.append(f"Your tone: {identity['tone']}")
    if deep and identity.get('backstory'):
        parts.append("")
        parts.append("<backstory>")
        parts.append(identity['backstory'].strip())
        parts.append("</backstory>")
    if memories:
        parts.append("")
        parts.append("<shared_history>")
        for mem in memories:
            parts.append(f"  - {mem}")
        parts.append("</shared_history>")
    if history_text:
        parts.append("")
        parts.append("<past_whispers>")
        parts.append(history_text)
        parts.append("</past_whispers>")
    parts.append("")
    parts.append(trigger)
    parts.append("")
    parts.append(
        "Whisper them FIRST. One short opener -- a "
        "greeting, a jab, a question, whatever fits "
        "who you are to each other. You are opening "
        "a door, not delivering a speech: max ~120 "
        "characters, first person, no stage "
        "directions."
    )
    parts.append(
        "\nRespond with ONLY a JSON object: "
        "{\"message\": \"...\"}. Double quotes, no "
        "code fences, nothing else."
    )

    # Guild culture bleed (lazy import).
    from chatter_guild import get_bot_culture_line
    initiate_prompt = (
        '\n'.join(parts)
        + get_bot_culture_line(
            db, client, config, bot_guid,
        )
    )

    result = run_single_reaction(
        db, client, config,
        prompt=initiate_prompt,
        speaker_name=bot_name,
        bot_guid=bot_guid,
        channel='whisper',
        delay_seconds=random.uniform(2.0, 8.0),
        allow_emote_fallback=False,
        context=f"whisper-init:{bot_name}",
        label='whisper_initiate',
        delivery_reason='whisper_initiate',
        player_guid=int(player_guid),
    )
    if result['ok']:
        logger.info(
            "[WHISPER-INIT] %s -> %s%s",
            bot_name, player_name,
            ' (return greeting)'
            if returned_after_hours else '',
        )
        return True
    return False


def check_initiated_whispers(db, client, config):
    """Periodic pass: maybe let a bot whisper first.

    Called from the bridge main loop worker pool.
    """
    cfg = _initiate_cfg(config)
    if not cfg['enable']:
        return False

    cursor = db.cursor(dictionary=True)
    cursor.execute(
        "SELECT c.guid, c.name, c.race, "
        "c.logout_time "
        "FROM characters c "
        "JOIN acore_auth.account a "
        "  ON c.account = a.id "
        "WHERE c.online = 1 "
        "AND a.username NOT LIKE 'RNDBOT%' "
        "AND NOT EXISTS ("
        "  SELECT 1 FROM llm_bot_identities bi"
        "  WHERE bi.bot_guid = c.guid"
        ")"
    )
    players = cursor.fetchall()

    now = time.time()
    initiated = False

    # Online-set diff: someone is "returning" only if
    # they were NOT online last pass but are now. The
    # first pass after a bridge start only establishes
    # the baseline (players already online then are not
    # "returning" — their logout_time is stale).
    global _online_baseline_set
    current_online = {int(p['guid']) for p in players}
    with _initiate_lock:
        baseline_ready = _online_baseline_set
        prev = set(_prev_online)
        _prev_online.clear()
        _prev_online.update(current_online)
        _online_baseline_set = True

    for p in players:
        player_guid = int(p['guid'])

        returned_after = None
        if (
            baseline_ready
            and player_guid not in prev
            and p['logout_time']
        ):
            away_h = (
                now - int(p['logout_time'])
            ) / 3600.0
            if away_h >= cfg['min_away_hours']:
                returned_after = away_h

        if not _daily_cap_ok(
            player_guid, cfg['daily_cap']
        ):
            continue

        candidates = _candidate_bots(
            db, player_guid, p['race'],
            cfg['include_guild'],
        )
        random.shuffle(candidates)
        for bot_row in candidates:
            bot_guid = int(bot_row['guid'])
            pair = (bot_guid, player_guid)

            with _initiate_lock:
                last = _pair_last_attempt.get(pair, 0)
            if now - last < cfg['attempt_hours'] * 3600:
                continue
            with _initiate_lock:
                _pair_last_attempt[pair] = now

            hours = _hours_since_last_whisper(
                db, bot_guid, player_guid,
            )
            if hours is not None \
                    and hours < cfg['quiet_hours']:
                continue

            chance = (
                cfg['login_chance']
                if returned_after
                else cfg['chance']
            )
            if random.randint(1, 100) > chance:
                continue

            if _initiate_whisper(
                db, client, config, bot_row,
                player_guid, p['name'],
                returned_after,
            ):
                _bump_daily(player_guid)
                initiated = True
            break  # at most one per player per pass
    return initiated


def process_player_whisper_event(db, client, config, event):
    """Reply to a real player's whisper."""
    event_id = event['id']
    extra_data = parse_extra_data(
        event.get('extra_data'),
        event_id, 'player_whisper_msg',
    )
    if not extra_data:
        _mark_event(db, event_id, 'skipped')
        return False

    try:
        bot_guid = int(extra_data.get('bot_guid', 0))
        bot_name = extra_data.get('bot_name', '')
        player_guid = int(
            extra_data.get('player_guid', 0)
        )
        player_name = extra_data.get(
            'player_name', 'someone'
        )
        player_message = extra_data.get(
            'player_message', ''
        )
        if not bot_guid or not player_guid \
                or not player_message:
            _mark_event(db, event_id, 'skipped')
            return False

        bot = {
            'guid': bot_guid,
            'name': bot_name,
            'class': get_class_name(
                int(extra_data.get('bot_class', 0))
            ),
            'race': get_race_name(
                int(extra_data.get('bot_race', 0))
            ),
            'level': int(
                extra_data.get('bot_level', 1)
            ),
            'gender': get_gender_label(
                int(extra_data.get('bot_gender', 0))
            ),
        }

        # Rapid-fire whispers: reply to the first,
        # skip the rest inside the debounce window
        # (each reply costs multiple LLM calls).
        if not _reply_debounce_ok(
            bot_guid, player_guid, config,
        ):
            _mark_event(db, event_id, 'skipped')
            return False

        identity = _load_identity(db, bot_guid)
        deep = is_companion(db, bot_guid, config)

        # Tier 1 command: a whispered "come help me"
        # both acts and answers — but ONLY when the
        # C++ same-group gate would actually execute
        # it; otherwise the bot would verbally commit
        # to an order that gets rejected.
        command_ack = ''
        if _same_group(db, bot_guid, player_guid):
            command_ack = maybe_queue_player_command(
                db, client, config, event_id,
                bot_guid, bot_name,
                player_guid, player_name,
                player_message,
            )

        knowledge_block = ''
        if not command_ack:
            knowledge_block = lookup_game_knowledge(
                client, config, player_message,
            )

        hist_limit = _cfg_int(
            config, 'LLMChatter.Whisper.HistoryLimit',
            _DEFAULT_HISTORY_LIMIT,
        )
        history_text = _format_history(
            get_whisper_history(
                db, bot_guid, player_guid,
                limit=hist_limit,
            ),
            bot_name, player_name,
        )

        memories = []
        if deep:
            raw = get_all_bot_memories(
                db, bot_guid, player_guid, limit=20,
            )
        else:
            raw = get_bot_memories(
                db, bot_guid, player_guid, count=3,
            ) or []
        memories = [
            s for s in (
                sanitize_memory_for_prompt(m)
                for m in raw
            ) if s
        ]

        max_lines = _cfg_int(
            config, 'LLMChatter.Whisper.MaxLines',
            _DEFAULT_MAX_LINES,
        )
        prompt = _build_whisper_prompt(
            bot, identity, player_name,
            player_message, history_text, memories,
            knowledge_block, command_ack,
            deep, max_lines,
        )

        # Guild culture bleed (lazy import).
        from chatter_guild import (
            get_bot_culture_line,
        )
        prompt += get_bot_culture_line(
            db, client, config, bot_guid,
        )

        if deep:
            response = call_llm(
                client, prompt, config,
                max_tokens_override=_cfg_int(
                    config,
                    'LLMChatter.Whisper.MaxTokens',
                    _DEFAULT_MAX_TOKENS_COMPANION,
                ),
                context=(
                    f"whisper:#{event_id}:{bot_name}"
                ),
                label='whisper_companion_msg',
            )
            if not response:
                _mark_event(db, event_id, 'skipped')
                return False
            entries = parse_conversation_response(
                response, [bot_name],
            )
            if not entries:
                parsed = parse_single_response(
                    response
                )
                single = cleanup_message(
                    strip_speaker_prefix(
                        parsed.get('message', ''),
                        bot_name,
                    ),
                    preserve_leading_action=True,
                )
                if single:
                    entries = [{
                        'name': bot_name,
                        'message': single,
                    }]
            if not entries:
                _mark_event(db, event_id, 'skipped')
                return False

            entries = entries[:max_lines]
            cumulative = 0.0
            prev_len = len(player_message)
            delivered = 0
            for i, entry in enumerate(entries):
                msg = cleanup_message(
                    strip_speaker_prefix(
                        entry.get('message', ''),
                        bot_name,
                    ),
                    preserve_leading_action=True,
                )
                if not msg:
                    continue
                if len(msg) > 255:
                    msg = msg[:252] + "..."
                # Delivery only sequence-gates say/
                # msay; whisper ordering rides
                # deliver_at (second resolution), so
                # keep line gaps comfortably > 1s.
                cumulative += max(
                    1.5,
                    calculate_dynamic_delay(
                        len(msg), config,
                        prev_message_length=prev_len,
                        responsive=True,
                    ),
                )
                insert_chat_message(
                    db, bot_guid, bot_name, msg,
                    channel='whisper',
                    delay_seconds=cumulative,
                    event_id=event_id,
                    sequence=i,
                    player_guid=player_guid,
                    config=config,
                    delivery_policy='responsive',
                    delivery_reason=(
                        'player_whisper_msg'
                    ),
                )
                prev_len = len(msg)
                delivered += 1
            if not delivered:
                _mark_event(db, event_id, 'skipped')
                return False
        else:
            result = run_single_reaction(
                db, client, config,
                prompt=prompt,
                speaker_name=bot_name,
                bot_guid=bot_guid,
                channel='whisper',
                delay_seconds=(
                    calculate_dynamic_delay(
                        40, config,
                        prev_message_length=len(
                            player_message
                        ),
                        responsive=True,
                    )
                ),
                event_id=event_id,
                allow_emote_fallback=False,
                context=(
                    f"whisper:#{event_id}:{bot_name}"
                ),
                label='whisper_msg',
                delivery_reason='player_whisper_msg',
                player_guid=player_guid,
            )
            if not result['ok']:
                _mark_event(db, event_id, 'skipped')
                return False

        _mark_event(db, event_id, 'completed')
        return True
    except Exception:
        fail_event(
            db, event_id, 'player_whisper_msg',
            'handler error',
        )
        return False
