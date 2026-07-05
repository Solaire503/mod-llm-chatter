"""Game knowledge system: real answers from acore_world.

When a player asks a gameplay question ("who gives The
People's Militia?", "where do I find Marshal Dughan?",
"who drops the Staff of Westfall?"), the bot answers from
ACTUAL world-database rows — quest_template,
creature_queststarter/questender, creature spawns,
item_template, npc_vendor, creature_loot_template — never
from the model's training data.

Anti-hallucination contract: the LLM is handed a
<game_data> block of verified facts and instructed to
answer ONLY from it. If the database has nothing, the
block says so and the bot admits it doesn't know, in
character. No facts, no guessing.

Flow per player message:
  1. cheap regex prefilter (question-shaped?)
  2. quick_llm_analyze classifier -> {type, entity}
  3. acore_world lookups by type
  4. return a prompt block for the caller to append

Callers: party single reply + companion (chatter_group /
chatter_companion) and General replies (chatter_general).
Imports only downward (chatter_llm, chatter_shared,
chatter_db). Every failure returns '' — knowledge is
garnish, never load-bearing.
"""

import json
import logging
import re

from chatter_llm import quick_llm_analyze
from chatter_shared import get_zone_name
from chatter_db import get_db_connection
from chatter_constants import ZONE_COORDINATES

logger = logging.getLogger(__name__)

_DEFAULT_ENABLE = 1
_MAX_ENTITY_CHARS = 60

_QUALITY_NAMES = {
    0: 'poor (grey)', 1: 'common (white)',
    2: 'uncommon (green)', 3: 'rare (blue)',
    4: 'epic (purple)', 5: 'legendary (orange)',
}
_RANK_NAMES = {
    1: 'elite', 2: 'rare elite', 3: 'boss', 4: 'rare',
}

# Question-shaped messages only; everything else skips the
# classifier call entirely.
_PREFILTER = re.compile(
    r"\?|\b(where|who|what|which|how do|how can|"
    r"how to|anyone know|any idea)\b",
    re.IGNORECASE,
)


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

def _classify(client, config, player_message):
    """Classify a message as a gameplay data question.

    Returns {'type': 'quest'|'npc'|'item',
             'entity': str} or None.
    """
    prompt = (
        "A player in World of Warcraft (Wrath of the "
        "Lich King) chat said:\n"
        f"\"{player_message}\"\n\n"
        "Is this a question asking for factual game "
        "data — e.g. where to find an NPC or creature, "
        "who gives or ends a quest, what a quest "
        "requires, what level something is, who drops "
        "or sells an item?\n"
        "Respond with ONLY a JSON object:\n"
        "{\"is_question\": true/false, "
        "\"type\": \"quest\"|\"npc\"|\"item\"|"
        "\"other\", "
        "\"entity\": \"exact name being asked about\"}\n"
        "Rules: opinions, greetings, chit-chat, "
        "strategy advice, and questions about people "
        "in the chat are NOT data questions — use "
        "is_question=false. \"npc\" covers any "
        "creature or monster. The entity must be the "
        "name as the player wrote it, without "
        "surrounding brackets or quotes."
    )
    raw = quick_llm_analyze(
        client, config, prompt,
        max_tokens=80,
        label='knowledge_classify',
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
    if not data.get('is_question'):
        return None
    qtype = data.get('type')
    entity = (data.get('entity') or '').strip()
    entity = entity.strip('"\'[]').strip()
    if qtype not in ('quest', 'npc', 'item'):
        return None
    if not entity or len(entity) > _MAX_ENTITY_CHARS:
        return None
    return {'type': qtype, 'entity': entity}


# ============================================================
# WORLD-DB LOOKUPS
# ============================================================

def _like(entity):
    """Escape LIKE wildcards and wrap for substring."""
    escaped = (
        entity.replace('\\', '\\\\')
        .replace('%', '\\%')
        .replace('_', '\\_')
    )
    return f"%{escaped}%"


def _find_rows(cursor, exact_sql, like_sql, entity, limit):
    """Exact-name match first, LIKE fallback."""
    cursor.execute(exact_sql, (entity, limit))
    rows = cursor.fetchall()
    if rows:
        return rows
    cursor.execute(like_sql, (_like(entity), limit))
    return cursor.fetchall()


def _zone_from_position(map_id, x, y):
    """Approximate zone id from a spawn position using
    the ZONE_COORDINATES bounding boxes. Most creature
    rows have zoneId = 0, so this geometric fallback is
    what usually answers "where does X live?".
    """
    best = None
    best_area = None
    for zone_id, box in ZONE_COORDINATES.items():
        bmap, x_min, x_max, y_min, y_max = box
        if bmap != map_id:
            continue
        if not (x_min <= x <= x_max
                and y_min <= y <= y_max):
            continue
        # Boxes overlap near borders; prefer the
        # tightest box containing the point.
        area = (x_max - x_min) * (y_max - y_min)
        if best_area is None or area < best_area:
            best, best_area = zone_id, area
    return best


def _spawn_zones(cursor, creature_entry, limit=3):
    """Zone names where a creature spawns, most first.

    Uses creature.zoneId when populated, otherwise
    falls back to bounding-box matching on spawn
    positions.
    """
    cursor.execute(
        "SELECT zoneId, COUNT(*) AS cnt FROM creature "
        "WHERE id1 = %s AND zoneId > 0 "
        "GROUP BY zoneId ORDER BY cnt DESC LIMIT %s",
        (creature_entry, limit),
    )
    zone_ids = [
        int(r['zoneId']) for r in cursor.fetchall()
    ]
    if not zone_ids:
        cursor.execute(
            "SELECT map, position_x, position_y "
            "FROM creature WHERE id1 = %s LIMIT 10",
            (creature_entry,),
        )
        votes = {}
        for r in cursor.fetchall():
            zid = _zone_from_position(
                int(r['map']),
                float(r['position_x']),
                float(r['position_y']),
            )
            if zid:
                votes[zid] = votes.get(zid, 0) + 1
        zone_ids = sorted(
            votes, key=votes.get, reverse=True,
        )[:limit]
    names = [get_zone_name(z) for z in zone_ids]
    return [n for n in names if n]


def _first_spawn_zone(cursor, creature_entry):
    """Name of the zone where a creature spawns most."""
    zones = _spawn_zones(cursor, creature_entry, limit=1)
    return zones[0] if zones else None


def _quest_giver_lines(cursor, quest_id, relation, verb):
    """Fact lines for quest starters or enders.

    relation: 'queststarter' or 'questender'.
    """
    lines = []
    cursor.execute(
        f"SELECT ct.entry, ct.name "
        f"FROM creature_{relation} r "
        f"JOIN creature_template ct ON ct.entry = r.id "
        f"WHERE r.quest = %s LIMIT 2",
        (quest_id,),
    )
    for row in cursor.fetchall():
        zone = _first_spawn_zone(cursor, row['entry'])
        where = f" in {zone}" if zone else ""
        lines.append(
            f"{verb} NPC {row['name']}{where}."
        )
    if not lines:
        cursor.execute(
            f"SELECT gt.name "
            f"FROM gameobject_{relation} r "
            f"JOIN gameobject_template gt "
            f"ON gt.entry = r.id "
            f"WHERE r.quest = %s LIMIT 2",
            (quest_id,),
        )
        for row in cursor.fetchall():
            lines.append(
                f"{verb} object \"{row['name']}\"."
            )
    return lines


def _lookup_quest(cursor, entity):
    """Fact lines about a quest by (partial) title."""
    rows = _find_rows(
        cursor,
        "SELECT ID, LogTitle, QuestLevel, MinLevel, "
        "QuestSortID, "
        "ObjectiveText1, ObjectiveText2, "
        "ObjectiveText3, ObjectiveText4 "
        "FROM quest_template WHERE LogTitle = %s "
        "LIMIT %s",
        "SELECT ID, LogTitle, QuestLevel, MinLevel, "
        "QuestSortID, "
        "ObjectiveText1, ObjectiveText2, "
        "ObjectiveText3, ObjectiveText4 "
        "FROM quest_template WHERE LogTitle LIKE %s "
        "LIMIT %s",
        entity, 2,
    )
    facts = []
    for q in rows:
        level = int(q['QuestLevel'] or 0)
        min_level = int(q['MinLevel'] or 0)
        bits = []
        if level > 0:
            bits.append(f"level {level}")
        if min_level > 1:
            bits.append(f"requires level {min_level}")
        # QuestSortID > 0 is the quest's zone id
        # (negative values are sort categories).
        sort_id = int(q.get('QuestSortID') or 0)
        if sort_id > 0:
            zone = get_zone_name(sort_id)
            if zone:
                bits.append(f"a {zone} quest")
        suffix = f" ({', '.join(bits)})" if bits else ""
        facts.append(
            f"Quest \"{q['LogTitle']}\"{suffix}."
        )
        facts.extend(_quest_giver_lines(
            cursor, q['ID'], 'queststarter',
            'It is given by',
        ))
        facts.extend(_quest_giver_lines(
            cursor, q['ID'], 'questender',
            'It is turned in to',
        ))
        objectives = [
            (q.get(f'ObjectiveText{i}') or '').strip()
            for i in range(1, 5)
        ]
        objectives = [o for o in objectives if o]
        if objectives:
            facts.append(
                "Objectives: "
                + '; '.join(objectives) + "."
            )
    return facts


def _lookup_npc(cursor, entity):
    """Fact lines about an NPC/creature by name."""
    rows = _find_rows(
        cursor,
        "SELECT entry, name, subname, minlevel, "
        "maxlevel, `rank` FROM creature_template "
        "WHERE name = %s LIMIT %s",
        "SELECT entry, name, subname, minlevel, "
        "maxlevel, `rank` FROM creature_template "
        "WHERE name LIKE %s LIMIT %s",
        entity, 2,
    )
    facts = []
    for c in rows:
        title = c['name']
        if c.get('subname'):
            title += f" <{c['subname']}>"
        lo, hi = int(c['minlevel']), int(c['maxlevel'])
        level = (
            f"level {lo}" if lo == hi
            else f"level {lo}-{hi}"
        )
        rank = _RANK_NAMES.get(int(c.get('rank') or 0))
        rank_str = f", {rank}" if rank else ""
        facts.append(f"{title} ({level}{rank_str}).")

        zones = _spawn_zones(cursor, c['entry'])
        if zones:
            facts.append(
                f"Found in: {', '.join(zones)}."
            )

        cursor.execute(
            "SELECT q.LogTitle "
            "FROM creature_queststarter cs "
            "JOIN quest_template q ON q.ID = cs.quest "
            "WHERE cs.id = %s LIMIT 3",
            (c['entry'],),
        )
        quests = [
            r['LogTitle'] for r in cursor.fetchall()
            if r.get('LogTitle')
        ]
        if quests:
            facts.append(
                "Gives quests such as: "
                + '; '.join(
                    f"\"{t}\"" for t in quests
                ) + "."
            )
    return facts


def _lookup_item(cursor, entity):
    """Fact lines about an item by name."""
    rows = _find_rows(
        cursor,
        "SELECT entry, name, ItemLevel, RequiredLevel, "
        "Quality FROM item_template WHERE name = %s "
        "LIMIT %s",
        "SELECT entry, name, ItemLevel, RequiredLevel, "
        "Quality FROM item_template WHERE name LIKE %s "
        "LIMIT %s",
        entity, 2,
    )
    facts = []
    for it in rows:
        quality = _QUALITY_NAMES.get(
            int(it.get('Quality') or 0), ''
        )
        bits = [b for b in (
            quality,
            f"item level {it['ItemLevel']}"
            if int(it.get('ItemLevel') or 0) > 1
            else '',
            f"requires level {it['RequiredLevel']}"
            if int(it.get('RequiredLevel') or 0) > 1
            else '',
        ) if b]
        suffix = f" ({', '.join(bits)})" if bits else ""
        facts.append(f"Item \"{it['name']}\"{suffix}.")

        cursor.execute(
            "SELECT ct.name, clt.Chance "
            "FROM creature_loot_template clt "
            "JOIN creature_template ct "
            "ON ct.lootid = clt.Entry "
            "WHERE clt.Item = %s AND clt.Chance > 0 "
            "ORDER BY clt.Chance DESC LIMIT 3",
            (it['entry'],),
        )
        drops = []
        for r in cursor.fetchall():
            chance = float(r['Chance'])
            drops.append(
                f"{r['name']} ({chance:g}% chance)"
            )
        if drops:
            facts.append(
                f"Dropped by: {', '.join(drops)}."
            )

        cursor.execute(
            "SELECT ct.name FROM npc_vendor nv "
            "JOIN creature_template ct "
            "ON ct.entry = nv.entry "
            "WHERE nv.item = %s LIMIT 3",
            (it['entry'],),
        )
        vendors = [
            r['name'] for r in cursor.fetchall()
        ]
        if vendors:
            facts.append(
                f"Sold by: {', '.join(vendors)}."
            )

        reward_cols = [
            f"RewardChoiceItemID{i}" for i in range(1, 7)
        ] + [f"RewardItem{i}" for i in range(1, 5)]
        where = ' OR '.join(
            f"{col} = %s" for col in reward_cols
        )
        cursor.execute(
            f"SELECT LogTitle FROM quest_template "
            f"WHERE {where} LIMIT 3",
            tuple([it['entry']] * len(reward_cols)),
        )
        reward_of = [
            r['LogTitle'] for r in cursor.fetchall()
            if r.get('LogTitle')
        ]
        if reward_of:
            facts.append(
                "Reward from quest: "
                + '; '.join(
                    f"\"{t}\"" for t in reward_of
                ) + "."
            )
    return facts


_LOOKUPS = {
    'quest': _lookup_quest,
    'npc': _lookup_npc,
    'item': _lookup_item,
}


# ============================================================
# PROMPT BLOCK
# ============================================================

def _format_block(entity, facts):
    """Wrap fact lines in the anti-hallucination block."""
    lines = ["", "<game_data>"]
    if facts:
        lines.append(
            "Verified game-database facts relevant to "
            "the player's question:"
        )
        for fact in facts:
            lines.append(f"  - {fact}")
    else:
        lines.append(
            f"The game database has NO information "
            f"matching \"{entity}\"."
        )
    lines.append("</game_data>")
    if facts:
        lines.append(
            "Answer the player's question using ONLY "
            "the facts above, in your own voice and "
            "character. Do not invent names, places, "
            "levels, drop rates, or directions that "
            "are not in the facts. If the facts don't "
            "fully answer the question, share what "
            "they do say and admit you're not sure "
            "about the rest."
        )
    else:
        lines.append(
            "You do not actually know the answer. "
            "Say so naturally, in character -- do NOT "
            "make up an answer from general knowledge. "
            "Mocking the player for asking is fair "
            "game."
        )
    return '\n'.join(lines)


# ============================================================
# ENTRY POINT
# ============================================================

def lookup_game_knowledge(client, config, player_message):
    """Return a <game_data> prompt block for a player
    message, or '' when the message isn't a gameplay
    data question (or on any failure).
    """
    try:
        if not _cfg_int(
            config, 'LLMChatter.Knowledge.Enable',
            _DEFAULT_ENABLE,
        ):
            return ''
        if not player_message:
            return ''
        if not _PREFILTER.search(player_message):
            return ''

        classified = _classify(
            client, config, player_message,
        )
        if not classified:
            return ''

        qtype = classified['type']
        entity = classified['entity']
        facts = []
        world_db = None
        try:
            world_db = get_db_connection(
                config, 'acore_world',
            )
            cursor = world_db.cursor(dictionary=True)
            facts = _LOOKUPS[qtype](cursor, entity)
        except Exception:
            logger.error(
                "Knowledge lookup failed for %s "
                "\"%s\"", qtype, entity,
                exc_info=True,
            )
            # DB failure: better silent than a forced
            # "I don't know" for a question the DB
            # could actually answer.
            return ''
        finally:
            if world_db:
                try:
                    world_db.close()
                except Exception:
                    pass

        block = _format_block(entity, facts)
        logger.info(
            "[KNOWLEDGE] %s \"%s\" -> %d fact(s)",
            qtype, entity, len(facts),
        )
        return block
    except Exception:
        logger.error(
            "Knowledge pipeline failed", exc_info=True,
        )
        return ''
