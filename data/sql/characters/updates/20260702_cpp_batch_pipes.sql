-- C++ batch: guild/whisper forwarding, solo-bot events, Tier 1 commands.
-- Idempotent. Safe to apply BEFORE the C++ compile (new enum values and
-- tables are inert until the new code writes to them).
--
-- REVISED 2026-07-05: the original version rebuilt the enum from a stale
-- value list and silently dropped 8 post-March event types (proximity_*,
-- bot_group_general_reaction, bot_tone_regen, bot_backstory_regen),
-- causing "Data truncated for column 'event_type'" on insert. This
-- version carries the FULL list and its guard detects both the fresh
-- and the broken state (missing new-batch values OR missing older
-- values), so it repairs a DB that ran the bad version.

-- 1. Event types (full list) ------------------------------------------
SET @enum_ok = (
  SELECT COUNT(*)
  FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME   = 'llm_chatter_events'
    AND COLUMN_NAME  = 'event_type'
    AND COLUMN_TYPE LIKE '%player_whisper_msg%'
    AND COLUMN_TYPE LIKE '%bot_group_general_reaction%'
);
SET @sql = IF(@enum_ok = 0,
  "ALTER TABLE `llm_chatter_events`
  MODIFY COLUMN `event_type` ENUM(
    'weather_change',
    'holiday_start',
    'holiday_end',
    'creature_death_boss',
    'creature_death_rare',
    'creature_death_guard',
    'player_enters_zone',
    'bot_pvp_kill',
    'bot_level_up',
    'bot_achievement',
    'bot_quest_complete',
    'world_boss_spawn',
    'rare_spawn',
    'transport_arrives',
    'day_night_transition',
    'enemy_player_near',
    'bot_loot_item',
    'bot_group_join',
    'bot_group_kill',
    'bot_group_death',
    'bot_group_loot',
    'bot_group_player_msg',
    'bot_group_combat',
    'bot_group_levelup',
    'bot_group_quest_complete',
    'bot_group_achievement',
    'bot_group_spell_cast',
    'bot_group_quest_objectives',
    'bot_group_resurrect',
    'bot_group_zone_transition',
    'bot_group_dungeon_entry',
    'bot_group_wipe',
    'bot_group_corpse_run',
    'player_general_msg',
    'minor_event',
    'bot_group_low_health',
    'bot_group_oom',
    'bot_group_aggro_loss',
    'bot_group_quest_accept',
    'bot_group_quest_accept_batch',
    'bot_group_discovery',
    'weather_ambient',
    'bot_group_nearby_object',
    'bot_group_join_batch',
    'bg_match_start',
    'bg_match_end',
    'bg_pvp_kill',
    'bg_flag_picked_up',
    'bg_flag_dropped',
    'bg_flag_captured',
    'bg_flag_returned',
    'bg_node_contested',
    'bg_node_captured',
    'bg_score_milestone',
    'bg_idle_chatter',
    'bg_player_arrival',
    'raid_boss_pull',
    'raid_boss_kill',
    'raid_boss_wipe',
    'raid_idle_morale',
    'bot_group_farewell',
    'bot_group_subzone_change',
    'bot_group_emote_observer',
    'bot_group_emote_reaction',
    'bot_group_screenshot_observation',
    'player_guild_msg',
    'player_whisper_msg',
    'bot_solo_kill',
    'bot_solo_levelup',
    'bot_solo_death',
    'bot_backstory_regen',
    'bot_group_general_reaction',
    'bot_tone_regen',
    'proximity_conversation',
    'proximity_player_conversation',
    'proximity_player_say',
    'proximity_reply',
    'proximity_say'
  ) NOT NULL",
  'SELECT 1');
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- 2. Guild chat history ------------------------------------------------
CREATE TABLE IF NOT EXISTS `llm_guild_chat_history` (
    `id` INT UNSIGNED NOT NULL AUTO_INCREMENT,
    `guild_id` INT UNSIGNED NOT NULL,
    `speaker_name` VARCHAR(48) NOT NULL,
    `is_bot` TINYINT(1) NOT NULL DEFAULT 0,
    `message` VARCHAR(512) NOT NULL,
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_guild_recent` (`guild_id`, `id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='Per-guild chat context for LLM guild replies';

-- 3. Whisper history ---------------------------------------------------
CREATE TABLE IF NOT EXISTS `llm_whisper_history` (
    `id` INT UNSIGNED NOT NULL AUTO_INCREMENT,
    `bot_guid` INT UNSIGNED NOT NULL,
    `player_guid` INT UNSIGNED NOT NULL,
    `from_bot` TINYINT(1) NOT NULL DEFAULT 0,
    `message` VARCHAR(512) NOT NULL,
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_pair_recent` (`bot_guid`, `player_guid`, `id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='Per (bot, player) whisper conversation context';

-- 4. Tier 1 command queue ----------------------------------------------
CREATE TABLE IF NOT EXISTS `llm_chatter_commands` (
    `id` INT UNSIGNED NOT NULL AUTO_INCREMENT,
    `bot_guid` INT UNSIGNED NOT NULL,
    `bot_name` VARCHAR(48) NOT NULL,
    `requester_guid` INT UNSIGNED NOT NULL,
    `requester_name` VARCHAR(48) NOT NULL,
    `command` VARCHAR(255) NOT NULL,
    `source_event_id` INT UNSIGNED DEFAULT NULL,
    `status` ENUM('pending','executed','rejected','expired')
        NOT NULL DEFAULT 'pending',
    `detail` VARCHAR(255) DEFAULT NULL,
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `executed_at` TIMESTAMP NULL DEFAULT NULL,
    `expires_at` TIMESTAMP NULL DEFAULT NULL,
    PRIMARY KEY (`id`),
    KEY `idx_pending` (`status`, `id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='LLM-issued playerbot commands (bridge writes, C++ executes)';
