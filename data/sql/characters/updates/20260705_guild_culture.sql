-- Guild culture: a per-guild personality layer for member
-- bots. Auto-generated from the guild name on first
-- encounter (auto_generated = 1); server owners can edit
-- the culture text freely — the bridge never overwrites a
-- row that exists.

CREATE TABLE IF NOT EXISTS `llm_guild_culture` (
    `guild_id` INT UNSIGNED NOT NULL,
    `guild_name` VARCHAR(48) NOT NULL DEFAULT '',
    `culture` VARCHAR(400) NOT NULL,
    `auto_generated` TINYINT(1) NOT NULL DEFAULT 1,
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`guild_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='Per-guild chat-culture prompt layer for member bots';
