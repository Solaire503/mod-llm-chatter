-- Explicit companion flag.
--
-- The backstory-length proxy for "is this a hand-authored
-- companion?" over-matches badly: auto-generated backstories
-- routinely exceed 500 chars (47 bots qualified on one test
-- server). is_manual makes companionship an explicit choice:
--   UPDATE llm_bot_identities SET is_manual = 1
--   WHERE bot_name IN ('Lariaraden', ...);

SET @has_col = (
  SELECT COUNT(*)
  FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME   = 'llm_bot_identities'
    AND COLUMN_NAME  = 'is_manual'
);
SET @sql = IF(@has_col = 0,
  'ALTER TABLE `llm_bot_identities`
   ADD COLUMN `is_manual` TINYINT(1) NOT NULL DEFAULT 0
   AFTER `backstory`',
  'SELECT 1');
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
