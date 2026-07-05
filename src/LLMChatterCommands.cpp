/*
 * mod-llm-chatter - LLM-issued playerbot commands
 *
 * Tier 1 of the playerbots integration: the Python
 * bridge classifies a player's natural-language request
 * ("hang back and keep me healed") into a playerbot
 * command string and inserts it into
 * llm_chatter_commands. This poller validates each row
 * against a hard allowlist and injects it through
 * PlayerbotAI::HandleCommand with the REQUESTING player
 * as sender, so playerbots' own security model applies
 * unchanged. Every row keeps an audit trail (status +
 * detail + timestamps).
 */

#include "LLMChatterCommands.h"
#include "LLMChatterConfig.h"
#include "LLMChatterShared.h"
#include "Chat.h"
#include "DatabaseEnv.h"
#include "Group.h"
#include "Log.h"
#include "ObjectAccessor.h"
#include "Player.h"
#include "Playerbots.h"

#include <algorithm>
#include <cctype>
#include <string>

namespace
{
    bool IsAllowlistedCommand(std::string const& cmd)
    {
        std::string lowered = cmd;
        std::transform(lowered.begin(), lowered.end(),
                       lowered.begin(),
                       [](unsigned char c)
                       {
                           return static_cast<char>(
                               std::tolower(c));
                       });
        for (std::string const& entry :
             sLLMChatterConfig->_commandsAllowlist)
        {
            if (entry.empty())
                continue;
            if (entry.back() == ' ')
            {
                // Prefix rule ("co ", "nc "): the
                // command must start with it and
                // carry an argument.
                if (lowered.size() > entry.size()
                    && lowered.compare(
                           0, entry.size(),
                           entry) == 0)
                    return true;
            }
            else if (lowered == entry)
                return true;
        }
        return false;
    }

    void MarkRejected(uint32 rowId, char const* why)
    {
        CharacterDatabase.Execute(
            "UPDATE llm_chatter_commands "
            "SET status = 'rejected', detail = '{}' "
            "WHERE id = {}",
            why, rowId);
    }
} // namespace

void ProcessPendingBotCommands()
{
    if (!sLLMChatterConfig
        || !sLLMChatterConfig->IsEnabled()
        || !sLLMChatterConfig->_commandsEnable)
        return;

    QueryResult result = CharacterDatabase.Query(
        "SELECT id, bot_guid, requester_guid, command "
        "FROM llm_chatter_commands "
        "WHERE status = 'pending' "
        "AND (expires_at IS NULL "
        "OR expires_at > NOW()) "
        "ORDER BY id ASC LIMIT 4");
    if (!result)
        return;

    do
    {
        Field* fields = result->Fetch();
        uint32 rowId = fields[0].Get<uint32>();
        uint32 botGuid = fields[1].Get<uint32>();
        uint32 requesterGuid =
            fields[2].Get<uint32>();
        std::string command =
            fields[3].Get<std::string>();

        // Claim immediately so a crash mid-loop can
        // never double-execute; failures downgrade
        // the row to rejected below.
        CharacterDatabase.DirectExecute(
            "UPDATE llm_chatter_commands "
            "SET status = 'executed', "
            "executed_at = NOW() "
            "WHERE id = {} AND status = 'pending'",
            rowId);

        if (!IsAllowlistedCommand(command))
        {
            MarkRejected(rowId, "not_allowlisted");
            continue;
        }

        Player* bot = ObjectAccessor::FindPlayer(
            ObjectGuid::Create<HighGuid::Player>(
                botGuid));
        Player* requester =
            ObjectAccessor::FindPlayer(
                ObjectGuid::Create<HighGuid::Player>(
                    requesterGuid));
        if (!bot || !requester)
        {
            MarkRejected(rowId, "offline");
            continue;
        }

        if (!IsPlayerBot(bot))
        {
            MarkRejected(rowId, "not_a_bot");
            continue;
        }

        if (sLLMChatterConfig
                ->_commandsRequireSameGroup
            && (!bot->GetGroup()
                || bot->GetGroup()
                   != requester->GetGroup()))
        {
            MarkRejected(rowId, "not_same_group");
            continue;
        }

        PlayerbotAI* ai = GET_PLAYERBOT_AI(bot);
        if (!ai)
        {
            MarkRejected(rowId, "no_ai");
            continue;
        }

        // Whisper-typed so playerbots routes replies
        // back to the requester; the security gate
        // inside HandleCommand checks the requester.
        ai->HandleCommand(
            CHAT_MSG_WHISPER, command, requester);

        LOG_INFO("module",
            "LLMChatter: executed bot command "
            "bot={} cmd='{}' requester={}",
            bot->GetName(), command,
            requester->GetName());
    } while (result->NextRow());
}
