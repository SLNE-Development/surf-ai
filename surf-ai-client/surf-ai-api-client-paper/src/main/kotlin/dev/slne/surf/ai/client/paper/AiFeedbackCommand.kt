@file:Suppress("UnstableApiUsage")

package dev.slne.surf.ai.client.paper

import com.mojang.brigadier.arguments.StringArgumentType
import dev.slne.surf.ai.api.model.AiCheckResult
import io.papermc.paper.command.brigadier.CommandSourceStack
import io.papermc.paper.command.brigadier.Commands
import io.papermc.paper.plugin.lifecycle.event.types.LifecycleEvents
import org.bukkit.entity.Player
import java.util.UUID

/**
 * Debug/test entry point only - the real trigger comes from the moderation plugin (out of scope).
 * The score list is empty here since the client does not hold the microservice's TTL cache; the
 * dialog still lets a mod submit feedback for that requestId, which the microservice resolves
 * against its own cache.
 */
object AiFeedbackCommand {
    fun register(plugin: PaperMain) {
        plugin.lifecycleManager.registerEventHandler(LifecycleEvents.COMMANDS) { event ->
            event.registrar().register(
                Commands.literal("aifeedback")
                    .then(
                        Commands.argument("id", StringArgumentType.word())
                            .executes { ctx ->
                                val player = ctx.source.sender as? Player
                                if (player == null) {
                                    ctx.source.sender.sendPlainMessage("This command can only be used by players.")
                                    return@executes 0
                                }
                                val id = runCatching { UUID.fromString(StringArgumentType.getString(ctx, "id")) }.getOrNull()
                                if (id == null) {
                                    player.sendPlainMessage("Invalid UUID.")
                                    return@executes 0
                                }
                                FeedbackDialog.open(player, id, AiCheckResult(id, emptyList()))
                                1
                            }
                    )
                    .build()
            )
        }
    }
}
