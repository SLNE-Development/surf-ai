@file:Suppress("UnstableApiUsage")
@file:OptIn(NmsUseWithCaution::class)

package dev.slne.surf.ai.client.paper

import com.github.shynixn.mccoroutine.folia.launch
import dev.slne.surf.ai.api.AIInstance
import dev.slne.surf.ai.api.model.AiCategory
import dev.slne.surf.ai.api.model.AiCheckResult
import dev.slne.surf.ai.api.model.AiFeedback
import dev.slne.surf.ai.api.model.AiFeedbackResult
import dev.slne.surf.api.core.messages.adventure.buildText
import dev.slne.surf.api.core.messages.adventure.text
import dev.slne.surf.api.paper.dialog.base
import dev.slne.surf.api.paper.dialog.builder.actionButton
import dev.slne.surf.api.paper.dialog.clearDialogs
import dev.slne.surf.api.paper.dialog.dialog
import dev.slne.surf.api.paper.dialog.type
import dev.slne.surf.api.paper.nms.NmsUseWithCaution
import io.papermc.paper.dialog.Dialog
import io.papermc.paper.registry.data.dialog.ActionButton
import io.papermc.paper.registry.data.dialog.DialogBase
import org.bukkit.entity.Player
import java.util.UUID

object FeedbackDialog {
    private fun falsePositiveKey(category: AiCategory) = "fp_${category.name}"
    private fun falseNegativeKey(category: AiCategory) = "fn_${category.name}"

    fun open(player: Player, requestId: UUID, result: AiCheckResult) {
        player.showDialog(create(requestId, result))
    }

    private fun create(requestId: UUID, result: AiCheckResult): Dialog = dialog {
        base {
            title(buildText { primary("KI-Feedback") })
            afterAction(DialogBase.DialogAfterAction.WAIT_FOR_RESPONSE)

            body {
                plainMessage {
                    if (result.scores.isEmpty()) {
                        spacer("Keine Scores für diese Anfrage verfügbar.")
                    } else {
                        for (score in result.scores) {
                            spacer("${score.category}: ${"%.1f".format(score.confidence)}")
                            appendNewline()
                        }
                    }
                }

                input {
                    for (category in AiCategory.entries) {
                        boolean(falsePositiveKey(category)) {
                            label(text("$category - war falsch?"))
                            initial(false)
                        }
                        boolean(falseNegativeKey(category)) {
                            label(text("$category - fehlte?"))
                            initial(false)
                        }
                    }
                }
            }
        }

        type {
            confirmation {
                yes(submitButton(requestId))
                no(cancelButton())
            }
        }
    }

    private fun submitButton(requestId: UUID): ActionButton = actionButton {
        label(text("Absenden"))
        tooltip(text("Feedback absenden"))

        action {
            customPlayerClick { response, player ->
                val falsePositives = AiCategory.entries.filterTo(mutableSetOf()) { response.getBoolean(falsePositiveKey(it)) == true }
                val falseNegatives = AiCategory.entries.filterTo(mutableSetOf()) { response.getBoolean(falseNegativeKey(it)) == true }

                val feedback: AiFeedback = when {
                    falseNegatives.isNotEmpty() -> AiFeedback.FalseNegative(falseNegatives)
                    falsePositives.isNotEmpty() -> AiFeedback.FalsePositive(falsePositives)
                    else -> AiFeedback.Correct
                }

                plugin.launch {
                    val result = AIInstance.INSTANCE.feedback(requestId, feedback)
                    player.showDialog(noticeDialog(result))
                }
            }
        }
    }

    private fun cancelButton(): ActionButton = actionButton {
        label(text("Abbrechen"))
        tooltip(text("Feedback verwerfen"))

        action {
            playerCallback { player ->
                player.clearDialogs()
            }
        }
    }

    private fun noticeDialog(result: AiFeedbackResult): Dialog = dialog {
        base {
            title(buildText { primary("Feedback") })

            body {
                plainMessage {
                    if (result == AiFeedbackResult.ACCEPTED) {
                        spacer("Danke – fließt ins nächste Training ein.")
                    } else {
                        spacer("Anfrage zu alt (abgelaufen).")
                    }
                }
            }
        }

        type {
            notice()
        }
    }
}
