@file:Suppress("UnstableApiUsage")
@file:OptIn(NmsUseWithCaution::class)

package dev.slne.surf.ai.client.paper

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
import net.kyori.adventure.audience.Audience
import net.kyori.adventure.text.Component
import net.kyori.adventure.text.event.HoverEvent

object ExampleDialog {
    fun renderDialogShowComponent(
        text: Component
    ): Component = buildText {
        append(text)

        clickCallback { audience ->
            audience.showDialog(create(audience))
        }

        hoverEvent(HoverEvent.showText(buildText {
            spacer("Information about this component")
        }))
    }

    private const val EXAMPLE_BOOLEAN_KEY = "example_boolean_key"

    fun create(audience: Audience): Dialog = dialog {
        base {
            title(buildText {
                primary("Example Dialog")
            })
            // Set to WAIT_FOR_RESPONSE if you need to perform some kind of suspending action in the action button
            afterAction(DialogBase.DialogAfterAction.NONE)

            body {
                // only use one plainMessage, work with appendNewline() or appendNewline(amount) to create new lines
                plainMessage {
                    spacer("This is an example dialog with a confirmation button and a cancel button.")
                    appendNewline(2)
                    spacer("Click the confirm button to perform an action, or click the cancel button to close the dialog.")
                }

                input {
                    boolean(EXAMPLE_BOOLEAN_KEY) {
                        label(text("Example Boolean Input"))
                        // set the initial value of the boolean input to a useful value
                        initial(true)
                    }
                }
            }
        }

        type {
            confirmation {
                yes(confirmationButton())
                no(cancelButton())
            }
        }
    }

    private fun confirmationButton(): ActionButton = actionButton {
        label(text("Confirm"))
        tooltip(text("Click to confirm your action"))

        action {
            customPlayerClick { response, player ->
                val exampleBooleanValue = response.getBoolean(EXAMPLE_BOOLEAN_KEY)

                player.sendMessage(buildText {
                    primary("You clicked the confirm button. The value of the boolean input is: ")
                    secondary("$exampleBooleanValue")
                })

                player.clearDialogs()
            }
        }
    }

    private fun cancelButton(): ActionButton = actionButton {
        label(text("Cancel"))
        tooltip(text("Click to cancel your action"))

        action {
            playerCallback { player ->
                player.clearDialogs()
            }
        }
    }
}