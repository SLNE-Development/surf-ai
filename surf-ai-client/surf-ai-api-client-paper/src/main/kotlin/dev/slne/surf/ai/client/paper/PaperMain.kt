package dev.slne.surf.ai.client.paper

import com.github.shynixn.mccoroutine.folia.SuspendingJavaPlugin
import dev.slne.surf.ai.client.common.AiClientInstance
import org.bukkit.plugin.java.JavaPlugin

class PaperMain : SuspendingJavaPlugin() {
    override suspend fun onLoadAsync() {
        AiClientInstance.INSTANCE.onLoad()
    }

    override suspend fun onEnableAsync() {
        AiClientInstance.INSTANCE.onEnable()
        AiFeedbackCommand.register(this)
    }

    override suspend fun onDisableAsync() {
        AiClientInstance.INSTANCE.onDisable()
    }
}

val plugin get() = JavaPlugin.getPlugin(PaperMain::class.java)