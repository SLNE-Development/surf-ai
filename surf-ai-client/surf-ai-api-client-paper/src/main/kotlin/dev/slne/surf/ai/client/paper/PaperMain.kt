package dev.slne.surf.ai.client.paper

import com.github.shynixn.mccoroutine.folia.SuspendingJavaPlugin
import org.bukkit.plugin.java.JavaPlugin

class PaperMain : SuspendingJavaPlugin() {
    override suspend fun onLoadAsync() {
    }

    override suspend fun onEnableAsync() {
    }

    override suspend fun onDisableAsync() {
    }
}

val plugin get() = JavaPlugin.getPlugin(PaperMain::class.java)