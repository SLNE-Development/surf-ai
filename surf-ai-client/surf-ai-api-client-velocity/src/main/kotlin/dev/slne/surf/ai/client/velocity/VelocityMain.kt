package dev.slne.surf.ai.client.velocity

import com.github.shynixn.mccoroutine.velocity.SuspendingPluginContainer
import com.google.inject.Inject
import com.velocitypowered.api.event.Subscribe
import com.velocitypowered.api.event.proxy.ProxyInitializeEvent
import com.velocitypowered.api.event.proxy.ProxyShutdownEvent
import com.velocitypowered.api.plugin.PluginContainer
import com.velocitypowered.api.plugin.annotation.DataDirectory
import com.velocitypowered.api.proxy.ProxyServer
import java.nio.file.Path

class VelocityMain @Inject constructor(
    val proxy: ProxyServer,
    val container: PluginContainer,
    suspendingContainer: SuspendingPluginContainer,
    @DataDirectory val dataPath: Path
) {
    init {
        INSTANCE = this
        suspendingContainer.initialize(this)
    }

    @Subscribe
    fun onProxyInitialize(event: ProxyInitializeEvent) {
    }

    @Subscribe
    fun onShutdown(event: ProxyShutdownEvent) {
    }

    companion object {
        lateinit var INSTANCE: VelocityMain
            private set
    }
}

val plugin get() = VelocityMain.INSTANCE
val container get() = plugin.container
val proxy get() = plugin.proxy
