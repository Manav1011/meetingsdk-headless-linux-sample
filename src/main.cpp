#include <csignal>
#include <glib.h>
#include "Config.h"
#include "Zoom.h"
#include "util/WebSocketClient.h"
#include "util/UDPSocketClient.h"

WebSocketClient* g_webSocketClient = nullptr;
UDPSocketClient* g_udpSocketClient = nullptr;

/**
 *  Callback fired atexit()
 */
void onExit() {
    auto* zoom = &Zoom::getInstance();
    zoom->leave();
    zoom->clean();

    cout << "exiting..." << endl;
}

/**
 * Callback fired when a signal is trapped
 * @param signal type of signal
 */
void onSignal(int signal) {
    onExit();
    _Exit(signal);
}


/**
 * Callback for glib event loop
 * @param data event data
 * @return always TRUE
 */
gboolean onTimeout (gpointer data) {
    return TRUE;
}

/**
 * Run the Zoom Meeting Bot
 * @param argc argument count
 * @param argv argument vector
 * @return SDKError
 */
SDKError run(int argc, char** argv) {
    SDKError err{SDKERR_SUCCESS};
    auto* zoom = &Zoom::getInstance();

    signal(SIGINT, onSignal);
    signal(SIGTERM, onSignal);

    atexit(onExit);

    g_webSocketClient = new WebSocketClient();
    g_webSocketClient->connect("ws://192.168.7.195:8001");
    g_udpSocketClient = new UDPSocketClient("192.168.7.195", 8080);

    // g_webSocketClient->setMessageHandler([zoom](const std::string& message) {
    //     std::cout << "Received message: " << message << std::endl;
    //     SDKError err = zoom->sendChatMessage(message);
    //     if (err != ZOOM_SDK_NAMESPACE::SDKERR_SUCCESS)
    //     {
    //         std::cout << "Failed to send a chat message: " << err << std::endl;
    //     }
    //     if (Zoom::hasError(err, "send chat message")) {
    //         Log::error("Failed to send chat message");
    //     }
    // });
    
    // read the CLI and config.ini file
    err = zoom->config(argc, argv);
    if (Zoom::hasError(err, "configure"))
        return err;

    // initialize the Zoom SDK
    err = zoom->init();
    if(Zoom::hasError(err, "initialize"))
        return err;

    // authorize with the Zoom SDK
    err = zoom->auth();
    if (Zoom::hasError(err, "authorize"))
        return err;

    return err;
}

int main(int argc, char **argv) {

    // Run the Meeting Bot
    SDKError err = run(argc, argv);

    if (Zoom::hasError(err))
        return err;

    // Use an event loop to receive callbacks
    GMainLoop* eventLoop;
    eventLoop = g_main_loop_new(NULL, FALSE);
    g_timeout_add(100, onTimeout, eventLoop);
    g_main_loop_run(eventLoop);

    return err;
}


