#ifndef WEBSOCKET_CLIENT_H
#define WEBSOCKET_CLIENT_H

#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>
#include <websocketpp/common/thread.hpp>
#include <websocketpp/common/memory.hpp>

typedef websocketpp::client<websocketpp::config::asio_client> client;

class WebSocketClient {
public:
    WebSocketClient();
    ~WebSocketClient();

    void connect(const std::string& uri);
    void send(const std::string& message);
    void sendBinary(const void* data, size_t len);
    void setMessageHandler(std::function<void(const std::string&)> handler);


private:
    client m_client;
    websocketpp::connection_hdl m_hdl;
    std::thread m_thread;
    bool m_connected;
    std::function<void(const std::string&)> m_messageHandler;
};

#endif // WEBSOCKET_CLIENT_H