#include "WebSocketClient.h"
#include <iostream>

WebSocketClient::WebSocketClient() : m_connected(false) {
    m_client.init_asio();


    m_client.clear_access_channels(websocketpp::log::alevel::all);
    m_client.clear_error_channels(websocketpp::log::elevel::all);

    
    m_client.set_open_handler([this](websocketpp::connection_hdl hdl) {
        m_hdl = hdl;
        m_connected = true;
    });
    m_client.set_fail_handler([this](websocketpp::connection_hdl hdl) {
        m_connected = false;
    });
    m_client.set_close_handler([this](websocketpp::connection_hdl hdl) {
        m_connected = false;
    });
    m_client.set_message_handler([this](websocketpp::connection_hdl, client::message_ptr msg) {
        if (m_messageHandler) {
            m_messageHandler(msg->get_payload());
        }
    });
}

WebSocketClient::~WebSocketClient() {
    m_client.stop();
    if (m_thread.joinable()) {
        m_thread.join();
    }
}

void WebSocketClient::connect(const std::string& uri) {
    websocketpp::lib::error_code ec;
    client::connection_ptr con = m_client.get_connection(uri, ec);
    if (ec) {
        std::cout << "Could not create connection because: " << ec.message() << std::endl;
        return;
    }

    m_client.connect(con);
    m_thread = std::thread([this]() { m_client.run(); });
}

void WebSocketClient::send(const std::string& message) {
    if (m_connected) {
        m_client.send(m_hdl, message, websocketpp::frame::opcode::text);
    }
}

void WebSocketClient::sendBinary(const void* data, size_t len) {
    if (m_connected) {
        m_client.send(m_hdl, data, len, websocketpp::frame::opcode::binary);
    }
}

void WebSocketClient::setMessageHandler(std::function<void(const std::string&)> handler) {
    m_messageHandler = handler;
}

void WebSocketClient::close() {
    if (m_connected) {
        m_client.close(m_hdl, websocketpp::close::status::going_away, "");
        if (m_thread.joinable()) {
            m_thread.join();
        }
        m_connected = false;
    }
}