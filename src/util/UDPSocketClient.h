#ifndef UDPSOCKETCLIENT_H
#define UDPSOCKETCLIENT_H

#include <boost/asio.hpp>
#include <string>

class UDPSocketClient {
public:
    UDPSocketClient(const std::string& address, unsigned short port);
    void send(const std::string& message);

private:
    boost::asio::io_context io_context;
    boost::asio::ip::udp::socket socket;
    boost::asio::ip::udp::endpoint receiver_endpoint;
};

#endif // UDPSOCKETCLIENT_H