#include "UDPSocketClient.h"

UDPSocketClient::UDPSocketClient(const std::string& address, unsigned short port)
    : socket(io_context, boost::asio::ip::udp::v4()),
      receiver_endpoint(boost::asio::ip::make_address(address), port) {}

void UDPSocketClient::send(const std::string& message) {
    socket.send_to(boost::asio::buffer(message), receiver_endpoint);
}