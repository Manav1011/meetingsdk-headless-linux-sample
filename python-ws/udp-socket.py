import asyncio

class EchoServerProtocol:
    def connection_made(self, transport):
        self.transport = transport
        print("Connection established")

    def datagram_received(self, data, addr):
        try:
            message = data.decode()
            asyncio.create_task(self.handle_datagram(message, addr))
        except Exception as e:
            print(f"Error decoding message: {e}")

    async def handle_datagram(self, message, addr):
        print(f"Handling message: {message!r} from {addr}")

async def main():
    print("Starting UDP server")

    # Get a reference to the event loop as we plan to use
    loop = asyncio.get_running_loop()

    # One protocol instance will be created to serve all client requests.
    transport, protocol = await loop.create_datagram_endpoint(
        EchoServerProtocol,
        local_addr=('192.168.7.195', 8080))

    try:
        # Keep the event loop running to listen for incoming UDP packets
        print("Server is running...")
        await asyncio.sleep(3600)  # Keep server running for 1 hour (or until you stop it)
    except asyncio.CancelledError:
        pass

# Run the asyncio event loop
asyncio.run(main())
