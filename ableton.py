import socket
import json


def send_osc_message(command, params=None, host='localhost', port=11002):
    """
    Send a message to the AbletonOSC server.
    
    :param command: The OSC command (e.g., "/live/test")
    :param params: List of parameters for the command (optional)
    :param host: The host address of the server (default: 'localhost')
    :param port: The port number of the server (default: 11001)
    :return: The server's response as a Python object
    """
    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    try:
        # Prepare the message
        message = {
            "command": command,
            "params": params or []
        }
        
        # Convert the message to JSON
        json_message = json.dumps(message)
        
        # Send the message
        sock.sendto(json_message.encode('utf-8'), (host, port))
        
        # Wait for a response (adjust the buffer size if needed)
        response, _ = sock.recvfrom(1024)
        
        # Parse and return the response
        return json.loads(response.decode('utf-8'))
    
    finally:
        # Always close the socket
        sock.close()

# Example usage
if __name__ == "__main__":
    # Example 1: Simple command without parameters
    result = send_osc_message("/live/test")
    print("Result of /live/test:", result)
    
    # Example 2: Command with parameters
    result = send_osc_message("/live/api/set/log_level", ["debug"])
    print("Result of setting log level:", result)
    
    # Example 3: Get Ableton Live version
    result = send_osc_message("/live/application/get/version")
    print("Ableton Live version:", result)