import socket

SERVER_ADDRESS = ('127.0.0.1', 39251)

def receive_messages():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(SERVER_ADDRESS)
        s.listen()
        print(f'Listening on {SERVER_ADDRESS}')
        conn, addr = s.accept()
        with conn:
            print(f'Connected by {addr}')
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                message = data.decode()
                print(message)

receive_messages()