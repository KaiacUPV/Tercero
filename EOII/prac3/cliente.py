# Importar la biblioteca de sockets
import socket
import pickle

def main():
    server_address = ('localhost', 10000)
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.settimeout(5.0)  # 5 segundos de timeout para recvfrom

    try:
        while True:
            mensaje = input("Mensaje (escribe FIN para terminar): ").strip()
            if not mensaje:
                continue

            try:
                udp_socket.sendto(mensaje.encode(), server_address)
            except Exception as e:
                print(f"Error al enviar: {e}")
                continue

            try:
                data, _ = udp_socket.recvfrom(65536)  # buffer grande para datos serializados
                # Intentar deserializar con pickle; si falla, tratar como texto
                try:
                    obj = pickle.loads(data)
                    print("Respuesta serializada (deserializada):", obj)
                except Exception:
                    try:
                        print("Respuesta del servidor:", data.decode())
                    except Exception:
                        print("Respuesta recibida (no decodificable).")
            except socket.timeout:
                print("El servidor no ha respondido en 5 segundos (timeout).")
            except Exception as e:
                print(f"Error al recibir respuesta: {e}")

            if mensaje.upper() == "FIN":
                print("Cliente termina.")
                break
    finally:
        udp_socket.close()

if __name__ == "__main__":
    main()