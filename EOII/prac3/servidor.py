# Importar la biblioteca de sockets
import socket
from datetime import datetime
import pickle

if __name__ == '__main__':
    # Crear un socket UDP
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Permitir reutilizar la dirección
    udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # Añadir timeout para que recvfrom no bloquee indefinidamente
    udp_socket.settimeout(1.0)  # 1 segundo

    server_address = ('localhost', 10000)

    try:
        udp_socket.bind(server_address)
    except OSError as e:
        print(f"Error al enlazar {server_address}: {e}")
        udp_socket.close()
        raise

    print(f"Servidor UDP escuchando en {udp_socket.getsockname()} (Ctrl+C para salir)")

    try:
        while True:
            try:
                data, client = udp_socket.recvfrom(4096)
            except socket.timeout:
                # timeout: volver a la comprobación del bucle (permite detectar Ctrl+C)
                continue
            except Exception as e:
                print(f"Error al recibir datos: {e}")
                continue

            mensaje = data.decode().strip()
            print(f"Recibido de {client}: {mensaje}")

            # Petición de lista serializada
            if mensaje.strip().lower() == "solicito mensaje serializado":
                # Ejemplo de lista para enviar
                lista = [1, 2, 3, "hola", datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                try:
                    bytes_serializados = pickle.dumps(lista)
                    udp_socket.sendto(bytes_serializados, client)
                except Exception as e:
                    print(f"Error al enviar datos serializados a {client}: {e}")
                continue

            if mensaje.upper() == "HORA":
                respuesta = datetime.now().strftime('%H:%M:%S')
                try:
                    udp_socket.sendto(respuesta.encode(), client)
                except Exception as e:
                    print(f"Error al enviar respuesta a {client}: {e}")
            elif mensaje.upper() == "FIN":
                respuesta = "Adios"
                try:
                    udp_socket.sendto(respuesta.encode(), client)
                except Exception as e:
                    print(f"Error al enviar respuesta a {client}: {e}")
            else:
                respuesta = f"OK: {mensaje}"
                try:
                    udp_socket.sendto(respuesta.encode(), client)
                except Exception as e:
                    print(f"Error al enviar respuesta a {client}: {e}")

    except KeyboardInterrupt:
        print("\nServidor detenido por el usuario.")
    finally:
        udp_socket.close()