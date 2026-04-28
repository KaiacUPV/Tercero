import socket

# Configuración
UDP_IP = "0.0.0.0" 
UDP_PORT = 8888 # Asegúrate de que sea el mismo en HyperIMU

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Esperando datos de rotación en el puerto {UDP_PORT}...")
print("Si no aparece nada, revisa el Firewall de Windows.")

try:
    while True:
        data, addr = sock.recvfrom(1024)
        # HyperIMU envía los datos como texto (string) separados por comas
        mensaje = data.decode('utf-8')
        print(f"Datos recibidos: {mensaje}")
except KeyboardInterrupt:
    print("\nDetenido por el usuario")
finally:
    sock.close()