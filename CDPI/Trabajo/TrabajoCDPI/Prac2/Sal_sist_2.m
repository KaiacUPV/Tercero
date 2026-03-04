% ===============================
% SELECCIONAR MODELO A VALIDAR
% ===============================

theta = theta3;   % <-- cambiar aquí: theta1, theta2, theta3 o theta4

na = 2;           % orden correspondiente al modelo elegido
nb = 2;           % orden correspondiente al modelo elegido
d = 1;

% ===============================
% PARÁMETROS DE SIMULACIÓN
% ===============================

T = 0.01;                       % periodo de muestreo
tiempo_fin = t1(end);           % mismo tiempo que experimento real
Entrada = u1(1);                % mismo escalón usado en Prueba.m

% Separar parámetros
a = theta(1:na);
b = theta(na+1:na+nb);

% Inicializar vectores
N = length(u1);
sal = zeros(1,N);
in = u1;             % usamos exactamente la misma entrada real
tiempo = t1;

% ===============================
% SIMULACIÓN GENÉRICA
% ===============================

for k = 1:N
    
    suma_salidas = 0;
    suma_entradas = 0;
    
    % Parte de salidas pasadas
    for i = 1:na
        if k-i > 0
            suma_salidas = suma_salidas - a(i)*sal(k-i);
        end
    end
    
    % Parte de entradas pasadas
    for j = 1:nb
        if k-d-j+1 > 0
            suma_entradas = suma_entradas + b(j)*in(k-d-j+1);
        end
    end
    
    sal(k) = suma_salidas + suma_entradas;
    
end

% ===============================
% GRÁFICA DE VALIDACIÓN
% ===============================

figure;
plot(t1,Vel1,'b'); hold on;
plot(tiempo,sal,'r');
legend('Motor real','Modelo');
title('Validación del modelo');