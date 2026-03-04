%**********************************************************
% OBTENCION DE Ku POR PRUEBA Y ERROR
%**********************************************************

Ku = 53.7;          % Ganancia última obtenida experimentalmente
Tu = 0.02;          % Periodo último

%**********************************************************
% VALORES DEL CONTROLADOR PID (Ziegler-Nichols)
Kc = 0.0035 * Ku;
Ti = Tu / 2.2;
Td = Tu / 14;
%**********************************************************

%**********************************************************
% PARAMETROS FIJOS: MODELO, ESCALON, T
a = -0.9599;
b =  0.0365;
T =  0.01;
Escalon = 1;
%**********************************************************

%**********************************************************
% COEFICIENTES DEL PID DISCRETO
c0 = Kc * (1 + T/Ti);
c1 = Kc * (-1 + T/Ti - 2*Td/T);
c2 = Kc * (Td/T);
%**********************************************************

% Inicializar Variables
MUEST = 1000;
y = zeros(MUEST+1,1);
error = zeros(MUEST+1,1);
u = zeros(MUEST+1,1);
t = zeros(MUEST+1,1);

%**********************************************************
% Bucle de Generación de Datos
for i = 1:MUEST+1
    
    % Almacenar valor del instante actual
    t(i) = (i-1)*T;
    
    % Calculo Salida Bucle Cerrado
    if i == 1
        y(i) = 0;
    else
        y(i) = -a*y(i-1) + b*u(i-1);
    end
    
    % Calculo error
    error(i) = Escalon - y(i);
    
    % Calculo accion de control PID
    if i == 1
        u(i) = c0 * error (i);
    elseif i == 2
        u(i) = u(i-1) ...
             + c0*error(i) ...
             + c1*error(i-1);
    else
        u(i) = u(i-1) ...
             + c0*error(i) ...
             + c1*error(i-1) ...
             + c2*error(i-2);
    end
    
end
%**********************************************************

% Graficas
figure;
plot(t,y);
title('Velocidad Motor');
xlabel('Tiempo (s)');
ylabel('Salida');

figure;
plot(t,u,'r');
title('Acción de Control - PID');
xlabel('Tiempo (s)');
ylabel('u(k)');
