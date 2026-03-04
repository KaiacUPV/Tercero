%**********************************************************
% CONTROL PROPORCIONAL INTEGRAL de la Velocidad del Motor
%**********************************************************

%Exploración del Hardware disponible
Dispositivos=daq.getDevices
Dispositivos=Dispositivos(1); %El primer dispositivo es la TAD USB-6001

%Creación de la sesión
Sesion=daq.createSession(Dispositivos.Vendor.ID)

%Inicialización de los canales de lectura/escritura
CanalIn = addAnalogInputChannel(Sesion, Dispositivos.ID, 0, 'Voltage');
CanalIn.InputType = 'SingleEnded';

CanalOut = addAnalogOutputChannel(Sesion, Dispositivos.ID, 0, 'Voltage');


% Poner la salida a cero
outputSingleScan(Sesion,0);


%Inicializar Variables
MUEST=1000; Vel=[]; u=[]; error=[]; Ref=[]; t=[];
periodo=0.010;
T=periodo;

% Se definen la Referencia
Escalon=  1;
Ref=ones(1,MUEST)*Escalon; 

% Parametros del Controlador PI
% *************************************
Kc = 0.0035 * Ku;
Ti = Tu / 2.2;
Td = Tu / 14;
% *************************************
% COEFICIENTES DEL PID DISCRETO
c0 = Kc * (1 + T/Ti);
c1 = Kc * (-1 + T/Ti - 2*Td/T);
c2 = Kc * (Td/T);

p=Kc*(1-(T/Ti));

%Bucle de Generación de Datos
for i=1:MUEST
    
    tic;
    
    % Almacenar valor del instante actual
    t(i)=i*periodo;
    
    % Lectura por los canales de entrada de la USB6001
    Lectura = Sesion.inputSingleScan();
    if i==1 offset=Lectura; end
    Vel(i)=Lectura-offset;
     
    % Cálculo del error / Error computation
    % *************************************************
    error(i)= Ref(i) - Vel(i);
    % *************************************************
    
    % Cálculo de la acción de control / Control action
    % *************************************************
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
    
    % *************************************************
      
     % Saturar acción de control (proteger motor)
    if (u(i)>10.0) u(i)=10.0;  end;  
    if (u(i)<0.0) u(i)=0.0;  end;   
      
    % Escritura en el canal 0 de la USB6001 del valor u
    outputSingleScan(Sesion,u(i));
    
    % Espera hasta el siguiente múltiplo del periodo
    while toc<periodo;
    end;
end;

% Poner la salida a cero
outputSingleScan(Sesion,0);
delete(Sesion);

% Almacenar datos en un fichero de texto
%t=t'/1000; %datos1=[Ref',Vel',error',u'];
Vel=Vel'; error=error'; u=u'; Ref=Ref';t=t';

figure;
plot(t,Vel);
hold;
plot(t,Ref,'k');
figure;
plot(t,u,'r');
figure;
plot(t,error,'k');