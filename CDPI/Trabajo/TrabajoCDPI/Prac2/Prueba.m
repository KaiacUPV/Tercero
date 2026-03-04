%**********************************************************
% Programa de Prueba del Motor de Corriente Continua
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
MUEST=1000; Vel=[]; u=[]; t=[];
periodo=0.010;
T=periodo;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ASIGNAR VALOR DEL ESCALON (entre 0..8) 
v_ent=  5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Bucle de Generación de Datos
for i=1:MUEST
    
    tic;
    
    % Almacenar valor del instante actual
    t(i)=i*periodo;
    
    % Lectura por los canales de entrada de la USB6001
    Lectura = Sesion.inputSingleScan();
    if i==1 offset=Lectura; end
    Vel(i)=Lectura-offset;
    
    % Escritura en el canal 0 de la USB6001 del valor u
    outputSingleScan(Sesion,v_ent);
    
       
    % Espera hasta el siguiente múltiplo del periodo
    while toc<periodo;
    end;
end;

% Poner la salida a cero
outputSingleScan(Sesion,0);
delete(Sesion);

% Almacenar datos en variables t1, Vel1, u1
t1=t'; Vel1=Vel'; u1=v_ent*ones(length(Vel1),1);
% t1=t'/1000;

figure;
plot(t1,Vel1);
hold;
plot(t1,u1,'r');
