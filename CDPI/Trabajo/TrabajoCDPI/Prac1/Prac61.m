%**********************************************************
% Programa de CONTROL PROPORCIONAL de la Velocidad del Motor
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

% Se definen la constante K del control proporcional 
% K for the Proportional Controller
% *************************************
Kc= 53.7*0.15 ;
% *************************************

% Se definen la Referencia
Escalon= 1;
Ref=ones(1,MUEST)*Escalon; 


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
    u(i)=Kc*error(i); 
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

