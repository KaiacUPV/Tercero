%**********************************************************
% Generación de datos para la identificación LSQ
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

% Cargar el fichero de datos PRBS13.txt
load prbs13.txt
valor_medio=5.0; desviacion=1.0; estiramiento=5.0;

%Bucle de Generación de Datos
for i=1:MUEST
    
    tic;
    
    % Almacenar valor del instante actual
    t(i)=i*periodo;
    
    % Lectura por los canales de entrada de la USB6001
    Lectura = Sesion.inputSingleScan();
    if i==1 offset=Lectura; end
    Vel(i)=Lectura-offset;
   
        
    % Acción de control
    u(i)=valor_medio+sign(prbs13(ceil(i/estiramiento))-0.5)*desviacion; 
       
    % Escritura en el canal 0 de la USB6001 del valor u
    outputSingleScan(Sesion,u(i));
           
    % Espera hasta el siguiente múltiplo del periodo
    while toc<periodo;
    end;
end;

% Poner la salida a cero
outputSingleScan(Sesion,0);
delete(Sesion);

% Almacenar datos en un variables t2, Vel2, u2
t2=t'; Vel2=Vel'; u2=u';
% t2=t'/1000;

figure;
plot(t2,Vel2);
hold;
plot(t2,u2,'r');