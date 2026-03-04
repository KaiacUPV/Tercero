%%

%**********************************************************

% OBTENCION SALIDA DE UN MODELO

%**********************************************************

%open('salida_real.fig')

plot(tiempo,salida_ante_escalon)

theta = theta_ex3; % elegir modelo

% na nb na nb

na = 2; % theta1 1 1 theta2 2 1

nb = 2; % theta3 2 2 theta4 3 3

d = 2; % theta5 4 4 theta6 5 5

% theta7 10 10 theta8 12 12

 

a = theta(1:na);

b = theta(na+1:na+nb);

%**********************************************************

% PERIODO DE MUESTREO T

Ti= T;

%**********************************************************

%Inicializar Variables

MUEST=length(tiempo); % mismo tamaño que datos reales;

y=[];

error=[];

u=[];

t=[];

% Fijando una entrada al modelo Escalon 1

u=ones(1,MUEST+1); %cambiarlo a 5

%Bucle de Generación de Datos

for i=1:MUEST

 

% Almacenar valor del instante actual

t(i)=((i-1)*Ti);

 

suma_salidas = 0;

suma_entradas = 0;

%Calculo Salida Bucle Abierto

% Parte salidas anteriores

for k = 1:na

if i-k > 0

suma_salidas = suma_salidas - a(k)*y(i-k);

end

end

 

% Parte entradas anteriores

for k = 1:nb

if i-d-k+1 > 0

suma_entradas = suma_entradas + b(k)*u(i-d-k+1);

end

end

 

y(i) = suma_salidas + suma_entradas;

 

 

 

end;

% Almacenar datos en un fichero de texto

t=t'; y=y'; u=u';

hold on

plot(t,y,'r','LineWidth',0.5)

legend('Sistema real','Modelo identificado')