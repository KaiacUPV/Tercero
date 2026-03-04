% ASIGNAR VALORES A LOS PARAMÉTROS

tiempo_fin= 10;      % Tiempo final de simulación 
Entrada=5;           % Entrada constante 
T= 0.01;             % Periodo de Muestreo
d= 1;                % Retardo del sistema
a= [-1 ;          % Parámetro ai del sistema  
b= 0.0345;           % Parámetro bi del sistema

k=1;
t=0;

for t=0:T:tiempo_fin
    
    in(k)=Entrada;
    
    % Implementación de la ecuación en diferencias
    if k==1
        sal(k)=0;   % condición inicial
    else
        if (k-d)<=0
            u_delay=0;   % entrada retrasada inicial
        else
            u_delay=in(k-d);
        end
        
        sal(k)= -a*sal(k-1) + b*u_delay;
    end
    
    tiempo(k)=t;
    k=k+1;
end

% Visualización
figure;
plot(tiempo,sal,'b');
hold;
plot(tiempo,in,'r');
legend('Salida','Entrada');
xlabel('Tiempo (s)');
ylabel('Amplitud');
grid on;