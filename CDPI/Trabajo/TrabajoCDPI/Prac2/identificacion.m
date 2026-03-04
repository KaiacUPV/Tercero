function theta4 = identificacion(y,u,na,nb,d,H)

% Longitud total de datos
N = length(y);

% Primer instante válido
M = H + 1 + max(na, nb-1+d);

% Inicializar matriz Phi
fi = [];

% Construcción de Phi
for k = M:N
    
    fila = [];
    
    % Parte de salidas anteriores (-y)
    for i = 1:na
        fila = [fila -y(k-i)];
    end
    
    % Parte de entradas anteriores (u)
    for j = 0:nb-1
        fila = [fila u(k-d-j)];
    end
    
    % Añadir fila a Phi
    fi = [fi; fila];
    
end

% Vector Y
Y = y(M:N);

% Cálculo de parámetros por Mínimos Cuadrados
theta4 = inv(fi'*fi)*fi'*Y;

end