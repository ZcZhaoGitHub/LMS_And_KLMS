function [ y ] = ker_eval( X1,X2,ker_type,ker_param )
N1 = size(X1,2);
N2 = size(X2,2);
%% kernel Gauss
%compare strings,if ture return 1,otherwise 0
if strcmp(ker_type,'Gauss')
    
    if N1 == N2
        y = (exp(-sum((X1-X2).^2)*ker_param))';
    elseif N1 == 1
        y = (exp(-sum((X1*ones(1,N2)-X2).^2)*ker_param))';
    elseif N2 == 1
        y = (exp(-sum((X1-X2*ones(1,N1)).^2)*ker_param))';
    else
        warning('error dimension--')
    end
    return
    
end
%% kernel Ploy
if strcmp(ker_type,'Ploy')
    
    if N1 == N2
        y = ((1 + sum(X1.*X2)).^ker_param)';
    elseif N1 == 1
        y = ((1 + X1'*X2).^ker_param)';
    elseif N2 == 1
        y = ((1 + X2'*X1).^ker_param)';
    else
        warning('error dimension--')
    end
    return
    
end

warning('no such kernel')

return


