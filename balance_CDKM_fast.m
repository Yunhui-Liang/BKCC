function [Y_label, iter_num, obj_min] = balance_CDKM_fast(X, label,c, entropy_type)
% Input
% X d*n data
% label is initial label n*1
% c is the number of clusters
%  F. Nie, J. Xue, D. Wu, R. Wang, H. Li, and X. Li,
% code for "Coordinate descent method for k-means" IEEE Transactions on Pattern Analysis and Machine Intelligence
% Output
% Y_label is the label vector n*1
% obj_max is the objective function value (max)
% iter_num is the number of iteration
%
% CD-KM-TPAMI2021 Coordinate Descent Method for k-means
% The paper "Coordinate Descent Method for k-means" is accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
%
% We reorganized the code， the code "CDKM_fast" is the fast version of "CDKM", which is especially fast for data whose category c is large.
%
% The code "CDKM_code-python" is faster, which is reorganized by Shenfei Pei.
% https://github.com/Sara-Jingjing-Xue/CD-KM-TPAMI
%

[~,n] = size(X);
Y = sparse(1:n,label,1,n,c,n);  % transform label into indicator matrix
last = 0;
iter_num = 0;

% store once
aa=sum(Y,1);
p_k = aa/n;
H = compute_entropy(p_k,entropy_type);
XX = X'*X;

XX2=2*XX;
XXi=diag(XX2)./2;

BBUU= XX2*Y;% BBUU(i,:)
ybby=diag(Y'*BBUU/2);
% tr(Y'AYH)
XXiY = bsxfun(@times, Y, XXi);
yay = diag(Y'*XXiY);
% H_test = ones(1,length(aa));

% compute Initial objective function value
obj_min(1) = H*yay - H*(ybby./aa') ; % max
while any(label ~= last)
    last = label;
    for i = 1:n
        m = label(i) ;
        if aa(m)==1
            continue;
        end

        p_k = (aa+1-Y(i,:))./n;
        H21 = compute_entropy(p_k,entropy_type);
        V21 = yay'+XXi(i)*(1-Y(i,:));
        V22 = ybby'+(BBUU(i,:)+XXi(i)).*(1-Y(i,:));


        p_k = (aa-Y(i,:))./n;
        H11 = compute_entropy(p_k,entropy_type);
%         H(i) = 1./sqrt((aa(i)-1)./n);
        V11 = yay'-XXi(i)*Y(i,:);
        V12 = ybby'-(BBUU(i,:)-XXi(i)).*Y(i,:);
        delta = H21.*(V21-V22./(aa+1-Y(i,:))) - H11.*(V11 - V12./(aa-Y(i,:)));
        [~,q] = min(delta);
        if m~=q
            aa(q) = aa(q) + 1; %  YY(p,p)=Y(:,p)'*Y(:,p);
            aa(m) = aa(m) - 1; %  YY(m,m)=Y(:,m)'*Y(:,m)
            yay(m) = V11(m);
            yay(q) = V21(q);
            ybby(m) = V12(m); %
            ybby(q) = V22(q);
            Y(i,m)=0;
            Y(i,q)=1;
            label(i)=q;
            BBUU(:,m)=BBUU(:,m)-XX2(:,i);%
            BBUU(:,q)=BBUU(:,q)+XX2(:,i);
        end
    end
    iter_num = iter_num+1;
    % compute objective function value
    p_k = aa/n;
    H = compute_entropy(p_k,entropy_type);
    obj_min(iter_num+1) = H*(yay-(ybby./aa')); % min
end
Y_label=label;
end