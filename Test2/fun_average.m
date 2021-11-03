function [cdfA] = fun_average(A)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
[M,N]=size(A);
cdfA=zeros(1,256);
%递增序列统计灰度频数
for i=1:M
    for j=1:N  
    cdfA(A(i,j)+1)=cdfA(A(i,j)+1)+1;  
    end  
end  
%转为频率统计
cdfA=cdfA./(M*N*1.0);

%累积分布函数
cdfA=cumsum(cdfA);  
%将累积分布函数映射到0-255的灰度级
%得到灰度映射向量
cdfA=uint8(255.*cdfA); 
end

