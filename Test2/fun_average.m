function [cdfA] = fun_average(A)
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
[M,N]=size(A);
cdfA=zeros(1,256);
%��������ͳ�ƻҶ�Ƶ��
for i=1:M
    for j=1:N  
    cdfA(A(i,j)+1)=cdfA(A(i,j)+1)+1;  
    end  
end  
%תΪƵ��ͳ��
cdfA=cdfA./(M*N*1.0);

%�ۻ��ֲ�����
cdfA=cumsum(cdfA);  
%���ۻ��ֲ�����ӳ�䵽0-255�ĻҶȼ�
%�õ��Ҷ�ӳ������
cdfA=uint8(255.*cdfA); 
end

