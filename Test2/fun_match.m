function [img] = fun_match(a,miu,sigma)
%ֱ��ͼƥ�亯����miu��sigma��������������ɸ�˹�ֲ�

c=a;%��������ԭͼ����Ϊ�Ա�

%���ݸ�˹��������һ�����ʷֲ�����
x = 0:1:255;
y0 = 1/(sqrt(2*pi)*sigma)*exp(-(x-miu).^2/(2*sigma^2));
b=sum(y0);%���������ʷֲ����Ƿ�Ϊ0
y0=y0/b;  %ʹ�ø��ʷֲ���������Ϊ1

y0=cumsum(y0);
y0=uint8(255.*y0);%y0����Ҫƥ�����õĻҶ�ӳ�� 

cdfa=fun_average(a);%%a���⻯�ĻҶ�ӳ��

%��ԭͼ�Ҷ�ӳ�����µ�ֱ��ͼ��ӳ����ϵ�������Ҷȶ�Ӧ
for i=1:256
    %Ѱ�һҶ�ֵ����Ķ�Ӧֵ
    k=min(abs(cdfa(i)-y0));
    for j=1:256
        if abs(cdfa(i)-y0(j))==k
           cdfa(i)=j-1;    %%%%%�Ҷ���ӳ��
            break
        end
        
    end
end
%%�õ��µĴ�a��y0�Ҷ�ӳ����󣨴�r��z��

%ͨ������ӳ���������ͼ��ת��Ϊ���ͼ��
[M,N]=size(a);
for ii=1:M
    for jj=1:N
        a(ii,jj)=cdfa(a(ii,jj)+1);  
    end  
end  
img=a;

end
