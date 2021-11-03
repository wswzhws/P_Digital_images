%qt4
clc
clear
A=imread('ͼ2.4.tif');
%�Ծ��������չ,���������Եֵ
B=uint8(zeros(258,258));
B(2:257,2:257)=A;
B(1,2:257)=A(1,1:256);
B(258,2:257)=A(256,1:256);
B(2:257,1)=A(1:256,1);
B(2:257,258)=A(1:256,256);
B(1,1)=A(1,1);
B(1,258)=A(1,256);
B(258,1)=A(256,1);
B(258,258)=A(256,256);
C=padarray(A,[2,2],'symmetric');%�ú���������չ
D=padarray(A,[3,3],'symmetric');%�ú���������չ������Ա߽�ֵ���и�ֵ

%��3*3������оֲ����⻯
for i=2:257
    for j=2:257
        z=B(i-1:i+1,j-1:j+1);
        cdfz=fun_average(z);
        for ii=1:3
            for jj=1:3
                z(ii,jj)=cdfz(z(ii,jj)+1);
            end
        end
        B(i,j)=z(2,2);
    end
end
%��5*5��������ֲ�ƽ�⻰
for i=3:258
    for j=3:258
        z=C(i-2:i+2,j-2:j+2);
        cdfz=fun_average(z);
        for ii=1:5
            for jj=1:5
                z(ii,jj)=cdfz(z(ii,jj)+1);
            end
        end
            B(i,j)=z(3,3);
    end
end
%7*7������д���
for i=4:259
    for j=4:259
        z=D(i-3:i+3,j-3:j+3);
        cdfz=fun_average(z);
        for ii=1:7
            for jj=1:7
                z(ii,jj)=cdfz(z(ii,jj)+1);
            end
        end
            B(i,j)=z(4,4);
    end
end

A_result1=B(2:257,2:257);

subplot(1,4,1);
imshow(A);
title('ԭʼͼ��');
subplot(1,4,2);
imshow(A_result1);
title('3*3��5*5��7*7�����Դ˴����ֲ����⻯ͼ��');
