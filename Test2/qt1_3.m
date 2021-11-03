c=imread('2.3.tif');

[a1]=fun_match(c,50,100);
[a2]=fun_match(c,100,100);
[a3]=fun_match(c,150,100);
[a4]=fun_match(c,200,100);

[M,N]=size(c);
b=c;%均衡化图
histc=fun_average(c);
for i=1:M
    for j=1:N
        b(i,j)=histc(b(i,j)+1);  
    end  
end  

subplot(2,3,1);
imshow(c);
title('原图');
subplot(2,3,2);
imshow(a1);
title('以50为中心的高斯分布直方图匹配');
subplot(2,3,3);
imshow(a2);
title('以100为中心的高斯分布直方图匹配');
subplot(2,3,4);
imshow(a3);
title('以150为中心的高斯分布直方图匹配');
subplot(2,3,5);
imshow(a4);
title('以200为中心的高斯分布直方图匹配');
subplot(2,3,6);
imshow(b);
title('均衡化后的图');