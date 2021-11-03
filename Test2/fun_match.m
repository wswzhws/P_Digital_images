function [img] = fun_match(a,miu,sigma)
%直方图匹配函数，miu，sigma输入变量用来生成高斯分布

c=a;%用来保存原图以作为对比

%根据高斯函数建立一个概率分布函数
x = 0:1:255;
y0 = 1/(sqrt(2*pi)*sigma)*exp(-(x-miu).^2/(2*sigma^2));
b=sum(y0);%用来检测概率分布和是否为0
y0=y0/b;  %使得概率分布函数积分为1

y0=cumsum(y0);
y0=uint8(255.*y0);%y0是想要匹配所得的灰度映射 

cdfa=fun_average(a);%%a均衡化的灰度映射

%将原图灰度映射与新的直方图的映射联系起来，灰度对应
for i=1:256
    %寻找灰度值最近的对应值
    k=min(abs(cdfa(i)-y0));
    for j=1:256
        if abs(cdfa(i)-y0(j))==k
           cdfa(i)=j-1;    %%%%%灰度逆映射
            break
        end
        
    end
end
%%得到新的从a到y0灰度映射矩阵（从r到z）

%通过最新映射矩阵将输入图像转化为输出图像
[M,N]=size(a);
for ii=1:M
    for jj=1:N
        a(ii,jj)=cdfa(a(ii,jj)+1);  
    end  
end  
img=a;

end
