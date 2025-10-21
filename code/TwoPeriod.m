% TwoPeriod.m
% Aiyagari model with human capital
% Oct 19, 2024 By YKL
%close all;
clear
%% Parameters

%Kai_n = 2.34/3*0.25;
Kai_n = 2.34/3*0.25;
Kai_e = 2.34*0.5/3*0.25; 
e_l = 1/6*3*2;
e_h = 1/3*3*2;
hbar = 6;
delta_h = 0.1;

lambda=0.4;
r=0;
w=1;
discount=0.95;

a_upper=1+lambda;
a_lower=0;
n_a=101;
agrid=linspace(a_lower,a_upper,n_a);
agrid=agrid(:); %n_a*1

zscaleupper=4;
zbar_Fcn = @(a) (exp(Kai_n)-1)*(1+r)*a/w;
z_lower=1e-4;
z_upper=zscaleupper*zbar_Fcn(a_upper)/(1-lambda);
n_z= 101;
zgrid = linspace(z_lower,z_upper,n_z);
zgrid=zgrid(:); %n_z*1

mean_z = 0;
sigma_z = 0.001;%0.3;%0.01;
zdensity = lognpdf(zgrid,mean_z,sigma_z);
z_prob = zdensity./sum(zdensity);


h_lower=0;
h_upper=hbar;
n_h = 101;
hgrid=linspace(h_lower,h_upper,n_h); %1*n_h

h_M = 2;
h_H = 4.5;

%% w*z*x(h) matrix n_z*n_h
wzx=zeros(n_z,n_h); 
for iz=1:n_z
    for ih=1:n_h
        if hgrid(ih)<h_M
        wzx(iz,ih)=w*zgrid(iz)*(1-lambda);
        elseif hgrid(ih)>h_H
        wzx(iz,ih)=w*zgrid(iz)*(1+lambda);
        else
        wzx(iz,ih)=w*zgrid(iz);
        end
    end
end

%% Period-2 Value Function n_a*n_h
EV_2=zeros(n_a,n_h);
for ia=1:n_a
    avalue=agrid(ia);
    V_2 = log(wzx+(1+r)*avalue) - Kai_n;
    V_notwork = log((1+r)*avalue);
    V_2(V_2<V_notwork) = V_notwork;
    EV_2temp=V_2'*z_prob;
    EV_2(ia,:)=EV_2temp';  % 1*n_h
end

% figure
% ih=round(n_h/2);
% ia=round(n_a/2);
% subplot(2,1,1)
% plot(agrid,EV_2(:,ih),'k')
% xlabel('a')
% subplot(2,1,2)
% plot(hgrid,EV_2(ia,:),'k')
% xlabel('h')

%% Period-1 Decision Rule
V1=zeros(n_z,n_h);
choice=zeros(n_z,n_h);
cstar=zeros(n_z,n_h);

h2_0=(1-delta_h)*hgrid;

% avalue needs to be such that if households know z'=1, it will for sure
% work even if they have the highest possible asset next period in
% equilibrium
avalue=a_upper/2*2;
c1_upper= (w*z_upper*(1+lambda)+(1+r)*avalue+w*1*(1+lambda)/(1+r))/(1+discount);
a2_upper = w*z_upper*(1+lambda)+(1+r)*avalue - c1_upper;
z2_cutoff = zbar_Fcn(a2_upper);
% needs z'=1>z2_cutoff


a2_nowork=(1+r)*avalue;
a2_work=wzx+(1+r)*avalue;

% condition for e_h,e_l,Kai_e and Kai_n: needs LHS>RHS
LHS = exp(Kai_e*e_h/(1+discount));
RHS = exp(Kai_e*e_l/(1+discount))/(exp(Kai_e*e_l/(1+discount))+exp(Kai_n/(1+discount))-exp((Kai_e*e_l+Kai_n)/(1+discount)));

n_c=1001;
cgrid_nowork=linspace(0,a2_nowork,n_c);
cstartemp=zeros(1,5);
for ih=1:n_h
    for iz=1:n_z
        [~,ih2_0] = min(abs(hgrid - h2_0(ih)));
        [~,ih2_l] = min(abs(hgrid - (h2_0(ih)+zgrid(iz)*e_l))) ;
        [~,ih2_h] = min(abs(hgrid - (h2_0(ih)+zgrid(iz)*e_h)));
        % n=0, e=0
        obj1=log(cgrid_nowork)+...
            discount*interp1(agrid,EV_2(:,ih2_0),a2_nowork-cgrid_nowork);
        [val1,ic1]=max(obj1);
        cstartemp(1)=cgrid_nowork(ic1);
        % n=0, e=e_l
        obj2=log(cgrid_nowork) - Kai_e*e_l +...
            discount*interp1(agrid,EV_2(:,ih2_l),a2_nowork-cgrid_nowork);
        [val2,ic2]=max(obj2);
        cstartemp(2)=cgrid_nowork(ic2);
        % n=0, e=e_h
        obj3=log(cgrid_nowork) - Kai_e*e_h + ...
            discount*interp1(agrid,EV_2(:,ih2_h),a2_nowork-cgrid_nowork);
        [val3,ic3]=max(obj3);
        cstartemp(3)=cgrid_nowork(ic3);

        cgrid_work=linspace(0,a2_work(iz,ih),n_c);
        % n=1, e=0
        obj4=log(cgrid_work) - Kai_n +...
            discount*interp1(agrid,EV_2(:,ih2_0),a2_work(iz,ih)-cgrid_work);
        [val4,ic4]=max(obj4);
        cstartemp(4)=cgrid_work(ic4);
        % n=1, e=e_l
        obj5=log(cgrid_work) - Kai_n - Kai_e*e_l +...
            discount*interp1(agrid,EV_2(:,ih2_l),a2_work(iz,ih)-cgrid_work);
        [val5,ic5]=max(obj5);
        cstartemp(5)=cgrid_work(ic5);

        % choose over 5 options
        [V1(iz,ih),choice(iz,ih)]=max([val1,val2,val3,val4,val5]);
        cstar(iz,ih)=cstartemp(choice(iz,ih));
    end
end

%% visualization
[z1,h1]=find(choice==1);
[z2,h2]=find(choice==2);
[z3,h3]=find(choice==3);
[z4,h4]=find(choice==4);
[z5,h5]=find(choice==5);
figure;
scatter(hgrid(h1),zgrid(z1),'k.') % n=0, e=0
hold on;
scatter(hgrid(h2),zgrid(z2),'g.') % n=0, e=e_l
scatter(hgrid(h3),zgrid(z3),'r.') % n=0, e=e_h
scatter(hgrid(h4),zgrid(z4),'y.') % n=1, e=0
scatter(hgrid(h5),zgrid(z5),'m.') % n=1, e=e_l
ylim([0,2])



%
futurewage=w*1/(1+r);
zupper_fast_L=((exp(Kai_e*e_l/(1+discount))*lambda*(exp(Kai_e*e_l/(1+discount))-1)^(-1)-1)*futurewage-(1+r)*avalue)/w
zlower_fast_L=(exp((Kai_n)/(1+discount))-1)*((1+r)*avalue+futurewage)/w
zupper_slow_L=((exp((Kai_n-Kai_e*e_h)/(1+discount))-1)*((1+r)*avalue+futurewage)+lambda*futurewage)/w
zupper_non_L=(exp((Kai_n)/(1+discount))-1)*((1+r)*avalue+futurewage*(1-lambda))/w

% lowerbound of lambda
lambda_lowb=(exp(Kai_e*e_h/(1+discount))-1)*((1+r)*avalue + futurewage)/futurewage;




