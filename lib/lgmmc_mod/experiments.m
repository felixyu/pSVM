addpath('G:\Dropbox\felix_codebase\matlab_lib\svm\libsvm\matlab')
clear
%%%%%%%%%%%%%%%%%%%%%%%%
%%%% load data
%%%%%%%%%%%%%%%%%%%%%%%%
disp('loading the data....');
filename = 'echocardiogram';
load(filename);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% preprocessing
disp('preprocessing....');
tmp = size(content,1);
x = content(1:tmp-1,1:size_n);
y = content(tmp,1:size_n);
y(logical(y == 2)) = -1;
x = x';
y = y';
clear content;
%%%%%%%%%%%%%%%%%%%%%%%%%

[n, dim] = size(x);
%%%%%%%%%%%%%%%%%%%%%%%%
%%%% call LG_GMMC
%%%%%%%%%%%%%%%%%%%%%%%%
addpath('LG_MMC_v0.1');
addpath('libsvm-mat-2.83-1');
disp('lg_mmc............');

C = [1];
ratio = [4];

opt.name = filename;
opt.iteration = 100;
%%%%%%%%%%%%%%%%%%%%%%%
% balance constraint
%%%%%%%%%%%%%%%%%%%%%%%
if abs(sum(y))/n < 0.2
    opt.ep  = floor(n*0.03);
else
    opt.ep  = floor(n*0.3);
end

l_err = 1;
g_err = 1;
for c_i = 1:length(C)
    opt.C = C(c_i);
    for bias = 0:1
        opt.bias = bias;
        for gaussian = 0:1
            opt.gaussian = gaussian;
            if gaussian == 1
                for ra = 1:length(ratio)
                    opt.ratio = ratio(ra);
                    [pre_y,ct] = lgmmc(x',opt);
                    m = size(find(pre_y' == y),1);
                    acc = max(m,n-m)/n;
                    if acc > 1 - g_err
                        g_err = 1 - acc;
                        g_time = ct;
                    end
                end
            else

                [pre_y,ct] = lgmmc(x',opt);
                m = size(find(pre_y' == y),1);
                acc = max(m,n-m)/n;
                if acc > 1 - l_err
                    l_err = 1 - acc;
                    l_time = ct;
                end               
            end
        end
    end
end

disp('Output the accuract and the time..');
disp(['LG_MMC_ACCURACY (linear kernel): ' num2str( 1 - l_err)]);
disp(['LG_MMC_CPUTIME (linear kernel): ' num2str(l_time)]);
disp(['LG_MMC_ACCURACY (rbf kernel): ' num2str( 1 - g_err)]);
disp(['LG_MMC_CPUTIME (rbf kernel): ' num2str(g_time)]);





