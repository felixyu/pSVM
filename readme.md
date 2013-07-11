In the interest of time, I am releasing a preliminary version of pSVM 
(felixyu.org/pSVM.html).

Please cite the following paper:
Felix X. Yu; Dong Liu; Sanjiv Kumar; Tony Jebara; Shih-Fu Chang.
$\propto$SVM for learning with label proportions, ICML 2013

0. The package implements the (slow) kernel version of InvCal, alter-pSVM and conv-pSVM.
Linear versions, faster kernel versions based on libsvm (instead of gurobi) 
will be released later.

1. Setup
The code requires CVX, gurobi, and libsvm. 
http://cvxr.com/cvx/
http://www.gurobi.com/
http://www.csie.ntu.edu.tw/~cjlin/libsvm/
Please download the software in the above websites, and setup init.m accordingly.

2. Demo
Run demo_toy and demo_heart for two simple demos.
The toy data was the one used in the ICML paper.

3. The conv-pSVM was modified from LGMMC(http://lamda.nju.edu.cn/code_LGMMC.ashx). 
(Y.-F. Li, I. W. Tsang, J. T. Kwok, and Z.-H. Zhou. 
Tighter and convex maximum margin clustering. AISTATS'09)
Some of their code is included for the ease of the users. All the terms of the 
LGMMC package apply.

4. Please direct any questions to Felix Yu (yuxinnan@ee.columbia.edu).



