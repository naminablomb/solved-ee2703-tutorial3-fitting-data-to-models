Download Link: https://assignmentchef.com/product/solved-ee2703-tutorial3-fitting-data-to-models
<br>
This week’s Python assignment will focus on the following topics:

<ul>

 <li>Reading data from files and parsing them</li>

 <li>Analysing the data to extract information</li>

 <li>Study the effect of noise on the fitting process</li>

 <li>Plotting graphs</li>

</ul>

<h1>Plotting graphs</h1>

You can use matplotlib to plot sophisticated graphs in Python.

Here is a simple program to plot <em>J</em><sub>0</sub>(<em>x</em>) for 0 <em>&lt; x </em><em>&lt; </em>10. (type it in and see)

In [48]: from pylab import *; from scipy.special import *

In [49]: x=arange(0,10,.1)

In [50]: y=jv(0,x)

In [51]: plot(x,y)

In [52]: show()

The import keyword imports a module as already discussed. The “pylab” module is a super module that imports everything needed to make python seem to be like Matlab.

The actual code is four lines. One defines the x values. The second computes the Bessel function. The third plots the curve while the last line displays the graphic. The plot command has the following syntax: plot(x1,y1,style1,x2,y2,style2,…)

helping you to plot multiple curves in the same plot.

<h1>The Assignment</h1>

<ol>

 <li>The site has a python script generate_data.py. Download the script and run it to generate a set of data. The data is written out to a file whose name is fitting.dat.</li>

</ol>

1          h<em>* </em>1i≡

# script to generate data files for the least squares assignment from pylab import * import scipy.special as sp N=101 # no of data points

k=9          # no of sets of data with varying noise # generate the data points and add noise t=linspace(0,10,N)               # t vector y=1.05*sp.jn(2,t)-0.105*t # f(t) vector

Y=meshgrid(y,ones(k),indexing=’ij’)[0] # make k copies scl=logspace(-1,-3,k) # noise stdev n=dot(randn(N,k),diag(scl)) # generate k vectors yy=Y+n           # add noise to signal

# shadow plot plot(t,yy) xlabel(r’$t$’,size=20) ylabel(r’$f(t)+n$’,size=20) title(r’Plot of the data to be fitted’) grid(True) savetxt(“fitting.dat”,c_[t,yy]) # write out matrix to file show()

<ol start="2">

 <li>Load fitting.dat (look up loadtxt). The data consists of 10 columns. The first column is time, while the remaining columns are data. Extract these. 3. The data columns correspond to the function</li>

</ol>

<em>f</em>(<em>t</em>)= 1<em>.</em>05<em>J</em><sub>2</sub>(<em>t</em>)−0<em>.</em>105<em>t </em>+<em>n</em>(<em>t</em>)

with different amounts of noise added. What is noise? For us here, it is random fluctuations in the value due to many small random effects. The noise is given to be normally distributed, i.e., its probability distribution is given by

Pr

with <em>σ </em>given by sigma=logspace(-1,-3,9)

Plot the curves in Figure 0 and add labels to indicate the amount of noise in each. (Look up plot and legend.)

<ol start="4">

 <li>We want to fit a function to this data. Which function? The function has the same general shape asthe data but with unknown coefficients:</li>

</ol>

<em>g</em>(<em>t</em>;<em>A</em><em>,B</em>)= <em>AJ</em><sub>2</sub>(<em>t</em>)+<em>Bt</em>

Create a python function g(t,A,B) that computes <em>g</em>(<em>t</em>;<em>A</em><em>,B</em>) for given <em>A </em>and <em>B</em>. Plot it in Figure 0 for <em>A </em>= 1<em>.</em>05, <em>B </em>= −0<em>.</em>105 (this should be labelled as the true value.)

<ol start="5">

 <li>Generate a plot of the first column of data with error bars. Plot every 5th data item to make the plotreadable. Also plot the exact curve to see how much the data diverges.</li>

</ol>

This is not difficult. Suppose you know the standard deviation of your data and you have the data itself, you can plot the error bars with red dots using

errorbar(t,data,stdev,fmt=’ro’)

Here, ’t’ and ’data’ contain the data, while ’stdev’ contains <em>σ<sub>n </sub></em>for the noise. In order to show every fifth data point, you can instead use

errorbar(t[::5],data[::5],stdev,fmt=’ro’)

After plotting the errorbars, plot the exact curve using the function written in part 4, and annotate the graph.

<ol start="6">

 <li>For our problem, the values of <em>t </em>are discrete and known (from the datafile). Obtain <em>g</em>(<em>t</em><em>,A</em><em>,B</em>) as a column vector by creating a matrix equation:</li>

</ol>

<em>g </em>     <em>B          M p</em>

<em>J</em>2(<em>t<sub>m</sub></em>) <em>t<sub>m</sub></em>

Construct <em>M </em>and then generate the vector <em>M </em> and verify that it is equal to g(t,A0,B0).

How will you confirm that two vectors are equal?

Note: To construct a matrix out of column vectors, create the column vectors first and then use c_[…]. For instance, if you have two vectors x and y, use M=c_[x,y]

<ol start="7">

 <li>For <em>A </em>= 0<em>,</em>0<em>.</em>1<em>,…,</em>2 and <em>B </em>= −0<em>.</em>2<em>,</em>−0<em>.</em>19<em>,…,</em>0, for the data given in columns 1 and 2 of the file, compute</li>

</ol>

1 101                                                    <sub>2</sub>

<em>ε<sub>ij </sub></em>= 101 <em>k</em>∑=0(<em>f<sub>k </sub></em>−<em>g</em>(<em>t<sub>k</sub></em><em>,A<sub>i</sub></em><em>,B<sub>j</sub></em>))

This is known as the “mean squared error” between the data (<em>f<sub>k</sub></em>) and the assumed model. Use the first column of data as <em>f<sub>k </sub></em>for this part.

<ol start="8">

 <li>Plot a contour plot of <em>ε<sub>ij </sub></em>and see its structure. Does it have a minimum? Does it have several?</li>

 <li>Use the Python function lstsq from scipy.linalg to obtain the best estimate of <em>A </em>and <em>B</em>. The array you created in part 6 is what you need. This is sent to the least squares program.</li>

 <li>Repeat this with the different columns (i.e., columns 1 and <em>i</em>). Each column has the same function above, with a different amount of noise added as mentioned above. Plot the <em>error in the estimate </em>of <em>A </em>and <em>B </em>for different data files versus the noise <em>σ</em>. Is the <em>error in the estimate </em>growing linearly with the noise?</li>

 <li>Replot the above curves using loglog. Is the error varying linearly? What does this mean?</li>

</ol>

<h1>Linear Fitting to Data</h1>

Perhaps the most common engineering use of a computer is the modelling of real data. That is to say, some device, say a tachometer on an induction motor, or a temperature sensor of an oven or the current through a photodiode provides us with real-time data. This data is usually digitised very early on in the acquisition process to preserve the information, leaving us with time sequences,

If the device has been well engineered, and if the sensor is to be useful in diagnosing and controlling the device, we must also have a model for the acquired data:

<em>f</em>(<em>t</em>; <em>p</em><sub>1</sub><em>,…, p<sub>N</sub></em>)

For example, our model could be

<em>f</em>(<em>t</em>; <em>p</em><sub>1</sub><em>, p</em><sub>2</sub>)= <em>p</em><sub>1</sub>+ <em>p</em><sub>2</sub>sin(<em>πt</em><sup>2</sup>)

Our task is to accurately predict <em>p</em><sub>1</sub><em>,…, p<sub>N </sub></em>given the real-time data. This would allow us to say things like, “Third harmonic content in the load is reaching dangerous levels”.

The general problem of the estimation of model parameters is going to be tackled in many later courses.

Here we will tackle a very simple version of the problem. Suppose my model is “linear in the parameters”, i.e.,

<em>N</em>

<em>f</em>(<em>t</em>; <em>p</em><sub>1</sub><em>,…, p<sub>N</sub></em>)= ∑ <em>p<sub>i</sub>F<sub>i</sub></em>(<em>t</em>)

<em>i</em>=1

where <em>F<sub>i</sub></em>(<em>t</em>) are arbitrary functions of time. Our example above fits this model:

<table width="368">

 <tbody>

  <tr>

   <td colspan="2" width="368">                                                                  <em>p</em>1<em>, p</em>2 :  parameters to be fitted</td>

  </tr>

  <tr>

   <td width="331">                                                                         <em>F</em><sub>1 </sub>:     1<em>F</em><sub>2 </sub>:      sin(<em>πt</em><sup>2</sup>)For each measurement, we obtain an equation:</td>

   <td width="37"> </td>

  </tr>

  <tr>

   <td width="331"></td>

   <td width="37">= <em>x</em><sub>1</sub></td>

  </tr>

  <tr>

   <td width="331"></td>

   <td width="37">= <em>x</em><sub>2</sub></td>

  </tr>

  <tr>

   <td width="331">                                                                        <em>…                  …</em></td>

   <td width="37"><em>…</em></td>

  </tr>

 </tbody>

</table>

<em> x</em><em>M</em>

Clearly the general problem reduces to the inversion of a matrix problem:

<sup> </sup><em>F</em><sub>1</sub>(<em>t</em><sub>1</sub>)          <em>F</em><sub>2</sub>(<em>t</em><sub>1</sub>) <em>…        </em><em>F<sub>N</sub></em>

<sub> </sub><em>F</em><sub>1</sub>(<em>t</em><sub>2</sub>)          <em>F</em><sub>2</sub>(<em>t</em><sub>2</sub>) <em>…        </em><em>F<sub>N</sub></em>



However, the number of parameters, <em>N</em>, is usually far less than the number of observations, <em>M</em>. In the absence of measurement errors and noise, any non-singular <em>N </em>×<em>N </em>submatrix can be used to determine the coefficients <em>p<sub>i</sub></em>. When noise is present, what can we do? The matrix equation now becomes

<em>F </em>

where<em>~n </em>is the added noise.<em>~x</em><sub>0 </sub>is the (unknown) ideal measurement, while<em>~x </em>is the actual noisy measurement we make. We have to assume something about the noise, and we assume that it is as likely to be positive as negative (zero mean) and it has a standard deviation <em>σ<sub>n</sub></em>. Clearly the above equation cannot be exactly satisfied in the presence of noise, since the rank of F is <em>N </em>wheras the number of observations is <em>M</em>.

We also make a very important assumption, namely that the noise added to each observation <em>x<sub>i</sub></em>, namely <em>n<sub>i</sub></em>, is “independent” of the noise added to any other observation. Let us see where this gets us.

We wish to get the “best” guess for <em>~p</em>. For us, this means that we need to minimize the <em>L</em><sub>2 </sub>norm of the error. The error is given by <em>ε </em>= <em>F </em>·<em>~</em><em>p</em>−<em>~</em><em>x</em>

The norm of the error is given by

2

<em>i i</em>

This norm can we written in matrix form as

(<em>F </em>·<em>~</em><em>p</em>−<em>~</em><em>x</em>)<em><sup>T </sup></em>·(<em>F </em>·<em>~</em><em>p</em>−<em>~</em><em>x</em>)

which is what must be minimized. Writing out the terms

<em>~p<sup>T </sup>F<sup>T </sup>F</em>

Suppose the minimum is reached at some. Then near it, the above norm should be greater than that minimum. If we plotted the error, we expect the surface plot to look like a cup. The gradient of the error at the minimum should therefore be zero. Let us take the gradient of the expression for the norm. We write <em>F<sup>T</sup>F </em>= <em>M</em>, and write out the error in “Einstein notation”:

error = <em>p</em><em>iM</em><em>ijp</em><em>j </em>+<em>x</em><em>jx</em><em>j </em>− <em>p</em><em>iF</em><em>ijTx</em><em>j </em>−<em>x</em><em>iF</em><em>ijp</em><em>j</em>

Here we have suppressed the summation signs over <em>i </em>and <em>j</em>. If an index repeats in an expression, it is assumed to be summed over. Differentiating with respect to <em>p<sub>k</sub></em>, <em>and assuming that ∂ p<sub>i</sub></em><em>/∂ p<sub>k </sub></em>=<em>δ<sub>ik</sub>, </em>we get

<table width="311">

 <tbody>

  <tr>

   <td width="49"><em>∂</em>error<em>∂ p<sub>k</sub></em></td>

   <td width="24">=</td>

   <td width="239"><em>δ</em><em>kiM</em><em>ijp</em><em>j </em>+ <em>p</em><em>iM</em><em>ijδ</em><em>jk </em>−<em>δ</em><em>kiF</em><em>ijTx</em><em>j </em>−<em>x</em><em>iF</em><em>ijδ</em><em>jk </em>= 0</td>

  </tr>

  <tr>

   <td width="49"> </td>

   <td width="24">=</td>

   <td width="239"><em>M</em><em>kjp</em><em>j </em>+ <em>p</em><em>iM</em><em>ik </em>−<em>F</em><em>kjTx</em><em>j </em>−<em>x</em><em>iF</em><em>ik</em></td>

  </tr>

 </tbody>

</table>

= ∑ <em>M</em><em>kj </em><em>F</em><em>kjTx</em><em>j</em>

<em>j                                                          j</em>

Now the matrix <em>M </em>is symmetric (just take the transpose and see). So the equation finally becomes, written as a vector expression

∇error(<em>~</em><em>p</em>)= 2 <em>F<sup>T</sup>F </em>i.e.,

<em> F<sup>T</sup>F</em>

This result is very famous. It is so commonly used that scientific packages have the operation as a built in command. In Python, it is a library function called lstsq:

from scipy.linalg import lstsq p,resid,rank,sig=lstsq(F,x)

where p returns the best fit, and sig, resid and rank return information about the process. In Scilab or Matlab, it take the form:

p0 = Fx;

Here F is the coefficient matrix and x is the vector of observations. When we write this, Scilab is actually calculating

p0=inv(F’*F)*F’*x;

What would happen if the inverse did not exist? This can happen if the number of observations are too few, or if the vectors <em>f<sub>i</sub></em>(<em>x<sub>j</sub></em>) (i.e., the columns of <em>F</em>) are linearly dependent. Both situations are the user’s fault. He should have a model with linearly independent terms (else just merge them together). And he should have enough measurements.

p0 obtained from the above formula is a prediction of the exact parameters. How accurate can we expect p0 to be? If x were x0 we should recover the exact answer limited only by computer accuracy. But when noise is present, the estimate is approximate. The more noise, the more approximate.

Where did we use all those properties of the noise? We assumed that the noise in different measurements was the same when we defined the error norm. Suppose some measurements are more noisy. Then we should give less importance to those measurements, i.e., weight them less. That would change the formula we just derived. If the noise samples were not independent, we would need equations to explain just how they depended on each other. That too changes the formula. Ofcourse, if the model is more complicated things get really difficult. For example, so simple a problem as estimating the <em>frequency </em>of the following model

<em>y </em>= <em>A</em>sin<em>ωt </em>+<em>B</em>cos<em>ωt </em>+<em>n</em>

is an extremely nontrivial problem. That is because it is no longer a “linear” estimation problem. Such problems are discussed in advanced courses like <em>Estimation Theory</em>. The simpler problems of correlated, non-uniform noise will be discussed in <em>Digital Communication </em>since that theory is needed for a cell phone to estimate the signal sent by the tower.

In this assignment (and in this course), we assume independent, uniform error that is normally distributed. For that noise, as mentioned above, Python already provides the answer:

p=lstsq(F,x)

Even with all these simplifications, the problem can become ill posed. Let us look at the solution again:

<em> F<sup>T</sup>F</em>

Note that we have to invert <em>F<sup>T</sup>F</em>. This is the coefficient matrix. For instance, if we were trying to estimate <em>A </em>and <em>B </em>of the following model:

<em>y </em>= <em>A</em>sin<em>ω</em><sub>0</sub><em>t </em>+<em>B</em>cos<em>ω</em><sub>0</sub><em>t</em>

(not the frequency – that makes it a nonlinear problem), the matrix <em>F </em>becomes

sin<em>ω</em>

<em>F </em>=        <em>…                …       </em>

sin<em>ω</em>0<em>t<sub>n        </sub></em>cos<em>ω</em>0<em>t<sub>n</sub></em>

Hence, <em>F<sup>T</sup>F </em>is a 2 by 2 matrix with the following elements

<em>F<sup>T </sup>F </em>

Whether this matrix is invertible depends only on the functions sin<em>ω</em><sub>0</sub><em>t </em>and cos<em>ω</em><sub>0</sub><em>t </em>and the times at which they are sampled. For instance, if the <em>t<sub>k </sub></em>= 2<em>kπ</em>, the matrix becomes

<em>F<sup>T</sup>F </em>

since sin<em>ω</em><sub>0</sub><em>t<sub>k </sub></em>≡ 0 for all the measurement times. Clearly this is not an invertible matrix, even though the <em>functions </em>are independent. Sometimes the functions are “nearly” dependent. The inverse exists, but it cannot be accurately calculated. To characterise this, when an inverse is calculated, the “condition number” is also returned. A poor condition number means that the inversion is not to be depended on and the estimation is going to be poor.

We will not use these ideas in this lab. Here we will do a very simple problem only, to study how the amount of noise affects the quality of the estimate. We will also study what happens if we use the wrong model.