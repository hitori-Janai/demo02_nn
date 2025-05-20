https://victorzhou.com/blog/intro-to-neural-networks/#code-mse-loss


This section explains the calculation of the partial derivative of the output with respect to the weight w5 in the neural network.\n\nThe output `o1` is calculated as `sigmoid(sum_o1)`, where `sum_o1` is the weighted sum of inputs to the output neuron: `sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3`.\n\nTo find how a change in `w5` affects the output $o_1$, we use the chain rule:


$$o_1 == y\_pred$$

$$\frac{\partial o_1}{\partial w_5} = \frac{\partial o_1}{\partial {sum\\_o_1}} \cdot \frac{\partial {sum\\_o_1}}{\partial w_5}$$

 > 1.  $\frac{\partial o_1}{\partial {sum\\_o_1}}$: Since $o_1 = {sigmoid}({sum\\_o_1})$, 
 the derivative of the sigmoid function with respect to its input (`sum_o1`) is `deriv_sigmoid(sum_o1)`.\n\n2.  \\(\\frac{\partial \\text{sum\\_o1}}{\partial w_5}\dana Since \\
 
 $$sum\\_o_1 = self.w_5 \cdot h_1 + self.w_6 \cdot h_2 + self.b_3$$
 , and we are taking the partial derivative with respect to \\(w_5\\), we treat \\(h_1\\), \\(w_6\\), \\(h_2\\), and \\(b_3\\) as constants. The derivative of \\(self.w_5 \\cdot h_1\\) with respect to \\(w_5\\) is \\(h_1\\), and the derivatives of the other terms are 0. So, \\(\\frac{\partial \\text{sum\\_o1}}{\partial w_5} = h_1\\).\n\nCombining these two parts using the chain rule:\n\n\\[ \
 
 $$\frac{\partial o_1}{\partial w_5} = {deriv\\_sigmoid}({sum\\_o_1}) \cdot h_1 $$
 
 The code line `d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)` implements this exact calculation, where `d_ypred_d_w5` represents \\(\\frac{\partial o_1}{\partial w_5}\\). This value tells us how sensitive the network\'s output is to changes in the weight `w5`. This is then used, along with the derivative of the loss with respect to the output (`d_L_d_ypred`), to calculate the gradient \\(\\frac{\partial L}{\partial w_5}\\) and update `w5` during the training process.
