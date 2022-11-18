
package cnn;

import UTIL.Mat;


public class SoftMax {
    

        public float[][] weights;
    
        public float[][] input;


        public float[][] bias;
    
        public float[][] output;
    public SoftMax(int input, int output) {
        weights = Mat.m_scale(Mat.m_random(input, output), 1.0f / input);
        bias = Mat.v_zeros(10);
        String s="aaa";
        
    }

    public float[][] forward(float[][][] input) {
        float[][] in = Mat.m_flatten(input);  //1X1342
        output = new float[1][bias.length];    //1X10
        output = Mat.mm_add(Mat.mm_mult(in, weights), bias);
        //compute softmax probabilities.
        float[][] totals = Mat.v_exp(output);
        float inv_activation_sum = 1 / Mat.v_sum(totals);
        //cache input
        this.input = in;
        return Mat.v_scale(totals, inv_activation_sum);
    }

    public float[][][] backprop(float[][] d_L_d_out, float learning_rate) {
        //gradient of loss w.r.t. the total probabilites of the softmax layer.
        float[][] d_L_d_t = new float[1][d_L_d_out[0].length];
        //repeat softmax probability computations (caching can be used to avoid this.)
        float[][] t_exp = Mat.v_exp(output);
        float S = Mat.v_sum(t_exp);
        float[][] d_L_d_inputs=null;
        
        for (int i = 0; i < d_L_d_out[0].length; i++) {
            float grad = d_L_d_out[0][i];
            if (grad == 0) {
                continue;
            }
            //gradient of the output layer w.r.t. the totals [1] X [10]
            float[][] d_out_d_t = Mat.v_scale(t_exp, -t_exp[0][i] / (S * S));
            d_out_d_t[0][i] = t_exp[0][i] * (S - t_exp[0][i]) / (S * S);
            
            d_L_d_t = Mat.m_scale(d_out_d_t, grad); 
            //gradient of totals w.r.t weights -- [1342] X [1]
            float[][] d_t_d_weight = Mat.m_transpose(input);
            //gradient of totals w.r.t inputs -- [1342] X [10] 
            float[][] d_t_d_inputs = weights;
            float[][] d_L_d_w = Mat.mm_mult(d_t_d_weight, d_L_d_t);
            d_L_d_inputs = Mat.mm_mult(d_t_d_inputs, Mat.m_transpose(d_L_d_t));
            //gradient of loss w.r.t. bias
            float[][] d_L_d_b = d_L_d_t;
            weights = Mat.mm_add(Mat.m_scale(d_L_d_w, -learning_rate), weights);
            bias = Mat.mm_add(Mat.m_scale(d_L_d_b, -learning_rate), bias);
        }
        return Mat.reshape(Mat.m_transpose(d_L_d_inputs),8,13,13);
    }
}