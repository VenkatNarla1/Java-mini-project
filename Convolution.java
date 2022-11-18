package cnn;

import UTIL.Mat;


public class Convolution {

        public float[][][] filters; // shape --> [3] X [8] X [8]
    public float[][] convolve3x3(float[][] image, float[][] filter) {
        input=image;
        float[][] result = new float[image.length - 2][image[0].length - 2];
        //loop through
        for (int i = 1; i < image.length - 2; i++) {
            for (int j = 1; j < image[0].length - 2; j++) {
                float[][] conv_region = Mat.m_sub(image, i - 1, i + 1, j - 1, j + 1);
                result[i][j] = Mat.mm_elsum(conv_region, filter);
            }
        }
        return result;
    }


    public float[][][] forward(float[][] image, float[][][] filter) {
        filters=filter; // 8 X 3 X 3
        float[][][] result = new float[8][26][26];
        for (int k = 0; k < filters.length; k++) {
            float[][] res = convolve3x3(image, filters[k]);
            result[k] = res;
        }
        return result;
    }
    

    public void backprop(float[][][] d_L_d_out,float learning_rate){
        float[][][] d_L_d_filters= new float[filters.length][filters[0].length][filters[0][0].length];

        for(int i=1;i<input.length-2;i++){
            for(int j=1;j<input[0].length-2;j++){
                for(int k=0;k<filters.length;k++){
                    //get a 3X3 region of the matrix
                    float[][] region=Mat.m_sub(input,  i - 1, i + 1, j - 1, j + 1);
                    d_L_d_filters[k]=Mat.mm_add(d_L_d_filters[k], Mat.m_scale(region,d_L_d_out[k][i-1][j-1]));
                }
            }
        }
        
        for(int m=0;m<filters.length;m++){
            filters[m]= Mat.mm_add(filters[m], Mat.m_scale(d_L_d_filters[m],-learning_rate));
        }  
    }
}